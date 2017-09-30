import os
import time
import argparse
import datetime
import visdom
import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from unet import UNetAtrous, UNetVanilla, UNetShortCut, UNetMulLoss
from inputs import RectusFemorisDataset

from bluntools.easy_visdom import EasyVisdom
from bluntools.layers import weights_init, active_flatten, avg_class_weight, get_statistic, pre_visdom, \
    DiceLoss, NLLLoss, multi_size
from bluntools.checkpoints import load_checkpoints, save_checkpoints
from bluntools.lr_scheduler import auto_lr_scheduler, step_lr_scheduler

parser = argparse.ArgumentParser(description='Carvana Image Masking Challenge')
parser.add_argument('-n', '--from-scratch', action='store_true',
                    help='train model from scratch')
parser.add_argument('-g', '--cuda-device', type=int, default=0,
                    help='choose which gpu to use')
parser.add_argument('-a', '--architecture', type=int, default=0,
                    help='see details in code')


class Configuration:
    def __init__(self, prefix, cuda_device, from_scratch):
        self.cuda = torch.cuda.is_available()
        self.cuda_device = cuda_device

        self.batch_size = 16
        self.epochs = 20
        self.augment_size = 100
        self.loss_size = 1
        self.loss_func = 'NLLLoss(conf.class_weight)(outputs, targets) + DiceLoss()(probs, trues)'
        self.learning_rate = 1e-4
        self.seed = 714
        self.threads = 4
        self.resume_step = -1
        self.show_interval = 10
        self.from_scratch = from_scratch
        self.prefix = prefix
        self.result_dir = None
        self.checkpoint_dir = None

    def generate_dirs(self):
        self.result_dir = os.path.join('./results', self.prefix)
        self.checkpoint_dir = os.path.join('./checkpoints', self.prefix)
        [os.makedirs(d) for d in [self.result_dir, self.checkpoint_dir] if not os.path.exists(d)]


args = parser.parse_args()
conf = Configuration(prefix='Untitled', cuda_device=args.cuda_device, from_scratch=args.from_scratch)

# Instantiate plot
# ----------------------------------------------------------------------
vis = visdom.Visdom()


def main():
    # GPU (Default) configuration
    # --------------------------------------------------------------------------------------------------------
    assert conf.cuda, 'Use GPUs default, check cuda available'
    torch.cuda.manual_seed(conf.seed)
    torch.cuda.set_device(conf.cuda_device)
    print('===> Current GPU device is', torch.cuda.current_device())

    # Set models
    # --------------------------------------------------------------------------------------------------------
    if args.architecture == 0:
        conf.prefix = 'UNetShortCut'
        model = UNetShortCut()
    elif args.architecture == 1:
        conf.prefix = 'UNetShortCut_NLL'
        conf.loss_func = 'NLLLoss(conf.class_weight)(outputs, targets)'
        model = UNetAtrous()
    elif args.architecture == 2:
        conf.prefix = 'UNetShortCut_Dice'
        conf.loss_func = 'DiceLoss()(probs, trues)'
        model = UNetAtrous()
    elif args.architecture == 3:
        conf.prefix = 'UNetVanilla'
        model = UNetVanilla()

    conf.generate_dirs()

    model = model.cuda()
    print('===> Building {}...'.format(conf.prefix))
    print('---> Batch size: {}'.format(conf.batch_size))

    start_i = 1
    total_i = conf.epochs * conf.augment_size

    # Dataset loader
    # --------------------------------------------------------------------------------------------------------
    training_data_loader = DataLoader(dataset=RectusFemorisDataset(mode='train'),
                                      num_workers=conf.threads, batch_size=conf.batch_size, shuffle=True)

    test_data_loader = DataLoader(dataset=RectusFemorisDataset(mode='test'),
                                  num_workers=conf.threads, batch_size=conf.batch_size, shuffle=True)

    # Weights
    # ----------------------------------------------------------------------------------------------------
    if conf.from_scratch:
        model.apply(weights_init)
    else:
        start_i = load_checkpoints(model, conf.checkpoint_dir, conf.resume_step, conf.prefix)
    print('===> Begin training at epoch: {}'.format(start_i))

    # Optimizer
    # ----------------------------------------------------------------------------------------------------
    optimizer = optim.RMSprop(model.parameters(), lr=conf.learning_rate)
    scheduler = auto_lr_scheduler(optimizer, patience=70, cooldown=30, verbose=1, min_lr=1e-8)

    print('===> Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    print('---> Initial learning rate: {:.0e} '.format(conf.learning_rate))

    # Loss
    # ----------------------------------------------------------------------------------------------------
    class_weight = avg_class_weight(test_data_loader).cuda()
    conf.class_weight = class_weight
    print('---> Rescaled class weights: {}'.format(class_weight.cpu().numpy().T))
    print('---> Loss function: {}'.format(conf.loss_func))

    # Visdom
    # ----------------------------------------------------------------------------------------------------
    ev = EasyVisdom(conf.from_scratch,
                    total_i,
                    start_i=start_i,
                    mode=['train', 'val'],
                    stats=['loss', 'acc', 'dice_overlap'],
                    results_dir=conf.result_dir,
                    env=conf.prefix)

    def train():
        epoch_loss, epoch_acc, epoch_overlap = np.zeros(3)

        # Sets the module in training mode, only on modules such as Dropout or BatchNorm.
        model.train()

        for partial_epoch, (image, label) in enumerate(training_data_loader, 1):
            image = Variable(image).float().cuda()

            outputs, targets = model(image), multi_size(label, size=conf.loss_size)  # 2D cuda Variable
            preds, trues, probs = active_flatten(outputs, targets)  # 1D

            loss = eval(conf.loss_func)

            pred, true = preds[0], trues[0]  # original size prediction
            accuracy, overlap = get_statistic(pred, true)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            epoch_acc += accuracy
            epoch_loss += loss.data[0]
            epoch_overlap += overlap

        avg_loss, avg_acc, avg_dice = np.array([epoch_loss, epoch_acc, epoch_overlap]) / partial_epoch
        print_format = [avg_loss, avg_acc, avg_dice]

        print('===> Training step {} ({}/{})\t'.format(i, i // conf.augment_size + 1, conf.epochs),
              'Loss: {:.5f}   Accuracy: {:.5f}  |   Dice Overlap: {:.5f}'.format(*print_format))

        return (avg_loss, avg_acc, avg_dice,
                *pre_visdom(image, label, pred))

    def validate():
        epoch_loss, epoch_acc, epoch_overlap = np.zeros(3)

        # Sets the module in training mode.
        model.eval()

        for partial_epoch, (image, label) in enumerate(test_data_loader, 1):
            image = Variable(image, volatile=True).float().cuda()

            outputs, targets = model(image), multi_size(label, size=conf.loss_size)  # 2D cuda Variable
            preds, trues, probs = active_flatten(outputs, targets)  # 1D
            loss = eval(conf.loss_func)

            pred, true = preds[0], trues[0]  # original size prediction
            accuracy, overlap = get_statistic(pred, true)

            epoch_acc += accuracy
            epoch_loss += loss.data[0]
            epoch_overlap += overlap

        avg_loss, avg_acc, avg_dice = np.array(
                [epoch_loss, epoch_acc, epoch_overlap]) / partial_epoch
        print_format = [avg_loss, avg_acc, avg_dice]
        print('===> ===> Validation Performance', '-' * 60,
              'Loss: {:.5f}   Accuracy: {:.5f}  |  Dice Overlap: {:.5f}'.format(*print_format))

        return (avg_loss, avg_acc, avg_dice,
                *pre_visdom(image, label, pred))

    best_result = 0.0
    prog_start_time = datetime.datetime.now()
    elapsed_time, total_time = 0.0, 0.0
    for i in range(start_i, total_i + 1):
        start_time = time.process_time()

        *train_results, train_image, train_label, train_pred = train()  # `results`: [loss, acc, dice]
        *val_results, val_image, val_label, val_pred = validate()

        elapsed_time += time.process_time() - start_time
        total_time += time.process_time() - start_time

        # ReduceLROnPlateau monitors: validate loss
        scheduler.step(val_results[0])

        # Visualize - scalar
        ev.vis_scalar(i, train_results, val_results)

        # Visualize - images
        ev.vis_images(i,
                      show_interval=1,
                      im_titles=['input', 'label', 'prediction'],
                      train_images=[train_image, train_label, train_pred],
                      val_images=[val_image, val_label, val_pred])

        # Save checkpoints
        if i % conf.show_interval == 0:
            temp_result = val_results[-1] + train_results[-1]
            is_best = temp_result > best_result
            best_result = max(temp_result, best_result)

            save_checkpoints(model, conf.checkpoint_dir, i, conf.prefix, is_best=is_best)
            np.save(os.path.join(conf.result_dir, 'results_dict.npy'), ev.results_dict)

            avg_time = elapsed_time / conf.show_interval
            est_time = divmod((total_i - i) * avg_time, 60)
            sum_time = divmod(total_time, 60)
            print('Average consumed time: {:.3f} s/epoch\n'
                  'Estimated remaining time: {:.0f}m {:.0f}s   Total time: {:.0f}m {:.0f}s\n'
                  '[start from: {} -- current time: {}]'
                  .format(avg_time, *est_time, *sum_time, prog_start_time, datetime.datetime.now()))
            elapsed_time = 0.0


if __name__ == '__main__':
    main()
