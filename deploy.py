import glob
import visdom
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm
from scipy.misc import imread
from skimage.transform import resize, rescale
from skimage.morphology import remove_small_objects, remove_small_holes, label

from unet import UNetVanilla, UNetAtrous
from inputs import DeployRectusFemoris

vis = visdom.Visdom()


def blend(image, label, alpha=0.5):
    """"Simulate colormap `jet`."""

    r = (label + 0.1) * 255 * alpha
    b = (image + image.mean()) * (1 - alpha)
    g = np.minimum(r, b)
    rgb = np.dstack([r, g, b] + image * 0.3)
    # vis.image(rgb.transpose(2, 0, 1))
    return rgb.astype(np.uint8)


def save_label_area():
    lable_dict = {p.replace('label', 'image'): np.sum((imread(p) > 127).astype(np.uint8))
                  for p in glob.glob('data/*_label/*.bmp')}
    np.save('deploy/label_areas.npy', lable_dict)


def slice_dict(d, s):
    def _digit(_s):
        return int(''.join(list(filter(str.isdigit, _s))))

    return {_digit(k.split('/')[-1]): v for k, v in d.items() if k.startswith(s)}


class DeployModel:
    def __init__(self, model=UNetAtrous(), model_name='UNetAtrous', deploy_mode='total',
                 batch_size=24, image_size=(256, 256), device_id=0):
        torch.cuda.set_device(device_id)

        self.model = model.cuda().eval()
        self.model_name = model_name
        self.batch_size = batch_size
        self.image_size = image_size
        self.deploy_mode = deploy_mode

    @staticmethod
    def _post_process(im, mask):
        # Resize
        mask = (resize(mask, im.shape, preserve_range=True) > 0.5).astype(np.uint8)
        # Remove discrete pixels
        bw = remove_small_objects(label(mask == 1), min_size=4096, connectivity=2)
        bw = remove_small_holes(bw, min_size=4096, connectivity=2)
        return bw.astype(np.uint8)

    def deploy(self):
        best_path = 'checkpoints/{}/{}_best.pth'.format(self.model_name, self.model_name)
        best_model = torch.load(best_path)
        print('===> Loading model from {}...'.format(best_path))

        self.plot_statistic(self.model_name)
        print('===> Look at statistics!')

        self.model.load_state_dict(best_model)

        submit_dataset = DeployRectusFemoris(mode=self.deploy_mode, image_size=self.image_size)
        dataloader = DataLoader(submit_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        areas = {}
        for i_batch, (x_batch, path_batch) in tqdm(enumerate(dataloader),
                                                   total=len(submit_dataset) // self.batch_size, unit='batch'):
            x_batch = Variable(x_batch, volatile=True).float().cuda()
            output = self.model(x_batch)
            if isinstance(output, (tuple, list)):
                output = output[0]

            pred = output.data.max(1)[1]
            pred = pred.cpu().numpy()

            for j, mask in enumerate(pred):
                path = path_batch[j]
                im = imread(path, mode='L')
                mask = self._post_process(im, mask)
                areas[path] = np.sum(mask)

            np.save('deploy/' + self.model_name + '_areas.npy', areas)

    def plot_statistic(self):
        res = np.load('results/{}/results_dict.npy'.format(self.model_name)).item()
        res = {k: v[v > 0] for k, v in res.items()}
        X = range(len(res['train_acc']))

        titles = ['Dice Overlap', 'Accuracy', 'Loss']
        keys = ['dice_overlap', 'acc', 'loss']
        for k, title in list(zip(keys, titles)):
            vis.line(X=np.column_stack((X, X)),
                     Y=np.column_stack((res['train_' + k], res['val_' + k])),
                     opts=dict(legend=['train', 'val'], title=title))

    def plot_seperate_area(self, mark_label=False):
        all_area = np.load('deploy/' + self.model_name + '_areas.npy').item()
        nested_all_dict = {c.split('_')[1]: slice_dict(all_area, c) for c in glob.glob('data/*_image')}

        if mark_label:
            label_area = np.load('deploy/label_areas.npy').item()
            nested_label_dict = {c.split('_')[1]: slice_dict(label_area, c) for c in glob.glob('data/*_image')}

        for cls, cls_dict in nested_all_dict.items():
            idx, area = [np.array(e).tolist() for e in zip(*sorted(cls_dict.items()))]

            # Plot
            win = cls
            data = [{'x': idx, 'y': area, 'mode': "lines", 'name': 'infer', 'type': 'line'}]
            layout = {'title': cls, 'xaxis': {'title': 'index'}, 'yaxis': {'title': 'area'}}

            if mark_label:
                label_idx, label_area = [np.array(e).tolist() for e in zip(*sorted(nested_label_dict[cls].items()))]
                data.append({'x': label_idx, 'y': label_area, 'type': 'scatter', 'mode': 'markers'})

            vis._send({'data': data, 'win': win, 'layout': layout})


if __name__ == '__main__':
    dm = DeployModel()
    dm.plot_seperate_area(mark_label=True)
