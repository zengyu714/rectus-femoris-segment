from bluntools.layers import *


class UNetAtrous(nn.Module):
    def __init__(self):
        super(UNetAtrous, self).__init__()
        self.in_fit = nn.Sequential(ConvBNReLU(1, 16), ConvBNReLU(16, 16))

        self.lhs_1x = DownBlock(16, 16)
        self.lhs_2x = DownBlock(16, 32)
        self.lhs_4x = DownBlock(32, 64)
        self.lhs_8x = DownBlock(64, 128)
        self.lhs_16x = DownBlock(128, 256)

        # 16x
        self.atrous_1 = AtrousBlock(128, 128)
        self.atrous_2 = AtrousBlock(128, 128)
        self.atrous_3 = AtrousBlock(128, 128)
        self.atrous_4 = AtrousBlock(128, 128)

        self.rhs_16x = UpBlock(128, 128)
        self.rhs_8x = UpBlock(128, 64)
        self.rhs_4x = UpBlock(64, 32)
        self.rhs_2x = UpBlock(32, 16)
        self.rhs_1x = UpBlock(16, 16)

        self.classify = nn.Conv2d(16, 2, kernel_size=1)

    def forward(self, x):
        lhs_1x, pool = self.lhs_1x(self.in_fit(x))
        lhs_2x, pool = self.lhs_2x(pool)
        lhs_4x, pool = self.lhs_4x(pool)
        lhs_8x, pool = self.lhs_8x(pool)
        lhs_16x, _ = self.lhs_16x(pool)

        atrous_1 = self.atrous_1(lhs_16x)
        atrous_2 = self.atrous_2(atrous_1) + lhs_16x
        atrous_3 = self.atrous_3(atrous_2)
        atrous_4 = self.atrous_4(atrous_3) + atrous_2

        _, up = self.rhs_16x(atrous_4, pool)
        _, up = self.rhs_8x(lhs_8x, pool)
        _, up = self.rhs_4x(lhs_4x, up)
        _, up = self.rhs_2x(lhs_2x, up)
        rhs_1x, _ = self.rhs_1x(lhs_1x, up)

        return self.classify(rhs_1x)


class UNetVanilla(nn.Module):
    """Input: [batch_size, 3, 256, 256]"""

    def __init__(self):
        super(UNetVanilla, self).__init__()
        # MaxPool
        self.lhs_1x = nn.Sequential(ConvBNReLU(1, 16), ConvBNReLU(16, 16))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvBNReLU(16, 32), ConvBNReLU(32, 32))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvBNReLU(32, 64), ConvBNReLU(64, 64))
        self.lhs_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvBNReLU(64, 128), ConvBNReLU(128, 128))
        self.lhs_16x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvBNReLU(128, 256), ConvBNReLU(256, 256))

        # Bottom 32x <==> [batch_size, 3, 8, 8]
        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvBNReLU(256, 512), ConvBNReLU(512, 512))

        # BilinearUp
        self.rhs_16x = UpConcat(512, 256)
        self.rhs_8x = UpConcat(256, 128)
        self.rhs_4x = UpConcat(128, 64)
        self.rhs_2x = UpConcat(64, 32)
        self.rhs_1x = UpConcat(32, 16)

        # Classify
        self.classify = nn.Conv2d(16, 2, kernel_size=1)

    def forward(self, x):
        lhs_1x = self.lhs_1x(x)
        lhs_2x = self.lhs_2x(lhs_1x)
        lhs_4x = self.lhs_4x(lhs_2x)
        lhs_8x = self.lhs_8x(lhs_4x)
        lhs_16x = self.lhs_16x(lhs_8x)

        bottom = self.bottom(lhs_16x)

        rhs_16x = self.rhs_16x(lhs_16x, bottom)
        rhs_8x = self.rhs_8x(lhs_8x, rhs_16x)
        rhs_4x = self.rhs_4x(lhs_4x, rhs_8x)
        rhs_2x = self.rhs_2x(lhs_2x, rhs_4x)
        rhs_1x = self.rhs_1x(lhs_1x, rhs_2x)

        return self.classify(rhs_1x)
