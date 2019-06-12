import torch
import torch.nn as nn
import torch.nn.functional as F
class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        # because class 0 is dominent
        self.down1 = downStep(1, 64, withPool=False)
        self.down2 = downStep(64, 128)
        self.down3 = downStep(128, 256)
        self.down4 = downStep(256, 512)
        self.bottom = downStep(512, 1024)

        self.up4 = upStep(1024, 512)
        self.up3 = upStep(512, 256)
        self.up2 = upStep(256, 128)
        self.up1 = upStep(128, 64, withReLU=False)
        self.output = nn.Conv2d(64, n_classes, 1)

        def init_with_xavier(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

        self.down1.apply(init_with_xavier)
        self.down2.apply(init_with_xavier)
        self.down3.apply(init_with_xavier)
        self.down4.apply(init_with_xavier)
        self.bottom.apply(init_with_xavier)
        self.up4.apply(init_with_xavier)
        self.up3.apply(init_with_xavier)
        self.up2.apply(init_with_xavier)
        self.up1.apply(init_with_xavier)
        self.output.apply(init_with_xavier)

    def forward(self, x):
        x_down1 = self.down1(x)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)
        x_down4 = self.down4(x_down3)
        x_bottom = self.bottom(x_down4)

        x_up4 = self.up4(x_bottom, x_down4)
        x_up3 = self.up3(x_up4, x_down3)
        x_up2 = self.up2(x_up3, x_down2)
        x_up1 = self.up1(x_up2, x_down1)
        x = self.output(x_up1)

        return x

class downStep(nn.Module):
    def __init__(self, inC, outC, withPool=True):
        super(downStep, self).__init__()
        self.withPool = withPool
        self.downPool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(inC, outC, 3)
        self.conv2 = nn.Conv2d(outC, outC, 3)

        self.batchNorm = nn.BatchNorm2d(outC)

    def forward(self, x):
        x = self.downPool(x) if self.withPool else x
        x = F.relu(self.conv1(x))
        x = self.batchNorm(x)
        x = F.relu(self.conv2(x))
        x = self.batchNorm(x)
        return x

class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        self.withReLU = withReLU
        self.convT = nn.ConvTranspose2d(inC, outC, 2, stride=2)

        self.conv1 = nn.Conv2d(inC, outC, 3)
        self.conv2 = nn.Conv2d(outC, outC, 3)

        self.batchNorm = nn.BatchNorm2d(outC)
    def forward(self, x, x_down):
        # up conv : N,C,H,W
        x = self.convT(x)

        # concatenate down and up
        diff_h = (x_down.shape[2] - x.shape[2])//2
        diff_w = (x_down.shape[3] - x.shape[3])//2
        x_cat = x_down[:, :, diff_h:x_down.shape[2]-diff_h, diff_w:x_down.shape[3]-diff_w]
        x = torch.cat([x, x_cat], dim=1)
        # two conv
        if(self.withReLU):
            x = F.relu(self.conv1(x))
            x = self.batchNorm(x)
            x = F.relu(self.conv2(x))
            x = self.batchNorm(x)
        else:
            x = self.conv2(self.conv1(x))

        return x

if __name__=="__main__":
    img_t = torch.randn(2, 1, 572, 572)
    print("image size: ", img_t.shape)
    net = UNet(2)
    x = net.forward(img_t)
