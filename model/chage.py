import torch
import torch.nn as nn
import torch.nn.functional as F
from module.deablock import DEABlockTrain, DEBlockTrain
from module.fusion import CGAFusion


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class REM_Net(nn.Module):

    def __init__(self, base_dim=32):
        super(REM_Net, self).__init__()

        # -------------Encoder--------------
        self.down1 = nn.Conv2d(5, base_dim, kernel_size=3, stride=1, padding=1)
        self.down2 = nn.Sequential(nn.Conv2d(base_dim, base_dim * 2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.down3 = nn.Sequential(nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))

        self.down4 = nn.Sequential(nn.Conv2d(base_dim * 4, base_dim * 8, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))

        self.down_level1_block1 = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level1_block2 = DEBlockTrain(default_conv, base_dim, 3)

        self.up_level1_block1 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block2 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block3 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block4 = DEBlockTrain(default_conv, base_dim, 3)

        self.fe_level_2 = nn.Conv2d(in_channels=base_dim * 2, out_channels=base_dim * 2, kernel_size=3, stride=1, padding=1)
        self.down_level2_block1 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.down_level2_block2 = DEBlockTrain(default_conv, base_dim * 2, 3)

        self.up_level2_block1 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level2_block2 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level2_block3 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level2_block4 = DEBlockTrain(default_conv, base_dim, 3)

        # level3
        self.fe_level_3 = nn.Conv2d(in_channels=base_dim * 4, out_channels=base_dim * 4, kernel_size=3, stride=1, padding=1)
        self.down_level3_block1 = DEBlockTrain(default_conv, base_dim * 4, 3)
        self.down_level3_block2 = DEBlockTrain(default_conv, base_dim * 4, 3)

        self.up_level3_block1 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level3_block2 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level3_block3 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level3_block4 = DEBlockTrain(default_conv, base_dim * 2, 3)

        # level4
        self.fe_level_4 = nn.Conv2d(in_channels=base_dim * 8, out_channels=base_dim * 8, kernel_size=3, stride=1, padding=1)
        self.down_level4_block1 = DEBlockTrain(default_conv, base_dim * 8, 3)
        self.down_level4_block2 = DEBlockTrain(default_conv, base_dim * 8, 3)

        self.up_level4_block1 = DEBlockTrain(default_conv, base_dim * 4, 3)
        self.up_level4_block2 = DEBlockTrain(default_conv, base_dim * 4, 3)
        self.up_level4_block3 = DEBlockTrain(default_conv, base_dim * 4, 3)
        self.up_level4_block4 = DEBlockTrain(default_conv, base_dim * 4, 3)

        self.fe_level_5 = nn.Conv2d(in_channels=base_dim, out_channels=base_dim, kernel_size=3, stride=1, padding=1)
        self.level5_block1 = DEABlockTrain(default_conv, base_dim * 8, 3)
        self.level5_block2 = DEABlockTrain(default_conv, base_dim * 8, 3)
        self.level5_block3 = DEABlockTrain(default_conv, base_dim * 8, 3)
        self.level5_block4 = DEABlockTrain(default_conv, base_dim * 8, 3)
        self.level5_block5 = DEABlockTrain(default_conv, base_dim * 8, 3)
        self.level5_block6 = DEABlockTrain(default_conv, base_dim * 8, 3)
        self.level5_block7 = DEABlockTrain(default_conv, base_dim * 8, 3)
        self.level5_block8 = DEABlockTrain(default_conv, base_dim * 8, 3)

        # up-sample
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(base_dim * 8, base_dim * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True))
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up1 = nn.Sequential(nn.Conv2d(base_dim, 1, kernel_size=3, stride=1, padding=1))


    def forward(self, x):
        inputs = x

        x_down1 = self.down1(inputs)
        x_down1 = self.down_level1_block1(x_down1)
        x_down1 = self.down_level1_block2(x_down1)

        print(x_down1.shape)

        x_down2 = self.down2(x_down1)
        x_down2_init = self.fe_level_2(x_down2)
        x_down2_init = self.down_level2_block1(x_down2_init)
        x_down2_init = self.down_level2_block2(x_down2_init)

        print(x_down2_init.shape)

        x_down3 = self.down3(x_down2_init)
        x_down3_init = self.fe_level_3(x_down3)
        x_down3_init = self.down_level3_block1(x_down3_init)
        x_down3_init = self.down_level3_block2(x_down3_init)

        print(x_down3_init.shape)

        x_down4 = self.down4(x_down3_init)
        x_down4_init = self.fe_level_4(x_down4)
        x_down4_init = self.down_level4_block1(x_down4_init)
        x_down4_init = self.down_level4_block2(x_down4_init)

        print(x_down4_init.shape)

        x1 = self.level5_block1(x_down4_init)
        x2 = self.level5_block2(x1)
        x3 = self.level5_block3(x2)
        x4 = self.level5_block4(x3)
        x5 = self.level5_block5(x4)
        x6 = self.level5_block6(x5)
        x7 = self.level5_block7(x6)
        x8 = self.level5_block8(x7)

        x_up4 = self.up4(x8)
        x_up4 = self.up_level4_block1(x_up4)
        x_up4 = self.up_level4_block2(x_up4)
        x_up4 = self.up_level4_block3(x_up4)
        x_up4 = self.up_level4_block4(x_up4) + x_down3_init

        x_up3 = self.up3(x_up4)
        x_up3 = self.up_level3_block1(x_up3)
        x_up3 = self.up_level3_block2(x_up3)
        x_up3 = self.up_level3_block3(x_up3)
        x_up3 = self.up_level3_block4(x_up3) + x_down2_init

        x_up2 = self.up2(x_up3)
        x_up2 = self.up_level2_block1(x_up2)
        x_up2 = self.up_level2_block2(x_up2)
        x_up2 = self.up_level2_block3(x_up2)
        x_up2 = self.up_level2_block4(x_up2)
        x_up2 = self.fe_level_5(x_up2) + x_down1

        out = self.up1(x_up2)
        return F.sigmoid(out)

def test():
    x = torch.randn((1, 1, 256, 256))
    y = torch.randn((1, 1, 256, 256))

    model = DC()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型的参数量为: {params}")

    model.cuda()
    preds = model(x.cuda(), y.cuda())
    print(preds.shape)


if __name__ == "__main__":
    test()
