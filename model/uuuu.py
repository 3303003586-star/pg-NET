import torch
import torch.nn as nn
from module.deconv import DEConv
from module.cga import SpatialAttention, ChannelAttention, PixelAttention


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class DEABlock(nn.Module):
    def __init__(self, conv, dim, kernel_size, reduction=8):
        super(DEABlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        cattn = self.ca(res)
        sattn = self.sa(res)
        pattn1 = sattn + cattn
        pattn2 = self.pa(res, pattn1)
        res = res * pattn2
        res = res + x
        return res


class DEBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DEBlock, self).__init__()
        self.conv1 = DEConv(dim)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        res = res + x
        return res


# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # ------------- Encoder Configuration -------------
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        # self.input_ca = ChannelAttention(in_planes=16)  # 请确认这里的数值与你 down1 的输出通道一致
        #
        # self.refine = DEBlock(default_conv, 160, 3)
        #
        # # 3. 最终输出层：从 160 通道映射到 1 通道残差


        # ------------- Pooling Layers -------------
        self.pool_2 = nn.MaxPool2d(2, 2)
        self.pool_4 = nn.MaxPool2d(4, 4)
        self.pool_8 = nn.MaxPool2d(8, 8)
        self.pool_16 = nn.MaxPool2d(16, 16)

        # ------------- Up_sampling Layers -------------
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        # ------------- Output Layer -------------
        self.out = nn.Conv2d(160, 1, kernel_size=1)

    def _build_encoder(self):
        """Build the encoder part of the network"""
        return nn.ModuleDict({
            'down1': nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            'down2': self._make_down_layer(16, 30),
            'down3': self._make_down_layer(46, 60),
            'down4': self._make_down_layer(122, 160),
            'down5': self._make_down_layer(344, 400),

            'conv31': nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=1),
            'conv32': nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1),
            'conv33': nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=1),
            'conv34': nn.Conv2d(400, 400, kernel_size=3, stride=1, padding=1),
            'reduce1': nn.Conv2d(400, 160, kernel_size=1),

            'down_block1': DEBlock(default_conv, 16, 3),
            'down_block11': DEBlock(default_conv, 16, 3),
            'down_block12': DEBlock(default_conv, 16, 3),
            'down_block13': DEBlock(default_conv, 16, 3),
            'down_block2': DEBlock(default_conv, 46, 3),
            'down_block21': DEBlock(default_conv, 46, 3),
            'down_block22': DEBlock(default_conv, 46, 3),
            'down_block23': DEBlock(default_conv, 46, 3),
            'down_block3': DEBlock(default_conv, 122, 3),
            'down_block31': DEBlock(default_conv, 122, 3),
            'down_block32': DEBlock(default_conv, 122, 3),
            'down_block33': DEBlock(default_conv, 122, 3),
            'down_block4': DEBlock(default_conv, 344, 3),
            'down_block41': DEBlock(default_conv, 344, 3),
            'down_block42': DEBlock(default_conv, 344, 3),
            'down_block43': DEBlock(default_conv, 344, 3),
            'down_block51': DEABlock(default_conv, 344, 3),
            'down_block52': DEABlock(default_conv, 344, 3),
            'down_block53': DEABlock(default_conv, 344, 3),
            'down_block54': DEABlock(default_conv, 344, 3)
        })

    def _build_decoder(self):
        """Build the decoder part of the network"""
        return nn.ModuleDict({
            'up1': nn.Sequential(
                nn.ConvTranspose2d(344, 172, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(True)),
            'up2': nn.Sequential(
                nn.ConvTranspose2d(160, 200, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(True)),
            'up3': nn.Sequential(
                nn.ConvTranspose2d(160, 80, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(True)),
            'up4': nn.Sequential(
                nn.ConvTranspose2d(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(True)),

            'reduce2': nn.Conv2d(516, 160, kernel_size=1),
            'reduce3': nn.Conv2d(666, 160, kernel_size=1),
            'reduce4': nn.Conv2d(630, 160, kernel_size=1),
            'reduce5': nn.Conv2d(744, 160, kernel_size=1),

            'up_block1': DEBlock(default_conv, 160, 3),
            'up_block11': DEBlock(default_conv, 160, 3),
            'up_block12': DEBlock(default_conv, 160, 3),
            'up_block13': DEBlock(default_conv, 160, 3),
            'up_block2': DEBlock(default_conv, 160, 3),
            'up_block21': DEBlock(default_conv, 160, 3),
            'up_block22': DEBlock(default_conv, 160, 3),
            'up_block23': DEBlock(default_conv, 160, 3),
            'up_block3': DEBlock(default_conv, 160, 3),
            'up_block31': DEBlock(default_conv, 160, 3),
            'up_block32': DEBlock(default_conv, 160, 3),
            'up_block33': DEBlock(default_conv, 160, 3),
            'up_block4': DEBlock(default_conv, 160, 3),
            'up_block41': DEBlock(default_conv, 160, 3),
            'up_block42': DEBlock(default_conv, 160, 3),
            'up_block43': DEBlock(default_conv, 160, 3)
        })

    def _make_down_layer(self, in_channels, out_channels):
        """Helper function to create downsampling layers"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        pl_mwm = x[:, 5:6, :, :]
        # # ------------- Encoder Path -------------
        # Level 1
        x1 = self.encoder['down1'](x)
        # x1 = self.input_ca(x1) * x1  # 你之前加的注意力机制
        x12 = self.encoder['down_block1'](x1)  # 16 channels
        x12 = self.encoder['down_block11'](x12)  # 16 channels
        x12 = self.encoder['down_block12'](x12)  # 16 channels
        x12 = self.encoder['down_block13'](x12)  # 16 channels

        # Level 2
        x2 = self.encoder['down2'](x12)  # 30 channels
        x21 = self.encoder['conv31'](x2)
        x1_down = self.pool_2(x12)  # 16 channels
        x22 = torch.cat([x21, x1_down], dim=1)  # 46 channels
        x23 = self.encoder['down_block2'](x22)  # 46 channels
        x23 = self.encoder['down_block21'](x23)  # 46 channels
        x23 = self.encoder['down_block22'](x23)  # 46 channels
        x23 = self.encoder['down_block23'](x23)  # 46 channels

        # Level 3
        x3 = self.encoder['down3'](x23)  # 60 channels
        x31 = self.encoder['conv32'](x3)
        x1_down = self.pool_4(x12)  # 16 channels
        x2_down = self.pool_2(x23)  # 46 channels
        x32 = torch.cat([x31, x1_down, x2_down], dim=1)  # 122 channels
        x33 = self.encoder['down_block3'](x32)  # 122 channels
        x33 = self.encoder['down_block31'](x33)  # 122 channels
        x33 = self.encoder['down_block32'](x33)  # 122 channels
        x33 = self.encoder['down_block33'](x33)  # 122 channels

        # Level 4
        x4 = self.encoder['down4'](x33)  # 160 channels
        x41 = self.encoder['conv33'](x4)
        x1_down = self.pool_8(x12)  # 16 channels
        x2_down = self.pool_4(x23)  # 46 channels
        x3_down = self.pool_2(x33)  # 122 channels
        x44 = torch.cat([x41, x1_down, x2_down, x3_down], dim=1)  # 344 channels
        x45 = self.encoder['down_block4'](x44)  # 344 channels
        x45 = self.encoder['down_block41'](x45)  # 344 channels
        x45 = self.encoder['down_block42'](x45)  # 344 channels
        x45 = self.encoder['down_block43'](x45)  # 344 channels

        # Level 5
        x5 = self.encoder['down5'](x45)  # 344 channels
        x51 = self.encoder['conv34'](x5)
        x51 = self.encoder['reduce1'](x51)  # 160 channels
        x1_down = self.pool_16(x12)  # 16 channels
        x2_down = self.pool_8(x23)  # 46 channels
        x3_down = self.pool_4(x33)  # 122 channels
        x52 = torch.cat([x51, x1_down, x2_down, x3_down], dim=1)  # 344 channels
        x53 = self.encoder['down_block51'](x52)
        x54 = self.encoder['down_block52'](x53)
        x55 = self.encoder['down_block53'](x54)
        x53 = self.encoder['down_block54'](x55)

        # ------------- Decoder Path -------------
        # Level 4 up
        x_up4 = self.decoder['up1'](x53)  # 172 channels
        x_up41 = torch.cat([x_up4, x45], dim=1)  # 516 channels
        x_up41 = self.decoder['reduce2'](x_up41)  # 160 channels
        x_up42 = self.decoder['up_block1'](x_up41)  # 160 channels
        x_up42 = self.decoder['up_block11'](x_up42)  # 160 channels
        x_up42 = self.decoder['up_block12'](x_up42)  # 160 channels
        x_up42 = self.decoder['up_block13'](x_up42)  # 160 channels

        # Level 3 up
        x_up3 = self.decoder['up2'](x_up42)  # 200 channels
        x53_up = self.up_4(x53)  # 344 channels
        x_up31 = torch.cat([x_up3, x33, x53_up], dim=1)  # 666 channels
        x_up31 = self.decoder['reduce3'](x_up31)  # 160 channels
        x_up3 = self.decoder['up_block2'](x_up31)  # 160 channels
        x_up3 = self.decoder['up_block21'](x_up3)  # 160 channels
        x_up3 = self.decoder['up_block22'](x_up3)  # 160 channels
        x_up3 = self.decoder['up_block23'](x_up3)  # 160 channels

        # Level 2 up
        x_up2 = self.decoder['up3'](x_up3)  # 80 channels
        x53_up = self.up_8(x53)  # 344 channels
        x_up42_up = self.up_4(x_up42)  # 160 channels
        x_up21 = torch.cat([x_up2, x23, x53_up, x_up42_up], dim=1)  # 630 channels
        x_up21 = self.decoder['reduce4'](x_up21)  # 160 channels
        x_up2 = self.decoder['up_block3'](x_up21)  # 160 channels
        x_up2 = self.decoder['up_block31'](x_up2)  # 160 channels
        x_up2 = self.decoder['up_block32'](x_up2)  # 160 channels
        x_up2 = self.decoder['up_block33'](x_up2)  # 160 channels

        # Level 1 up
        x_up1 = self.decoder['up4'](x_up2)  # 64 channels
        x53_up = self.up_16(x53)  # 344 channels
        x_up42_up = self.up_8(x_up42)  # 160 channels
        x_up3_up = self.up_4(x_up3)  # 160 channels
        x_up11 = torch.cat([x_up1, x12, x53_up, x_up42_up, x_up3_up], dim=1)  # 744 channels
        x_up11 = self.decoder['reduce5'](x_up11)  # 160 channels
        x_up1 = self.decoder['up_block4'](x_up11)  # 160 channels
        x_up1 = self.decoder['up_block41'](x_up1)  # 160 channels
        x_up1 = self.decoder['up_block42'](x_up1)  # 160 channels
        x_up1 = self.decoder['up_block43'](x_up1)  # 160 channels
        # residual = self.out(x_up1)
        # out = pl_mwm + residual

        return self.out(x_up1)

def test():
    x = torch.randn((4, 8, 256, 256))
    models = MyModel()
    params = sum(p.numel() for p in models.parameters() if p.requires_grad)
    print(f"模型的参数量为: {params}")
    models.cuda()
    preds = models(x.cuda())
    print(preds.shape)


if __name__ == "__main__":
    test()
