import torch.nn as nn
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate
from net.modeling.resnet import resnet34
from functools import partial
from torch.nn import functional as F
from net.modeling.swintrans import FTUNetFormer
tensor_rotate = partial(rotate, interpolation=InterpolationMode.BILINEAR)
nonlinearity = partial(F.relu, inplace=True)


class initblock_plus4(nn.Module):
    # 激活函数用的relu bing xing
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, strip=9):
        super(initblock_plus4, self).__init__()

        self.conv_a0 = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size, stride, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True)
            )

        self.multi_conv1 = nn.Conv2d(in_ch, 8, (1, strip), stride=stride, padding=(0, strip // 2))
        self.multi_conv2 = nn.Conv2d(in_ch, 8, (strip, 1), stride=stride, padding=(strip // 2, 0))
        self.multi_conv3 = nn.Conv2d(in_ch, 8, (1, strip), stride=stride, padding=(0, strip // 2))
        self.multi_conv4 = nn.Conv2d(in_ch, 8, (1, strip), stride=stride, padding=(0, strip // 2))

        self.channel_concern = nn.Sequential(nn.Conv2d(32, 32, 1, 1, padding=0, dilation=dilation, bias=bias),
                                             nn.BatchNorm2d(32),
                                             nn.ELU(inplace=True)
                                             )

        self.angle = [0, 45, 90, 135, 180]
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x1 = self.multi_conv1(x)
        x2 = self.multi_conv2(x)
        x3 = self.conv_a0(x)
        x4 = self.multi_conv3(tensor_rotate(x, self.angle[1]))
        x5 = self.multi_conv4(tensor_rotate(x, self.angle[3]))

        out = torch.cat((x1,
                         x2,
                         tensor_rotate(x4, -self.angle[1]),
                         tensor_rotate(x5, -self.angle[3]),
                         ), 1)
        out = torch.cat((self.channel_concern(out),
                         x3), 1)
        out = self.down2(out)
        return out


class ResidualBlock(nn.Module):
    # 实现子module：Residual Block
    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = shortcut

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = x if self.downsample is None else self.downsample(x)
        out += residual
        return F.relu(out)


class connecter(nn.Module):
    # 连接器
    def __init__(self, in_ch, out_ch, scale_factor=0.5):
        super(connecter, self).__init__()
        self.downsample = partial(F.interpolate, scale_factor=scale_factor, mode='area', recompute_scale_factor=True)
        if not in_ch == out_ch:
            shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=1, bias=False),  # 1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
                nn.BatchNorm2d(out_ch))
        else:
            shortcut = None
        self.connect_conv = ResidualBlock(in_ch, out_ch, shortcut=shortcut)

    def forward(self, x,x1):
        x = self.downsample(x)
        x = self.connect_conv(x)
        x=x+x1
        return x





class DecoderBlock_v4fix(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm=nn.BatchNorm2d, in_p=True, strip=9):
        super(DecoderBlock_v4fix, self).__init__()
        out_pad = 1 if in_p else 0
        stride = 2 if in_p else 1

        self.cbr1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), )

        self.cbr2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            BatchNorm(in_channels // 2),
            nn.ReLU(inplace=True), )

        self.deconv1 = nn.Conv2d(
            in_channels // 4, in_channels // 4, (1, strip), padding=(0, strip // 2)
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 4, in_channels // 4, (strip, 1), padding=(strip // 2, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels // 4, in_channels // 4, (strip, 1), padding=(strip // 2, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 4, in_channels // 4, (strip, 1), padding=(strip // 2, 0)
        )

        self.cbr3_1 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), )
        self.cbr3_2 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), )
        self.cbr3_3 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), )
        self.cbr3_4 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), )

        self.deconvbr = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels // 4 + in_channels // 4,
                                                         3, stride=stride, padding=1, output_padding=out_pad),
                                      nn.BatchNorm2d(in_channels // 4 + in_channels // 4),
                                      nn.ReLU(inplace=True), )

        self.conv3 = nn.Conv2d(in_channels // 4 + in_channels // 4, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()

    def forward(self, x, inp=False):
        x01 = self.cbr1(x)

        x02 = self.cbr2(x)

        x1 = self.deconv1(x01)
        x2 = self.deconv2(x01)
        x3 = tensor_rotate(self.deconv3(tensor_rotate(x01, 45)), -45)
        x4 = tensor_rotate(self.deconv4(tensor_rotate(x01, 135)), -135)

        x1 = self.cbr3_1(torch.cat((x1, x02), 1))
        x2 = self.cbr3_2(torch.cat((x2, x02), 1))
        x3 = self.cbr3_3(torch.cat((x3, x02), 1))
        x4 = self.cbr3_4(torch.cat((x4, x02), 1))
        x = torch.cat((x1, x2, x3, x4), 1)

        x = self.deconvbr(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x


class MSMDFF_Net_v3_plus(nn.Module):
    def __init__(self, in_c=3, num_classes=1):
        super(MSMDFF_Net_v3_plus, self).__init__()

        layers = [64, 128, 256, 512]
        embed_dim = [96, 192, 384, 768]
        self.init_block = initblock_plus4(3, 64, stride=2)
        resnet = resnet34(pretrained=True)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.swin = FTUNetFormer(backbone_pretrained=True, num_classes=1)
        self.conv1 = nn.Conv2d(embed_dim[0], layers[0], 1)
        self.conv2 = nn.Conv2d(embed_dim[1], layers[1], 1)
        self.conv3 = nn.Conv2d(embed_dim[2], layers[2], 1)
        self.conv4 = nn.Conv2d(embed_dim[3], layers[3], 1)

        # 编码器 512 256 128 64
        self.encoder1_2 = connecter(3, layers[0], scale_factor=0.25)
        # self.max_pool1 = nn.MaxPool2d(3, 2, 1)
        self.encoder2_2 = connecter(3, layers[1], scale_factor=0.125)
        # self.max_pool2 = nn.MaxPool2d(3, 2, 1)
        self.encoder3_2 = connecter(3, layers[2], scale_factor=0.0625)
        # self.max_pool3 = nn.MaxPool2d(3, 2, 1)
        self.encoder4_2 = connecter(3, layers[3], scale_factor=0.03175)
        # 解码器
        self.decoder4 = DecoderBlock_v4fix(layers[3], layers[2], in_p=True)
        self.decoder3 = DecoderBlock_v4fix(layers[2], layers[1], in_p=True)
        self.decoder2 = DecoderBlock_v4fix(layers[1], layers[0], in_p=True)
        self.decoder1 = DecoderBlock_v4fix(layers[0], layers[0], in_p=True)

        # self.finaldeconv1 = nn.ConvTranspose2d(layers[0], 32, 4, 2, 1)
        # self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(layers[0], 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        res1, res2, res3, res4 = self.swin(x)
        res1 = self.conv1(res1)
        res2 = self.conv2(res2)
        res3 = self.conv3(res3)
        res4 = self.conv4(res4)

        x1 = self.init_block(x)

        x1 = self.encoder1(x1)
        e1 = self.encoder1_2(x, x1)  # 128*256*256
        x2 = self.encoder2(e1)
        e2 = self.encoder2_2(x, x2)  # 256*128*128
        x3 = self.encoder3(e2)
        e3 = self.encoder3_2(x, x3)  # 512*64*64
        x4 = self.encoder4(e3)
        e4 = self.encoder4_2(x, x4)  # 512*32*32

        e1 = e1 + res1
        e2 = e2 + res2
        e3 = e3 + res3
        e4 = e4 + res4

        # Decoder
        d4 = self.decoder4(e4) + e3  # 256*64*64
        d3 = self.decoder3(d4) + e2  # 128*128*128
        d2 = self.decoder2(d3) + e1  # 64*256*256
        d1 = self.decoder1(d2)  # 64*256*256

        # out = self.finaldeconv1(d1)
        # out = self.finalrelu1(out)
        out = self.finalconv2(d1)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        # torch.cuda.empty_cache()
        return torch.sigmoid(out)


'''
中间跳跃层的代价十分大，后面可考虑是否使用中间跳跃层
此外，反卷积的那个 带状卷积 再次尝试一下。
'''

if __name__ == '__main__':
    from torchinfo import summary
    from thop import profile  # 用于计算 FLOPs

    # 创建模型并移动到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSMDFF_Net_v3_plus().to(device)

    # 构造输入张量
    input_tensor = torch.randn(4, 3, 512, 512).to(device)

    # 使用 torchinfo 的 summary（兼容复杂输出）
    summary(model, input_size=(4, 3, 512, 512), device=device.type)

    # 使用 thop 计算 FLOPs 和参数量
    flops, params = profile(model, inputs=(input_tensor,))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Parameters: {params / 1e6:.2f} M")

'''
src
================================================================
Total params: 39,256,881
Trainable params: 39,256,881
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.75
Forward/backward pass size (MB): 986.62
Params size (MB): 149.75
Estimated Total Size (MB): 1137.13
----------------------------------------------------------------
v2
================================================================
Total params: 24,431,921
Trainable params: 24,431,921
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.75
Forward/backward pass size (MB): 971.12
Params size (MB): 93.20
Estimated Total Size (MB): 1065.08
----------------------------------------------------------------
'''