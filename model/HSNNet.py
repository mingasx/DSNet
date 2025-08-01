
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from functools import partial
from net.modeling.swintrans import FTUNetFormer
from net.modeling.resnet import resnet34
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate
tensor_rotate = partial(rotate, interpolation=InterpolationMode.BILINEAR)
nonlinearity = partial(F.relu,inplace=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result

class RCPM(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=9):
        super(RCPM, self).__init__()
        pad = ksize // 2
        self.conv_h = nn.Conv2d(in_channels, out_channels, kernel_size=(1, ksize), padding=(0, pad))
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size=(ksize, 1), padding=(pad, 0))
        self.conv_d1 = nn.Conv2d(
            in_channels, out_channels, (ksize, 1), padding=(ksize // 2, 0)
        )
        self.conv_d2 = nn.Conv2d(
            in_channels, out_channels, (ksize, 1), padding=(ksize // 2, 0)
        )
        self.fuse = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)



    def forward(self, x):
        h = self.conv_h(x)
        v = self.conv_v(x)
        d1 = tensor_rotate(self.conv_d1(tensor_rotate(x, 45)), -45)
        d2 = tensor_rotate(self.conv_d2(tensor_rotate(x, 135)), -135)
        return self.fuse(torch.cat([h, v, d1, d2], dim=1))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters,ksize=9):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        self.rcpm = RCPM(in_channels // 4,in_channels // 4,ksize)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.rcpm(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
class LinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super(LinkNet34, self).__init__()
        embed_dim = [96, 192, 384, 768]
        filters = [64, 128, 256, 512]
        resnet = resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.aspp = ASPP(512,512)

        self.swin = FTUNetFormer(backbone_pretrained=True, num_classes=1)
        self.conv1 = nn.Conv2d(embed_dim[0], filters[0] , 1)
        self.conv2 = nn.Conv2d(embed_dim[1], filters[1], 1)
        self.conv3 = nn.Conv2d(embed_dim[2], filters[2], 1)
        self.conv4 = nn.Conv2d(embed_dim[3], filters[3], 1)


        self.decoder4 = DecoderBlock(filters[3], filters[2],ksize=9)
        self.decoder3 = DecoderBlock(filters[2], filters[1],ksize=7)
        self.decoder2 = DecoderBlock(filters[1], filters[0],ksize=5)
        self.decoder1 = DecoderBlock(filters[0], filters[0],ksize=3)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        res1, res2, res3, res4 = self.swin(x)
        res1 = self.conv1(res1)
        res2 = self.conv2(res2)
        res3 = self.conv3(res3)
        res4 = self.conv4(res4)

        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e1 = e1 + res1
        e2 = e2 + res2
        e3 = e3 + res3
        e4 = e4 + res4

        e4 = self.aspp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
if __name__ == "__main__":
    import numpy as np

    a = np.random.rand(2, 3, 512, 512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.from_numpy(a).to(torch.float32)
    RCFSNet = LinkNet34()
    predicted = RCFSNet(a)
    print(predicted.shape)

