from torch import nn
import torch

class ConvBNReLU(nn.Module):#CBR块

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):#256,256
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                # out_chan//4,
                out_chan//2,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//2,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)#torch.Size([14, 256, 128, 256])
        feat = self.convblk(fcat)#torch.Size([14, 256, 128, 256])
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class fc_discriminator(nn.Module):
    def __init__(self, num_classes, ndf=128):
        super(fc_discriminator, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),#new
            nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1, dilation=1),
        )
        self.upsample = nn.Upsample(size=5)
        self.downsample = nn.MaxPool2d(3, 2, 1)
        self.ffm = FeatureFusionModule(2, 2)
        self.final = nn.Conv2d(2,1,kernel_size=3,stride=1,padding=1)
        self.init_weight()

    def forward(self, x):
        # x=torch.cat([self.upsample(self.branch1(x)), self.downsample(self.branch2(x))], dim=1)
        x_32=self.upsample(self.branch1(x))
        x_8=self.downsample(self.branch2(x))
        x=self.ffm(x_8,x_32)
        x=self.final(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

#--------Deeplab-v2--------------
def get_fc_discriminator_add(num_classes, ndf=128):
    return nn.Sequential(
        nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
    )

# def get_fc_discriminator_add(num_classes, ndf=128):
#     return nn.Sequential(
#         nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1),
#     )

#--------UNET--------------D1--------------------
# def get_fc_discriminator(num_classes, ndf=128):
#     return nn.Sequential(
#         nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=2, dilation=2),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=2, dilation=2),
#     )

#--------UNET--------------D2--------------------
def get_fc_discriminator(num_classes, ndf=128):#8倍下采样
    return nn.Sequential(
        nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=1, padding=1, dilation=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1, dilation=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),#new
        nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1, dilation=1),
    )

# --------UNET--------------D3--------------------
# def get_fc_discriminator(num_classes, ndf=128):#4倍下采样
#     return nn.Sequential(
#         nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=1, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=1, padding=1, dilation=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1, dilation=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),#new
#         nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1, dilation=1),
#     )


#--------UNET--------------D4--------------------
# def get_fc_discriminator(num_classes, ndf=128):
#     return nn.Sequential(
#         nn.Conv2d(num_classes, ndf, kernel_size=3, stride=1, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=1, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=2, dilation=2),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=2, dilation=2),
#     )

# def get_fc_discriminator(num_classes, ndf=128):
#     return nn.Sequential(
#         nn.Conv2d(num_classes, ndf, kernel_size=3, stride=1, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=1, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=2, dilation=2),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=2, dilation=2),
#     )

if __name__ == "__main__":
    a=torch.randn([12,192,80,80])
    dis_model = fc_discriminator(num_classes=192)
    x=dis_model(a)