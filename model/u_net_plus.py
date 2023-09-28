import torch
import torch.nn as nn
from torch.nn import functional as F

from model.resnet import resnet50
from model.vgg import VGG16

class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x
    
class unetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(unetUp, self).__init__()
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.conv = double_conv(in_channels,out_channels)
        
    def forward(self, inputs1, inputs2):
        inputs2 = self.up(inputs2)
        diffY = torch.tensor([inputs1.size()[2] - inputs2.size()[2]])
        diffX = torch.tensor([inputs1.size()[3] - inputs2.size()[3]])
        inputs2 = F.pad(inputs2, (torch.div(diffX, 2, rounding_mode='floor'),
                                  diffX-torch.div(diffX, 2, rounding_mode='floor'),
                                  torch.div(diffY, 2, rounding_mode='floor'),
                                  diffY-torch.div(diffY, 2, rounding_mode='floor')))
        
        outputs = torch.cat([inputs1, inputs2], 1)
        
        return self.conv(outputs)

class Unet(nn.Module):
    def __init__(self, params, pretrained = False, backbone = 'vgg'):
        num_classes = params.n_classes
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained,
                                in_channels=params.n_channels)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained,
                                   in_channels=params.n_channels)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError(f'Unsupported backbone - `{backbone}`, Use vgg, resnet50.')
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes,1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        # you must use a sigmoid or softmax function to be activation 
        # Or you include this layer in your loss function
        return torch.sigmoid(final)

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True