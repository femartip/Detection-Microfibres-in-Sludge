import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
from torch.nn import Module
from torchvision.models import vgg16


#Symetric padding
def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad):
    if pad[2]+pad[3] > 0:
        x = x[:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,pad[0]:-pad[1]]
    return x

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2
    )

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)

class DoubleConv_no_relu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.conv_op(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        encoder = vgg16(pretrained=False).features
        self.down1 = nn.Sequential(*encoder[:6])
        self.down2 = nn.Sequential(*encoder[6:13])
        self.down3 = nn.Sequential(*encoder[13:20])
        
        self.bottle_neck = nn.Sequential(*encoder[20:27])
        self.conv_bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = up_conv(1024, 512)
        self.conv_1 = double_conv(1024, 512)
        self.up_convolution_2 = up_conv(512, 256)
        self.conv_2 = double_conv(512, 256)
        self.up_convolution_3 = up_conv(256, 128)
        self.conv_3 = double_conv(256, 128)

        self.out = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        x, pads = pad_to(x, 32)

        down_1 = self.down1(x)
        #print(f"down_1: {down_1.shape}")
        down_2 = self.down2(down_1)
        #print(f"down_2: {down_2.shape}")
        down_3 = self.down3(down_2)
        #print(f"down_3: {down_3.shape}")

        bottle_neck = self.bottle_neck(down_3)
        bottle_neck = self.conv_bottle_neck(bottle_neck)

        up_1 = self.up_convolution_1(bottle_neck)
        #print(f"up_1: {up_1.shape}, down_3: {down_3.shape}")
        up_1 = torch.cat([up_1, down_3], 1)
        up_1 = self.conv_1(up_1)

        up_2 = self.up_convolution_2(up_1)
        #print(f"up_2: {up_2.shape}, down_2: {down_2.shape}")
        up_2 = torch.cat([up_2, down_2], 1)
        up_2 = self.conv_2(up_2)

        up_3 = self.up_convolution_3(up_2)
        #print(f"up_3: {up_3.shape}, down_1: {down_1.shape}")
        up_3 = torch.cat([up_3, down_1], 1)
        up_3 = self.conv_3(up_3)
        
        out = self.out(up_3)
        out = unpad(out, pads)
        out = F.interpolate(out, size=(750, 1000), mode='bilinear', align_corners=False)
        #print(f"out: {out.shape}")
        return out



"""
BB-Unet


class DownConv(Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(DownConv, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)
        self.conv1_drop = nn.Dropout2d(drop_rate)

        self.conv2 = nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)
        self.conv2_drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)

        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        return x


class UpConv(Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(UpConv, self).__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downconv = DownConv(in_feat, out_feat, drop_rate, bn_momentum)

    def forward(self, x, y):
        x = self.up1(x)
        x = torch.cat([x, y], dim=1)
        x = self.downconv(x)
        return x

class BBConv(Module):
    def __init__(self, in_feat, out_feat, pool_ratio, no_grad_state):
        super(BBConv, self).__init__()
        self.mp = nn.MaxPool2d(pool_ratio)
        self.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1)
        if no_grad_state is True:
            self.conv1.requires_grad = False
        else:
            self.conv1.requires_grad = True
    def forward(self, x):
        x = self.mp(x)
        x = self.conv1(x)
        x = F.sigmoid(x)
        return x

class BB_Unet(Module):
    def __init__(self, no_grad=False, BB_boxes = 1):
        super(BB_Unet, self).__init__()
        if no_grad is True:
            no_grad_state = True
        else:
            no_grad_state = False
        
        #Downsampling path, encoder
        encoder = vgg16(pretrained=True).features
        encoder = vgg16(pretrained=True).features
        self.down1 = nn.Sequential(*encoder[:6])
        self.down2 = nn.Sequential(*encoder[6:13])
        self.down3 = nn.Sequential(*encoder[13:20])
        # Bottle neck
        #self.conv4 = DownConv(256, 256, drop_rate, bn_momentum)
        self.bottle_neck = nn.Sequential(*encoder[20:27])
        self.conv_bottle_neck = DoubleConv(512, 1024)

        # bounding box encoder path:
        self.b1 = BBConv(BB_boxes, 256, 4, no_grad_state)
        self.b2 = BBConv(BB_boxes, 128, 2, no_grad_state)
        self.b3 = BBConv(BB_boxes, 64, 1, no_grad_state)

        # Upsampling path
        self.up_convolution_1 = up_conv(1024, 512)
        self.conv_1 = double_conv(1024, 512)
        self.up_convolution_2 = up_conv(512, 256)
        self.conv_2 = double_conv(512, 256)
        self.up_convolution_3 = up_conv(256, 128)
        self.conv_3 = double_conv(256, 128)

        self.out = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)

    def forward(self, x, bb, comment= 'tr'):
        down1 = self.down1(x)
        print(f"down1: {down1.shape}")
        down2 = self.down2(down1)
        print(f"down2: {down2.shape}")
        down3 = self.down3(down2)
        print(f"down3: {down3.shape}")

        # Bottle neck
        #x4 = self.conv4(p3)
        bottle_neck = self.bottle_neck(down3)
        bottle_neck = self.conv_bottle_neck(bottle_neck)
        print(f"bottle_neck: {bottle_neck.shape}")
        
        # bbox encoder
        if comment == 'tr':
            f1_1 = self.b1(bb)
            f2_1 = self.b2(bb)
            f3_1 = self.b3(bb)
        
            x3_1 = down3*f3_1
            x2_1 = down2*f2_1
            x1_1 = down1*f1_1
            # Up-sampling
            #u1 = self.up1(x4, x3_1)
            #u2 = self.up2(u1, x2_1)
            #u3 = self.up3(u2, x1_1)
            u1 = self.up_convolution_1(bottle_neck)
            u1 = torch.cat([u1, x3_1], 1)
            u1 = self.conv_1(u1)
            u2 = self.up_convolution_2(u1)
            u2 = torch.cat([u2, x2_1], 1)
            u2 = self.conv_2(u2)
            u3 = self.up_convolution_3(u2)
            u3 = torch.cat([u3, x1_1], 1)
            u3 = self.conv_3(u3)


        elif comment == 'val':
            x3_1 = down3
            x2_1 = down2
            x1_1 = down1
            # Up-sampling
            #u1 = self.up1(x4, x3_1)
            #u2 = self.up2(u1, x2_1)
            #u3 = self.up3(u2, x1_1)
            u1 = self.up_convolution_1(bottle_neck)
            u1 = torch.cat([u1, x3_1], 1)
            u1 = self.conv_1(u1)
            u2 = self.up_convolution_2(u1)
            u2 = torch.cat([u2, x2_1], 1)
            u2 = self.conv_2(u2)
            u3 = self.up_convolution_3(u2)
            u3 = torch.cat([u3, x1_1], 1)
            u3 = self.conv_3(u3)

        o = self.out(u3)
        #o = F.sigmoid(o)
        return o
"""