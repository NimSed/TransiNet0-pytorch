"""
Copyright © Nima Sedaghat 2017-2021

All rights reserved under the GPL license enclosed with the software. Over and
above the legal restrictions imposed by this license, if you use this software
for an academic publication then you are obliged to provide proper attribution
to the below paper:

    Sedaghat, Nima, and Ashish Mahabal. "Effective image differencing with
    convolutional neural networks for real-time transient hunting." Monthly
    Notices of the Royal Astronomical Society 476, no. 4 (2018): 5365-5376.
"""

import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )


def predict(in_planes):
    return nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4,
                           stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


# =====================================================================================
class TN_enc(nn.Module):

    def __init__(self, batchNorm=False, f=1):
        super(TN_enc, self).__init__()

        self.batchNorm = batchNorm

        self.conv1 = conv(self.batchNorm,   2,    f *
                          64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm,  f*64,  f *
                          128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, f*128,  f *
                          256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, f*256,  f*256)
        self.conv4 = conv(self.batchNorm, f*256,  f*512, stride=2)
        self.conv4_1 = conv(self.batchNorm, f*512,  f*512)
        self.conv5 = conv(self.batchNorm, f*512,  f*512, stride=2)
        self.conv5_1 = conv(self.batchNorm, f*512,  f*512)
        self.conv6 = conv(self.batchNorm, f*512,  f*1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, f*1024, f*1024)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        return out_conv1, out_conv2, out_conv3, out_conv4, out_conv5, out_conv6

# =====================================================================================


class TN_dec(nn.Module):

    def __init__(self, enc, batchNorm=False, f=1):
        super(TN_dec, self).__init__()

        self.batchNorm = batchNorm

        N = enc.conv6._modules['0'].out_channels
        self.predict6 = predict(N)
        self.upsampled6_to_5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=True)
        # concat happens here
        self.deconv5 = deconv(N, f*512)  # (out_conv6_1)

        N = self.deconv5._modules['0'].out_channels + \
            enc.conv5._modules['0'].out_channels + \
            self.upsampled6_to_5.out_channels
        self.predict5 = predict(N)
        self.upsampled5_to_4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=True)
        # concat happens here
        self.deconv4 = deconv(N, f*256)  # (out_deconv5+out_conv5+out6_up)

        N = self.deconv4._modules['0'].out_channels + \
            enc.conv4._modules['0'].out_channels + \
            self.upsampled5_to_4.out_channels
        self.predict4 = predict(N)
        self.upsampled4_to_3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=True)
        # concat happens here
        self.deconv3 = deconv(N, f*128)  # (out_deconv4+out_conv4+out5_up)

        N = self.deconv3._modules['0'].out_channels + \
            enc.conv3._modules['0'].out_channels + \
            self.upsampled4_to_3.out_channels
        self.predict3 = predict(N)
        self.upsampled3_to_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=True)
        # concat happens here
        self.deconv2 = deconv(N, f*64)  # (out_deconv3+out_conv3+out4_up)

        N = self.deconv2._modules['0'].out_channels + \
            enc.conv2._modules['0'].out_channels + \
            self.upsampled3_to_2.out_channels
        self.predict2 = predict(N)
        self.upsampled2_to_1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=True)
        # concat happens here
        self.deconv1 = deconv(N, f*32)  # (out_deconv2+out_conv2+out3_up)

        N = self.deconv1._modules['0'].out_channels + \
            enc.conv1._modules['0'].out_channels + \
            self.upsampled2_to_1.out_channels
        self.predict1 = predict(N)
        self.upsampled1_to_0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=True)
        # concat happens here
        self.deconv0 = deconv(N, f*16)  # (out_deconv1+out_conv1+out2_up)

        # 2 is the total number of channels of the input
        N = self.deconv0._modules['0'].out_channels + \
            2 + self.upsampled1_to_0.out_channels
        self.predict0 = predict(N)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, enc_outputs):

        out_conv1, out_conv2, out_conv3, out_conv4, out_conv5, out_conv6 = enc_outputs

        out6 = self.predict6(out_conv6)
        out6_up = self.upsampled6_to_5(out6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, out6_up), 1)
        out5 = self.predict5(concat5)
        out5_up = self.upsampled5_to_4(out5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, out5_up), 1)
        out4 = self.predict4(concat4)
        out4_up = self.upsampled4_to_3(out4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, out4_up), 1)
        out3 = self.predict3(concat3)
        out3_up = self.upsampled3_to_2(out3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, out3_up), 1)
        out2 = self.predict2(concat2)
        out2_up = self.upsampled2_to_1(out2)
        out_deconv1 = self.deconv1(concat2)

        concat1 = torch.cat((out_conv1, out_deconv1, out2_up), 1)
        out1 = self.predict1(concat1)
        out1_up = self.upsampled1_to_0(out1)
        out_deconv0 = self.deconv0(concat1)

        concat0 = torch.cat((x, out_deconv0, out1_up), 1)
        out0 = self.predict0(concat0)

        return (out0, out1, out2, out3, out4, out5, out6)


# =====================================================================================
# =====================================================================================
class TN(nn.Module):

    def __init__(self, batchNorm=False):
        super(TN, self).__init__()

        f = 2

        self.batchNorm = batchNorm
        self.enc = TN_enc(batchNorm=self.batchNorm, f=f)
        self.dec = TN_dec(self.enc, batchNorm=self.batchNorm, f=f)

    def forward(self, x):
        out_conv1, out_conv2, out_conv3, out_conv4, out_conv5, out_conv6 = self.enc.forward(
            x)
        return self.dec.forward(x,
                                (out_conv1, out_conv2, out_conv3,
                                 out_conv4, out_conv5, out_conv6))

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def mtn(data=None):
    """
    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = TN(batchNorm=False)
    if data is not None:

        if isinstance(data, dict):
            model.load_state_dict(data, strict=True)
        else:
            model.load_state_dict(data['state_dict'], strict=True)
    return model


def mtn_bn(data=None):
    model = TN(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'], strict=True)
    return model
