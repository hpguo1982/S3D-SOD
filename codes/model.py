import torch
import torch.nn as nn
from timm.models.swin_transformer import swin_base_patch4_window12_384

from backbones.swin import SwinTransformer
from modules import Conv2D, DecoderBlock, UpBlock, ResidualBlock, RCSA, RSA
from backbones.pvtv2 import pvt_v2_b3

import warnings
import torch.nn.functional as F


class SSDSeg(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.backbone = pvt_v2_b3()  ## [64, 128, 320, 512]
        #self.backbone.
        path = 'pvt_v2_b3.pth'
        save_model = torch.load(path)

        model_dict = self.backbone.state_dict()

        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}

        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        """Channel-Spatial Attention Blocks"""
        self.rcsa1 = RCSA(64)
        self.rcsa2 = RCSA(128)
        self.rcsa3 = RCSA(320)
        self.rsa = RSA(512)

        """ Channel Reduction """
        self.c1 = Conv2D(64, 64, kernel_size=1, padding=0)
        self.c2 = Conv2D(128, 64, kernel_size=1, padding=0)
        self.c3 = Conv2D(320, 64, kernel_size=1, padding=0)
        self.c4 = Conv2D(512, 64, kernel_size=1, padding=0)


        self.d1 = DecoderBlock(64, 64)
        self.d2 = DecoderBlock(64, 64)
        self.d3 = DecoderBlock(64, 64)
        self.d4 = UpBlock(64, 64, 4)

        self.r1 = ResidualBlock(64, 64)
        self.y = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        #self.rrs_1d = RRS_1D(in_ch=1, out_ch=64)

        # s


    def forward(self, inputs):
        """ Encoder """
        pvt1 = self.backbone(inputs)
        e1 = pvt1[0]     ## [-1, 64, h/4, w/4]
        e2 = pvt1[1]     ## [-1, 128, h/8, w/8]
        e3 = pvt1[2]     ## [-1, 320, h/16, w/16]
        e4 = pvt1[3]     ## [-1, 512, h/32, w/32]

        ae1 = self.rcsa1(e1)
        ae2 = self.rcsa2(e2)
        ae3 = self.rcsa3(e3)
        ae4 = self.rsa(e4)

        c1 = self.c1(ae1)
        c2 = self.c2(ae2)
        c3 = self.c3(ae3)
        c4 = self.c4(ae4)

        d1 = self.d1(c4,c3)
        d2 = self.d2(d1,c2)
        d3 = self.d3(d2,c1)
        x = self.d4(d3)
        x = self.r1(x)
        y = self.y(x)

        return y



if __name__ == "__main__":
    x = torch.randn((4, 3, 256, 256))
    model = SSDSeg()
    y = model(x)
    print(y.shape)
