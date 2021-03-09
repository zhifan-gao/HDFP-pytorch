
from __future__ import division
import sys
sys.path.append('')
from numpy import ogrid, repeat, newaxis
from skimage import io
import skimage.transform
import numpy as np
import torch
import torch.nn as nn

def upsample_skimage(factor, input_img):
    # Pad with 0 values, similar to how Tensorflow does it.
    # Order=1 is bilinear upsampling
    return skimage.transform.rescale(input_img,factor,mode='constant',cval=0,order=1)

class upsample(nn.Module):
    def __init__(self, factor, channel):
        super(upsample, self).__init__()
        self.factor = factor
        self.channel = channel

    def forward(self, input):
        upsampled_input = torch.nn.functional.interpolate(input=input, scale_factor=self.factor, mode='bilinear', align_corners=True)
        upsampled_input = upsampled_input.cuda()
        return upsampled_input






