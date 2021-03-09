#coding=utf-8#
import sys
import numpy as np
import sys
sys.path.append('/home/fengtianyuan/code1')
from config import Config as cg
import torch.nn as nn
import torch
from upsample_skimage import  upsample
import torchvision
import os
batch_size = 8
log_dir = cg.root_path + "/log"
class single_salicency_model(nn.Module):
    def __init__(self, drop_rate, layers):
        super(single_salicency_model, self).__init__()
        self.drop_rate = drop_rate
        self.layers = layers
        self.conv2d = conv2d(in_features=3, out_features=16, kernel_size=3)
        #block1 256*256 d=1
        self.block1 = block(layers=12, dilated_rate=1, drop_rate=self.drop_rate)
        self.conv2d1 = conv2d(in_features=160, out_features=16, kernel_size=3)
        self.bac1 = batch_activ_conv1(in_features=160, out_features=80, kernel_size=1, dilated_rate=1, drop_rate=self.drop_rate)
        self.avg1 = avg_pool(s=2)
        #block2 128*128 d=1
        self.block2 = block(layers=12, dilated_rate=1, drop_rate=self.drop_rate)
        self.conv2d2 = conv2d(in_features=224, out_features=16, kernel_size=3)
        self.bac2 = batch_activ_conv1(in_features=224, out_features=112, kernel_size=1, dilated_rate=1, drop_rate=self.drop_rate)
        self.avg2 = avg_pool(s=2)
        # block3 64X64 d = 2
        self.block3 = block(layers=12, dilated_rate=2, drop_rate=self.drop_rate)
        self.conv2d3 = conv2d(in_features=256, out_features=16, kernel_size=3)
        self.bac3 = batch_activ_conv1(in_features=256, out_features=128, kernel_size=1, dilated_rate=1, drop_rate=self.drop_rate)
        # block4 64X64 d = 4
        self.block4 = block(layers=12, dilated_rate=4, drop_rate=self.drop_rate)
        self.conv2d4 = conv2d(in_features=272, out_features=16, kernel_size=3)
        self.bac4 = batch_activ_conv1(in_features=272, out_features=136, kernel_size=1, dilated_rate=1, drop_rate=self.drop_rate)
        # block5 64X64 d = 8
        self.block5 = block(layers=12, dilated_rate=8, drop_rate=self.drop_rate)
        self.conv2d5 = conv2d(in_features=280, out_features=16, kernel_size=3)
        #logits_scale_64_3
        self.ppm64 = pyramid_pooling_64(features_channel=16, pyramid_feature_channels=1)
        self.batchnorm64_3 = batchnorm(eps=1e-05)
        self.relu64_3 = torch.nn.ReLU(inplace=False)
        self.conv2d64_3 = conv2d(in_features=20, out_features=1, kernel_size=3)
        #logits_scale_64_2
        self.batchnorm64_2 = batchnorm(eps=1e-05)
        self.relu64_2 = torch.nn.ReLU(inplace=False)
        self.conv2d64_2 = conv2d(in_features=36, out_features=1, kernel_size=3)
        #logits_scale_64_1
        self.batchnorm64_1 = batchnorm(eps=1e-05)
        self.relu64_1 = torch.nn.ReLU(inplace=False)
        self.conv2d64_1 = conv2d(in_features=52, out_features=1, kernel_size=3)
        #upsample to 128
        self.upsample128 = upsample(factor=2, channel=52)
        self.batchnorm128 = batchnorm(eps=1e-05)
        self.relu128 = torch.nn.ReLU(inplace=False)
        self.conv2d128 = conv2d(in_features=68, out_features=1, kernel_size=3)
        #upsample to 256
        self.upsample256 = upsample(factor=2, channel=68)
        self.batchnorm256 = batchnorm(eps=1e-05)
        self.relu256 = torch.nn.ReLU(inplace=False)
        self.conv2d256 = conv2d(in_features=84, out_features=1, kernel_size=3)
        self.upsample1 = upsample(factor=2, channel=1)
        self.upsample = upsample(factor=4, channel=1)
        #logsit后的卷积
        self.convlast = conv2d(in_features=5, out_features=1, kernel_size=3)
        self.convlast2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[3, 3], stride=[1, 1], padding=1)

    def forward(self, xs):
        input = xs
        current = self.conv2d(input)
        #block1 256*256 d=1
        current, features = self.block1(current, 16, 12)
        scale_256 = self.conv2d1(current)
        current = self.bac1(current)
        current = self.avg1(current)
        # block2 128X128 d=1
        current, features = self.block2(current, 80, 12)
        scale_128 = self.conv2d2(current)
        current = self.bac2(current)
        current = self.avg2(current)
        # block3 64X64 d = 2
        current, features = self.block3(current, 112, 12)
        scale_64_1 = self.conv2d3(current)
        current = self.bac3(current)
        # block4 64X64 d = 4
        current, features = self.block4(current, 128, 12)
        scale_64_2 = self.conv2d4(current)
        current = self.bac4(current)
        # block5 64X64 d = 8
        current, features = self.block5(current, 136, 12)
        scale_64_3 = self.conv2d5(current)
        # 64_3 Map
        ppm_64_3, ppm_channels_64_3 = self.ppm64(scale_64_3)
        concat_64_3 = torch.cat([scale_64_3, ppm_64_3], 1)
        current_64_3 = self.batchnorm64_3(concat_64_3)
        current_64_3 = self.relu64_3(current_64_3)
        logits_scale_64_3 = self.conv2d64_3(current_64_3)
        # 64_2 Map
        concat_64_2 = torch.cat([scale_64_2, concat_64_3], 1)
        current_64_2 = self.batchnorm64_2(concat_64_2)
        current_64_2 = self.relu64_2(current_64_2)
        logits_scale_64_2 = self.conv2d64_2(current_64_2)
        # 64_1 Map
        concat_64_1 = torch.cat([scale_64_1, concat_64_2], 1)
        current_64_1 = self.batchnorm64_1(concat_64_1)
        current_64_1 = self.relu64_1(current_64_1)
        logits_scale_64_1 = self.conv2d64_1(current_64_1)
        # recovery 128
        concat_scale_64_upsamped = self.upsample128(concat_64_1)
        logits_scale_128_concat = torch.cat((scale_128, concat_scale_64_upsamped), 1)
        logits_scale_current_128 = self.batchnorm128(logits_scale_128_concat)
        logits_scale_current_128 = self.relu128(logits_scale_current_128)
        logits_scale_128 = self.conv2d128(logits_scale_current_128)
        # recovery 256
        logits_scale_128_upsamped = self.upsample256(logits_scale_128_concat)
        logits_scale_256_concat = torch.cat((scale_256, logits_scale_128_upsamped), 1)  # 按照通道堆叠起来
        logits_scale_current_256 = self.batchnorm256(logits_scale_256_concat)
        logits_scale_current_256 = self.relu256(logits_scale_current_256)
        logits_scale_256 = self.conv2d256(logits_scale_current_256)

        logits_scale_64_3_upsampled_to_256 = self.upsample(logits_scale_64_3)
        logits_scale_64_2_upsampled_to_256 = self.upsample(logits_scale_64_2)
        logits_scale_64_1_upsampled_to_256 = self.upsample(logits_scale_64_1)
        logits_scale_128_upsampled_to_256 = self.upsample1(logits_scale_128)

        logits_scale_64_3_upsampled_to_256_sigmoid = torch.sigmoid(logits_scale_64_3_upsampled_to_256)
        logits_scale_64_2_upsampled_to_256_sigmoid = torch.sigmoid(logits_scale_64_2_upsampled_to_256)
        logits_scale_64_1_upsampled_to_256_sigmoid = torch.sigmoid(logits_scale_64_1_upsampled_to_256)
        logits_scale_128_upsampled_to_256_sigmoid = torch.sigmoid(logits_scale_128_upsampled_to_256)
        logits_scale_256_upsampled_to_256_sigmoid = torch.sigmoid(logits_scale_256)

        logits_concat = torch.cat((logits_scale_64_3_upsampled_to_256,
                                   logits_scale_64_2_upsampled_to_256,
                                   logits_scale_64_1_upsampled_to_256,
                                   logits_scale_128_upsampled_to_256,
                                   logits_scale_256
                                   ), 1)

        logits = self.convlast(logits_concat)
        logits = self.convlast2(logits)
        yp = torch.sigmoid(logits)
        return yp, logits_scale_64_3_upsampled_to_256_sigmoid, logits_scale_64_2_upsampled_to_256_sigmoid, logits_scale_64_1_upsampled_to_256_sigmoid, logits_scale_128_upsampled_to_256_sigmoid, logits_scale_256_upsampled_to_256_sigmoid


class conv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size):
        super(conv2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.convlution = torch.nn.Conv2d(self.in_features, self.out_features, self.kernel_size, stride=[1, 1], padding=0,
                                 dilation=1, groups=1,
                                 bias=True)
    def forward(self, current):
        convlution = self.convlution
        conv = torch.nn.functional.pad(current, pad=[self.kernel_size - 1, 0, 0, self.kernel_size-1])
        conv = convlution(conv)

        return conv

class batch_activ_conv(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, dilated_rate, drop_rate):
        super(batch_activ_conv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.dilated_rate = dilated_rate
        self.drop_rate = drop_rate
        self.batchnorm = batchnorm(eps=1e-05)
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv = torch.nn.Conv2d(self.in_features, self.out_features, self.kernel_size, stride=[1, 1], padding=0,
                                    dilation=self.dilated_rate, groups=1, bias=True)
        self.dropout = torch.nn.Dropout(p=self.drop_rate)

    def forward(self, current):
        current = self.batchnorm(current)  # conv2d(input, 3, 16, 3)，最后生成的featuremap
        current = self.relu(current)
        current = torch.nn.functional.pad(current, pad=[0, 1, 1, 0], mode='constant', value=0)
        current = self.conv(current)
        current = self.dropout(current)
        return current

class batch_activ_conv1(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, dilated_rate, drop_rate):
        super(batch_activ_conv1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.dilated_rate = dilated_rate
        self.drop_rate = drop_rate
        self.batchnorm = batchnorm(1e-05)
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv = torch.nn.Conv2d(self.in_features, self.out_features, self.kernel_size, stride=[1, 1], padding=0,
                                    dilation=self.dilated_rate, groups=1, bias=True)
        self.dropout = torch.nn.Dropout(p=self.drop_rate)

    def forward(self, current):
        #current = torch.nn.functional.pad(current, pad=[0, 1, 1, 0], mode='constant', value=0)
        current = self.batchnorm(current)  # conv2d(input, 3, 16, 3)，最后生成的featuremap
        current = self.relu(current)
        current = self.conv(current)
        current = self.dropout(current)
        return current

# 每一个块中的节点是相互连通的
class block(nn.Module):
    def __init__(self, layers, dilated_rate, drop_rate):
        super(block, self).__init__()
        self.layers = layers
        self.dilated_rate = dilated_rate
        self.drop_rate = drop_rate

    def forward(self, current, features, growth):
        for idx in range(self.layers):
            bac = batch_activ_conv(features, growth, 3, self.dilated_rate, self.drop_rate).cuda()
            tmp = bac(current)
            tmp = torch.nn.functional.pad(tmp, pad=[0, 2 * self.dilated_rate - 1, 0, 2 * self.dilated_rate - 1], mode='constant', value=0)
            current = torch.cat((current, tmp), 1)  # 按照通道堆叠起来
            features += growth
        return current, features



class avg_pool(nn.Module):
    def __init__(self, s):
        super(avg_pool, self).__init__()
        self.s = s
    def forward(self,input):
        avg_pool = torch.nn.functional.avg_pool2d(input, kernel_size=[self.s, self.s], stride=[self.s, self.s], padding=0)
        return avg_pool


#ppm for 64X64



class pyramid_pooling_64(nn.Module):
    def __init__(self, features_channel, pyramid_feature_channels):
        super(pyramid_pooling_64, self).__init__()
        self.features_channel = features_channel
        self.pyramid_feature_channels = pyramid_feature_channels

        self.layer1 = nn.Sequential(avg_pool(64),
                                    conv2d(self.features_channel, self.pyramid_feature_channels, 1),
                                    upsample(64, 1))
        self.layer2 = nn.Sequential(avg_pool(32),
                                    conv2d(self.features_channel, self.pyramid_feature_channels, 1),
                                    upsample(32, 1))
        self.layer3 = nn.Sequential(avg_pool(21),
                                    conv2d(self.features_channel, self.pyramid_feature_channels, 1),
                                    upsample(21, 1))
        self.layer4 = nn.Sequential(avg_pool(10),
                                    conv2d(self.features_channel, self.pyramid_feature_channels, 1),
                                    upsample(10, 1))

    def forward(self, tensor):
        pyramid_layer1_upsampled = self.layer1(tensor)
        pyramid_layer2_upsampled = self.layer2(tensor)
        pyramid_layer3_upsampled = self.layer3(tensor)
        pyramid_layer3_upsampled = nn.functional.pad(pyramid_layer3_upsampled, pad=[0, 1, 1, 0], mode='reflect')
        pyramid_layer4_upsampled = self.layer4(tensor)
        pyramid_layer4_upsampled = nn.functional.pad(pyramid_layer4_upsampled, pad=[0, 4, 4, 0], mode='reflect')

        pyramid = torch.cat((pyramid_layer1_upsampled,
                             pyramid_layer2_upsampled,
                             pyramid_layer3_upsampled,
                             pyramid_layer4_upsampled),
                            dim=1)
        pyramid = torch.reshape(pyramid, [-1, 4 * self.pyramid_feature_channels, 64, 64])
        pyramid_feature_channels1 = self.pyramid_feature_channels
        return pyramid, pyramid_feature_channels1*4


def fused_loss(yp,gt):
    mae_loss = torch.mean(torch.log(1 + torch.exp(torch.abs(yp - gt))))  # yp是激活函数
    # tf.compat.v1.summary.scalar("mae_loss", mae_loss)#显示标量信息，可视化张量
    mask_front = gt
    mask_background = 1 - gt
    pro_front = yp
    pro_background = 1 - yp

    w1 = 1 / (torch.pow(torch.sum(mask_front), 2) + 1e-12).item()
    w2 = 1 / (torch.pow(torch.sum(mask_background), 2) + 1e-12).item()
    numerator = w1 * torch.sum(mask_front * pro_front) + w2 * torch.sum(mask_background * pro_background)
    denominator = w1 * torch.sum(mask_front + pro_front) + w2 * torch.sum(mask_background + pro_background)
    dice_loss = 1 - 2 * numerator / (denominator + 1e-12)
    dice = 1 - dice_loss
    # tf.compat.v1.summary.scalar("dice_loss", dice_loss)#画出dice_loss

    w = (256 * 256 * batch_size - torch.sum(yp).item()) / (torch.sum(yp).item() + 1e-12)
    # w = (cg.image_size * cg.image_size * 1 - tf.reduce_sum(yp)) / (tf.reduce_sum(yp) + 1e-12)

    cross_entropy_loss = -torch.mean(0.1 * w * mask_front * torch.log(pro_front + 1e-12) + mask_background * torch.log(
        pro_background + 1e-12))

    return  dice_loss + mae_loss + cross_entropy_loss, dice
    #return  mae_loss + cross_entropy_loss


class batchnorm(nn.Module):
    def __init__(self, eps):
        super(batchnorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        x2 = x.shape[1]
        batchnormfunction = torch.nn.BatchNorm2d(x2, eps=self.eps, momentum=0.1, affine=True, track_running_stats=True)
        batchnormfunction = batchnormfunction.cuda()
        output = batchnormfunction(x)
        return output


def F_measure(gt, map):
    mask = torch.gt(map, 0.5)   # map>0.5, return ture, map<0.5,return false
    mask = mask.type(torch.float32)   # 把mask抓换为float32格式mask = mask.type(torch.float32)

    gtCnt = torch.sum(gt)   # 计算gt中所有元素的总和
    hitMap = torch.where(gt > 0, mask, torch.zeros(mask.size()).cuda())   # 如果gt>0，tensor中对应位置取mask，gt<=0，tensor对应取0 ---- 规定tensor中的值大于等于0

    hitCnt = torch.sum(hitMap)
    algCnt = torch.sum(mask)

    prec = hitCnt / (algCnt + 1e-12)
    recall = hitCnt / (gtCnt + 1e-12)

    beta_square = 0.3

    F_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall + 1e-32)
    return  prec, recall, F_score

