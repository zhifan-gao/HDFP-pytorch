#coding=utf-8#
import numpy as np
# from config import Config as cg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

from auxiliary_functions import *
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
from loadnpy import ImageDataset_pred
import matplotlib.pyplot as plt
import torch
import cv2
import nibabel as nib

def prediction(data_for_pred):
    ssm = single_salicency_model(drop_rate=0.2, layers=12)
    ssm = torch.nn.DataParallel(ssm, device_ids=[0,1])
    ssm.cuda()
    ssm.load_state_dict(torch.load('/home/fengtianyuan/ma_2_162.pth'))
    ssm.eval()
    data = DataLoader(data_for_pred, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    for i, (imagedata) in enumerate(data):
        print(i)
        xs = imagedata.cuda()
        yp, _, _, _, _, _ = ssm(xs)
        img = torch.gt(yp, 0.5)  # map>0.5, return ture, map<0.5,return false
        img = img.type(torch.float32)
        yp1 = img.clone().cpu()
        yp1_1 = yp1.squeeze(0)
        # yp1_2 = trans(yp1_1)
        yyyy = yp1_1.detach().numpy()
        # plt.imshow(yyyy[0, :, :])
        # plt.axis('off')
        # plt.show()

        # img = cv2.resize(yyyy[0, :, :], (526, 526))
        # _, img = cv2.threshold(img, 0.5, 1, cv2.THRESH_BINARY)
        # img_pad = np.pad(img.T, ((89,89),(0,178)),'constant',constant_values = (0,0))
        img = cv2.resize(yyyy[0, :, :], (500, 500))

        _, img = cv2.threshold(img, 0.9, 2, cv2.THRESH_BINARY)
        print('sbdxwsnd')
        img_pad = img.T


        # plt.imshow(img_pad)
        # plt.axis('off')
        # plt.show()

        if 0 == i:
            img_flat_mat = np.reshape(img_pad,(1,500,500))
        else:
            img_flat_mat = np.vstack((img_flat_mat, np.reshape(img,(1,500,500))))

        # new_image=nib.Nifti1Image(img_flat_mat.transpose(1,2,0), np.eye(4))
        # nib.save(new_image,'/home/zhiyihua/exam.nii.gz')

        # xs1 = xs.clone().cpu()
        # xs_1 = xs1.squeeze(0)
        # plt.imshow(np.transpose(xs_1.detach().numpy(), (1, 2, 0)))
        # plt.axis('off')
        # plt.show()
    new_image=nib.Nifti1Image(img_flat_mat.transpose(1,2,0),np.eye(4))
    nib.save(new_image,'/home/fengtianyuan/zhanghailin.nii.gz')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    data1 = ImageDataset_pred('/home/fengtianyuan/saliencydetectdataset/ivus_t.npy')
    prediction(data1)
