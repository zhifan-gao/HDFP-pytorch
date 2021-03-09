import numpy as np
import torch
from PIL import Image
from config import Config as cg
from torch.utils.data import Dataset
from torchvision import transforms

torch.manual_seed(1)  # reproducible
class ImageDataset(Dataset):
    def __init__(self, image, label):
        self.image = np.load(image)  # 加载npy数据
        self.label = np.load(label)
        self.transforms = transforms.Compose([transforms.ToTensor()])  # 转为tensor形式

    def __getitem__(self, index):
        # cv数据集上使用
        image = np.array(self.image, dtype=np.float32)
        # batchs = image.shape[0]
        # image = np.reshape(image, newshape=[batchs, cg.image_size, cg.image_size, cg.image_channel])
        # image = image[index, :, :, :]
        # image = Image.fromarray(np.uint8(image))
        # image = self.transforms(image)

        #ivus图像使用：

        batchs = image.shape[0]
        image = np.reshape(image, newshape=[batchs, cg.image_size, cg.image_size, cg.image_channel])
        image = torch.from_numpy(image)
        image = image.permute(0, 3, 1, 2)
        image = image[index, :, :, :]
        image = images_preprocessing(image)



        label = np.array(self.label, dtype=np.float32)
        labelbatchs = label.shape[0]
        label = np.reshape(label, newshape=[labelbatchs, cg.image_size, cg.image_size, 1])
        label = label[index, :, :, :]
        label = np.squeeze(label)
        label = Image.fromarray(np.uint8(label))
        label = self.transforms(label)
        return image, label

    def __len__(self):
        image = np.array(self.image, dtype=np.float32)
        imagelength = image.shape[0]
        return imagelength # 返回数据的总个数

class ImageDataset_pred(Dataset):
    def __init__(self, image):
        self.image = np.load(image)  # 加载npy数据
        self.transforms = transforms.Compose([transforms.ToTensor()])  # 转为tensor形式

    def __getitem__(self, index):
        # cv数据集上使用
        image = np.array(self.image, dtype=np.float32)

        batchs = image.shape[0]
        image = np.reshape(image, newshape=[batchs, 256, 256, 3])
        image = torch.from_numpy(image)
        image = image.permute(0, 3, 1, 2)
        image = image[index, :, :, :]
        image = images_preprocessing(image)
        return image

    def __len__(self):
        image = np.array(self.image, dtype=np.float32)
        imagelength = image.shape[0]
        return imagelength


def images_preprocessing(images):


    images[2, :, :] -= torch.mean(images[2,:,:])
    images[1, :, :] -= torch.mean(images[1,:,:])
    images[0, :, :] -= torch.mean(images[0,:,:])

    images[2, :, :] /= ( torch.std(images[2,:,:], unbiased=False) + 1e-12)
    images[1, :, :] /= ( torch.std(images[1,:,:], unbiased=False) + 1e-12)
    images[0, :, :] /= ( torch.std(images[0,:,:], unbiased=False) + 1e-12)


    return images
