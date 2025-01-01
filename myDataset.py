# coding: gbk
import torch
from torch import tensor
from torch.utils.data import Dataset
from PIL import Image
import os

from torchvision import transforms


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_listPath = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_listPath[idx]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_listPath)


path = "C:\\Users\\ASUS\\Desktop\\cs skill tree.png"
img = Image.open(path)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img.shape)