from torch.utils.data import Dataset
import numpy as np
import os
import json


class MyDataset(Dataset):

    def __init__(self):
        super(MyDataset, self).__init__()
        self.img = []
        self.gt_bbox = []

    def add(self, data):
        self.img.append(data[0])
        self.gt_bbox.append(np.asarray(data[1]))

    def __getitem__(self, idx):
        return self.img[idx], self.gt_bbox[idx]

    def __len__(self):
        return len(self.img)

    # def load_dataset(self, dataset_root, dataset):


