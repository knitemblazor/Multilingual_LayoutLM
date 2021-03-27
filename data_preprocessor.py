from torch.utils.data import Dataset
import numpy as np


class DatasetCheckbox(Dataset):

    def __init__(self, feature_vect,label):
        self.data = feature_vect
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index].astype(np.uint8).reshape((1,50,50))
        label = self.label[index]
        return image, label