import torch
import numpy as np
from torch.utils.data import Dataset


class LoadData(Dataset):
    def __init__(self, filename, ratio):
        self.name = filename

        t_file = "../data/%s/text.npz" % filename
        i_file = "../data/%s/img.npz" % filename

        text = np.load(t_file)
        self.data_text = torch.from_numpy(text["data"]).float()

        img = np.load(i_file)
        self.data_img = torch.from_numpy(img["data"]).squeeze().float()
        self.labels = torch.from_numpy(text["label"]).squeeze().long()
        self.l = len(self.labels)

        self.train_idx = np.arange(int(self.l * 0.893910812646193 * ratio))
        self.test_idx = np.arange(int(self.l * 0.893910812646193), int(self.l))


    def __len__(self):
        return self.data_text.shape[0]

    def __getitem__(self, item):
        return self.data_text[item], self.data_img[item], self.labels[item]






