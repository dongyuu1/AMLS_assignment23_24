import os
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image


class TaskDataset(Dataset):
    def __init__(self, cfg, x, y, train):
        self.images = x
        y = torch.tensor(y).to(torch.int64)
        self.labels = y
        if train:
            self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.ToTensor(),
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.RandomVerticalFlip(p=0.5),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
                                                 ])
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
                                                 ])

    def __getitem__(self, index):
        image = self.transform(self.images[index])

        item = (image, self.labels[index])
        return item

    def __len__(self):
        return len(self.images)


def get_test_score(label_pred, label_true):
    b = label_pred.shape[0]
    #if len(label_true.shape) == 1:
    #    label_true = F.one_hot(label_true)
    pred_indices = label_pred.max(dim=1, keepdim=False).indices
    true_indices = label_true.max(dim=1, keepdim=False).indices
    match_nums = (pred_indices == true_indices).sum().item()
    return match_nums / b


def one_hot_to_indices(one_hot):
    return one_hot.max(dim=1, keepdim=False).indices


def numpy_to_pil(x_batch):
    b = x_batch.shape[0]
    pil_list = []
    for i in range(b):
        x = Image.fromarray(x_batch[i])
        pil_list.append(x)
    return pil_list

