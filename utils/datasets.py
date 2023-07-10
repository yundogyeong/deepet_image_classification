import os
import cv2
import json
import torch
import pickle
import albumentations as A
import numpy as np

from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from functools import partial

# 데이터셋의 절대경로

def read_json(p:str) -> dict:
    with open(p, 'r') as f:
        data = f.read()
        obj = json.loads(data)
        return obj

def write_cache(p:str, data:list) -> None:
    with open(p, 'wb') as file:
        pickle.dump(data, file)

def load_cache(p):
    with open(p, 'rb') as f:
        data = pickle.load(f)
        return list(data)

def read_img(p:str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


class TrainDataset(Dataset):
    def __init__(self, root:str, imgsz:int=224):
        self.idx2class = os.path.join(root, "idx2class.json")
        super().__init__()
        # default size
        if imgsz is None:
            imgsz = 224
        
        self.root = os.path.join(root, 'train')
        self.phase = 'train'
        
        # parsing
        if not os.path.exists(os.path.join(root, self.phase+".pkl")):
            self.data_list = self._parsing()
            # write cache
            write_cache(os.path.join(root, self.phase+".pkl"), self.data_list)
            print(f"# write cahce file({self.phase})")
        else:
            print(f"# load cache file({self.phase})")
            self.data_list = load_cache(os.path.join(root, self.phase+".pkl"))

        self.data_length = len(self.data_list)

        # data augmentation
        self.transform = A.Compose([
            A.SmallestMaxSize(max_size=256),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomCrop(height=imgsz, width=imgsz),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
    def _parsing(self) -> list:
        data_list = []
        for dir in os.listdir(self.root):
            # dir = "0000_safety pin"
            class_index = int(dir.split('_')[0])
            for img_path in os.listdir(os.path.join(self.root, dir)):
                abs_img_path = os.path.join(self.root, dir, img_path)
                data_list.append((class_index, abs_img_path))
        return data_list
            
        
    def __len__(self) -> int:
        return self.data_length
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        class_index, abs_img_path = self.data_list[idx]
        img_obj = read_img(abs_img_path)    # img_obj:np.ndarray
        img_tensor = self.transform(image=img_obj)["image"]
        return img_tensor, class_index

class ValidDataset(Dataset):
    def __init__(self, root, imgsz:int=224):
        super().__init__()
        if imgsz is None:
            imgsz = 224
            
        self.root = os.path.join(root, 'val')
        self.phase = 'val'
        
        # parsing
        if not os.path.exists(os.path.join(root, self.phase+".pkl")):
            self.data_list = self._parsing()
            # write cache
            write_cache(os.path.join(root, self.phase+".pkl"), self.data_list)
            print(f"# write cahce file({self.phase})")
        else:
            print(f"# load cache file({self.phase})")
            self.data_list = load_cache(os.path.join(root, self.phase+".pkl"))

        self.data_length = len(self.data_list)
        
        # data augmentation
        self.transform = A.Compose([
            A.SmallestMaxSize(max_size=256),
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
    def _parsing(self) -> list:
        data_list = []
        for img_path in os.listdir(self.root):
            # img_path = "0000_020996.JPEG"
            class_index = int(img_path.split("_")[0])
            abs_img_path = os.path.join(self.root, img_path)
            data_list.append((class_index, abs_img_path))
        return data_list
     
    def __len__(self) -> int:
        return self.data_length
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        class_index, abs_img_path = self.data_list[idx]
        img_obj = read_img(abs_img_path)
        img_tensor = self.transform(image=img_obj)["image"]
        return img_tensor, class_index


def get_dataloader(root:str=None, batch_size:int=16, imgsz:int=224, num_workers:int=8) -> Tuple[DataLoader, DataLoader]:
    _train_dataset = TrainDataset(root, imgsz)
    _valid_dataset = ValidDataset(root, imgsz)
    train_loader = DataLoader(_train_dataset, batch_size, True, num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(_valid_dataset, batch_size, True, num_workers=num_workers, persistent_workers=True)
    return train_loader, val_loader

if __name__ == "__main__":
    _ = get_dataloader()