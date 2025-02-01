import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class UTKFaceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []

        for fname in os.listdir(root_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                parts = fname.split('_')
                if len(parts) >= 3 and parts[0].isdigit() and parts[1] in ('0', '1'):
                    self.image_paths.append(os.path.join(root_dir, fname))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        fname = os.path.basename(img_path)

        age = int(fname.split('_')[0])
        gender = int(fname.split('_')[1])

        image = Image.open(img_path).convert('RGB')

        # Return PIL Image and raw values
        return image, {'age': age, 'gender': gender}


def get_train_transforms():
    return A.Compose([
        A.Rotate((-20,20)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])