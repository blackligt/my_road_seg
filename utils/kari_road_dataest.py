import torch
from pathlib import Path
import cv2

class KariRoadDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=False):
        self.root = Path(root)
        self.train = train
        if train:
            self.img_dir = self.root/'train'/'images'
        else:
            self.img_dir = self.root/'val'/'images'
            self.img_files = sorted(self.img_dir.glob('*.png'))
            self.transform = get_transforms(train)
    def __getitem__(self, idx):
        img_file= self.img_files[idx].as_posix()
        label_file = img_file.replace('images', 'labels')
        img = cv2.imread(img_file)
        label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
        img, label = self.transform(img, label)
        return img, label, img_file
    def __len__(self):
        return len(self.img_files)