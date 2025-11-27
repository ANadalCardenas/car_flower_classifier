import os

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir 
        self.transform = transform
        
        # Labels:
        self.samples = []
        self.class_to_idx = {"car": 0, "flower" : 1}

        # Load all images from each class folder
        for cls_name, label in self.class_to_idx.items():
            folder = os.path.join(root_dir, cls_name)
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(folder, fname)
                    self.samples.append((img_path, label))
                    

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label
    
    
