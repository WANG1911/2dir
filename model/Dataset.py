import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from config import Config

class CustomDataset(Dataset):
    def __init__(self, img_folder, csv_folder, transform=None):
        self.img_folder = img_folder
        self.csv_folder = csv_folder
        self.transform = transform
        self.file_list = [f for f in os.listdir(img_folder) if f.endswith('.png')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        prefix = img_name.split('.')[0]
        csv_path = os.path.join(self.csv_folder, prefix + ".csv")
        img_path = os.path.join(self.img_folder, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            print(f"Error loading image: {img_path}, Error: {str(e)}")
            img = None

        matrix = pd.read_csv(csv_path, skiprows=1, header=None).values.astype(np.float32)

        return img, torch.tensor(matrix)