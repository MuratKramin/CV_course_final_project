import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from data.base_dataset import BaseDataset
import numpy as np

class PanoFromCsvDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        # === Указываем пути к данным ===
        self.csv_path = r"filtered_ids.csv"

        self.pano_dir = r"panoramas\images"
        self.sat_dir = r"satelite\images"
        self.mask_dir = r"segmentation_masks\images"

        self.df = pd.read_csv(self.csv_path)

        self.transform_sat = transforms.Compose([
            #transforms.Resize((256, 256), interpolation=Image.NEAREST),

            #transforms.Resize((704, 704), interpolation=Image.BICUBIC),
            #transforms.Resize((352, 352), interpolation=Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.transform_pano = transforms.Compose([
            #transforms.Resize((256, 1024), interpolation=Image.NEAREST),

            #transforms.Resize((704, 2816), interpolation=Image.BICUBIC),
            #transforms.Resize((352, 1408), interpolation=Image.NEAREST),
            #transforms.Resize((1280, 3584), interpolation=Image.BICUBIC),
            
            #transforms.Resize((640, 1792), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, index):
        row = self.df.iloc[index]
        pano_id = row['id']
        pano_path = os.path.join(self.pano_dir, row['pano_file'])
        sat_path = os.path.join(self.sat_dir, row['sat_file'])
        mask_path = os.path.join(self.mask_dir, row['mask_file'])

        A_img = Image.open(sat_path).convert("RGB")
        B_img = Image.open(pano_path).convert("RGB")
        D_img = Image.open(mask_path).convert("RGB")

        A = self.transform_sat(A_img)
        B = self.transform_pano(B_img)
        D = self.transform_pano(D_img)

        # Добавляем повороты спутникового изображения → [3, 704, 704*4]
        A_rotated = [A]
        for i in range(1, 4):
            A_rotated.append(transforms.functional.rotate(A, 90 * i))
        A_concat = torch.cat(A_rotated, dim=2)

        return {'A': A_concat, 'B': B, 'D': D, 'A_paths': pano_id, 'B_paths': pano_id}

    def __len__(self):
        return len(self.df)
