import os

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as v_transforms

import pandas as pd
import numpy as np
from PIL import Image

class nih_dataset(Dataset):
    def __init__(
            self,
            path_to_images,
            fold,
            transform,
            pred_label,
            csv_path):

        self.transform = transform
        self.path_to_images = path_to_images

        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['fold'] == fold]

        #set_index: Set the DataFrame index using existing columns.
        self.df = self.df.set_index("Image_Index")

        self.PRED_LABEL = pred_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = Image.open(
            os.path.join(
                self.path_to_images,
                self.df.index[idx]))
        image = image.convert('RGB')

        label = np.zeros(len(self.PRED_LABEL), dtype=int)

        for i in range(0, len(self.PRED_LABEL)):
             # can leave zero if zero, else make one
            if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') > 0):
                label[i] = self.df[self.PRED_LABEL[i].strip()
                                   ].iloc[idx].astype('int')

        if self.transform:
            image = self.transform(image)

        return (image, label, self.df.index[idx])

class nih_dataloader:
    def __init__(self, config, train=None):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        torchvision_train_transform = v_transforms.Compose([
                v_transforms.RandomHorizontalFlip(),
                v_transforms.Resize((config.img_resize[0], config.img_resize[1])),
                v_transforms.ToTensor(),
                v_transforms.Normalize(mean, std)
        ])

        torchvision_eval_transform = v_transforms.Compose([
                    v_transforms.Resize((config.img_resize[0], config.img_resize[1])),
                    v_transforms.ToTensor(),
                    v_transforms.Normalize(mean, std)
        ])


        if train == "train":
            transformed_datasets = nih_dataset(
                path_to_images=config.data_path,
                fold='train',
                transform=torchvision_train_transform,
                pred_label=config.num_classes,
                csv_path=config.label_data_path)
        elif train == "val":
            transformed_datasets = nih_dataset(
                path_to_images=config.data_path,
                fold='val',
                transform=torchvision_eval_transform,
                pred_label=config.num_classes,
                csv_path=config.label_data_path)
        elif train == "test":
            transformed_datasets = nih_dataset(
                path_to_images=config.data_path,
                fold='test',
                transform=torchvision_eval_transform,
                pred_label=config.num_classes,
                csv_path=config.label_data_path)
        else:
            print("no exist dataloader setting")
        
        self.transformed_datasets_= transformed_datasets

        self.dataset_len = len(transformed_datasets)

        self.num_iterations = (self.dataset_len + config.batch_size - 1) // config.batch_size

        self.train_loader = DataLoader(transformed_datasets,
            batch_size = config.batch_size,
            shuffle=True,
            num_workers=config.data_loader_workers,
            pin_memory = config.pin_memory,
            drop_last=True)

        self.val_loader = DataLoader(transformed_datasets,
            batch_size = 1,
            shuffle=False,
            num_workers=config.data_loader_workers,
            pin_memory = config.pin_memory,
            drop_last=True)

        self.test_loader = DataLoader(transformed_datasets,
            batch_size = 1,
            shuffle=False,
            num_workers=config.data_loader_workers,
            pin_memory = config.pin_memory,
            drop_last=False)
        

