import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
import sqlite3


class BunkerSensors(Dataset):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.db_file = '../ensort.db'
        self.tables = []

        self.con = sqlite3.connect(self.db_file)
        self.cur = self.con.execute("SELECT name FROM sqlite_master WHERE type='table';")

        for table in self.cur:
            if "Bunkersensoren" in table[0]:
                self.tables.append(table[0])

        self.num_data = 10e6

        for table in self.tables:
            self.cur = self.con.execute("SELECT COUNT(*) FROM " + table)
            num = self.cur.fetchone()[0]
            if num < self.num_data:
                self.num_data = num
    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):

        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.num_data)
        num_rows = end_idx - start_idx

        sql_statement = ""
        for table in self.tables:
            sql_statement += "SELECT Sensor_01, Sensor_02, Sensor_03, Sensor_04, Sensor_05, Sensor_06, " \
                             "Sensor_07, Sensor_08, Sensor_09, Sensor_10, Sensor_11, Sensor_12 " \
                             "FROM " + str(table) + " UNION ALL "
        sql_statement = sql_statement[:-11]
        sql_statement += " LIMIT "+str(start_idx) + ", " + str(num_rows)
        sql_statement += ";"
        # print(sql_statement)
        self.cur = self.con.execute(sql_statement)
        data = self.cur.fetchall()

        # return the data as a tuple, assuming the first column is the label
        return data

    def __del__(self):
        self.con.close()


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def _check_integrity(self) -> bool:
        return True


class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """

    def __init__(self,
                 data_path: str,
                 split: str,
                 transform: Callable,
                 **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])

        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, 0.0  # dummy datat to prevent breaking


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            patch_size: Union[int, Sequence[int]] = (256, 256),
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        #       =========================  OxfordPets Dataset  =========================

        #         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                               transforms.CenterCrop(self.patch_size),
        # #                                               transforms.Resize(self.patch_size),
        #                                               transforms.ToTensor(),
        #                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        #         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                             transforms.CenterCrop(self.patch_size),
        # #                                             transforms.Resize(self.patch_size),
        #                                             transforms.ToTensor(),
        #                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        #         self.train_dataset = OxfordPets(
        #             self.data_dir,
        #             split='train',
        #             transform=train_transforms,
        #         )

        #         self.val_dataset = OxfordPets(
        #             self.data_dir,
        #             split='val',
        #             transform=val_transforms,
        #         )

        #       =========================  CelebA Dataset  =========================

        '''
                train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                      transforms.CenterCrop(148),
                                                      transforms.Resize(self.patch_size),
                                                      transforms.ToTensor(),])

                val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                    transforms.CenterCrop(148),
                                                    transforms.Resize(self.patch_size),
                                                    transforms.ToTensor(),])
        '''
        self.train_dataset = BunkerSensors(10)
        self.val_dataset = BunkerSensors(10)
        # print(self.train_dataset.__getitem__(1))

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
