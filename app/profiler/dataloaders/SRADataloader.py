import pandas as pd
import pytorch_lightning as pl
import torch
import os
from torch.utils.data import DataLoader, random_split, TensorDataset


class SRADataloader(pl.LightningDataModule):

    def __init__(
            self,
            sra: str,
            seed=41,
            batch_size=256,
            filename='data_short.csv',
            data_dir='app/profiler/data',
            no_workers=0
    ):
        super().__init__()
        self.batch_size = batch_size
        self.no_workers = no_workers

        # load data
        df = pd.read_csv(os.path.join(data_dir, filename), low_memory=False, index_col=False)

        # choose labels
        df_labels = {
            's': df['SUSP_SEX'].values,
            'r': df['SUSP_RACE'].values,
            'a': df['SUSP_AGE_GROUP'].values
        }.get(sra, 's')
        df = df.drop(columns=['SUSP_AGE_GROUP', 'SUSP_RACE', 'SUSP_SEX']).values

        # convert to torch dtypes
        self.dataset = torch.tensor(df).float()
        self.labels = torch.tensor(df_labels.reshape(-1)).long()

        # split
        self.train, self.val, self.test = random_split(
            TensorDataset(self.dataset, self.labels),
            [.6, .2, .2],
            generator=torch.Generator().manual_seed(seed)
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]

    def train_dataloader(self):
        return DataLoader(
            self.train,
            self.batch_size,
            shuffle=True,
            num_workers=self.no_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            self.batch_size,
            shuffle=True,
            num_workers=self.no_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            self.batch_size,
            num_workers=self.no_workers
        )
