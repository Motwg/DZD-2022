import pandas as pd
import numpy as np
import torch
import torch.utils.data as data_utils
from app.data_utils import create_df, nn_pipeline


if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    df = create_df('files/NYPD_short.csv')
    df, target = nn_pipeline(df)
    print(target.head(10))

    # print(df['PREM_TYP_DESC'].value_counts())
    # target = torch.tensor(df['Targets'].values)
    # features = torch.tensor(df.drop('Targets', axis=1).values)
    #
    # train = data_utils.TensorDataset(features, target)
    # train_loader = data_utils.DataLoader(train, batch_size=10, shuffle=True)
    #
    # print(train)