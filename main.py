import os
import pandas as pd

from app.data_utils import create_df

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # "short" for testing and debugging, "full" for production
    df = create_df('files/NYPD_short.csv')
    print(df.head(10))

    df = pd.read_csv(os.path.join('app/profiler/data', 'data_full.csv'), low_memory=False, index_col=False)
    print(df.isnull().sum())
