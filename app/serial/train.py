import pandas as pd
from app.data_utils import create_df, serial_pipeline

if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    df = create_df('files/NYPD_short.csv')
    df, target = serial_pipeline(df)
    print(target.head(10))

    print(target['SUSP_AGE_GROUP'].value_counts())