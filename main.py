import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv(
        'files/NYPD_short.csv',
        dtype={'CMPLNT_NUM': 'str'},
        on_bad_lines='error',
        low_memory=False,
        parse_dates={
            'CMPLNT_FR': ['CMPLNT_FR_TM', 'CMPLNT_FR_DT'],
            'CMPLNT_TO': ['CMPLNT_TO_TM', 'CMPLNT_TO_DT'],
            'RP_DT': ['RPT_DT']
        }
    )
    for t in ['CMPLNT_FR', 'CMPLNT_TO']:
        df[t] = pd.to_datetime(df[t], format='%H:%M:%S %m/%d/%Y', errors='coerce')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    df.head(7)
