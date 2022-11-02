import numpy as np
import pandas as pd

from app.data_imputation import data_imputation

if __name__ == '__main__':
    df = pd.read_csv(
        # "short" for testing and debugging, "full" for production
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

    # edit incorrect age groups
    for col in ['SUSP_AGE_GROUP', 'VIC_AGE_GROUP']:
        df.loc[df[col].isna() | df[col].str.isdigit() | df[col].str.startswith('-'), col] = np.NaN

    # edit hispanic race
    for col in ['VIC_RACE', 'SUSP_RACE']:
        df[col] = df[col].str.removesuffix(' HISPANIC')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    data = df[['CMPLNT_FR', 'BORO_NM', 'PREM_TYP_DESC',
               'VIC_AGE_GROUP', 'VIC_RACE', 'VIC_SEX',
               'SUSP_AGE_GROUP', 'SUSP_RACE', 'SUSP_SEX',
               'Latitude', 'Longitude']]
    print(data.head(6))

    data_imputation(data)
