import numpy as np
import pandas as pd

from app.serial.data_imputation import data_imputation


def categorical_to_dummy(data_frame: pd.DataFrame, col_name: str):
    """Replaces column categorical data with dummies"""
    dummies = pd.get_dummies(data_frame[col_name], prefix=col_name)
    merged = pd.concat([data_frame, dummies], axis='columns')
    merged = merged.drop(columns=[col_name])
    return merged


def create_df(filepath: str):
    df = pd.read_csv(
        filepath,
        dtype={'CMPLNT_NUM': 'str'},
        on_bad_lines='error',
        low_memory=False,
        parse_dates={
            'CMPLNT_FR': ['CMPLNT_FR_TM', 'CMPLNT_FR_DT'],
            'CMPLNT_TO': ['CMPLNT_TO_TM', 'CMPLNT_TO_DT'],
            'RP_DT': ['RPT_DT']
        }
    )
    # convert times to one format
    for t in ['CMPLNT_FR', 'CMPLNT_TO']:
        df[t] = pd.to_datetime(df[t], format='%H:%M:%S %m/%d/%Y', errors='coerce')

    # edit incorrect age groups
    for col in ['SUSP_AGE_GROUP', 'VIC_AGE_GROUP']:
        df.loc[df[col].isna() | df[col].str.isdigit() | df[col].str.startswith('-'), col] = np.NaN

    # edit hispanic race
    for col in ['VIC_RACE', 'SUSP_RACE']:
        df[col] = df[col].str.removesuffix(' HISPANIC')

    df = df[['CMPLNT_FR', 'BORO_NM', 'PREM_TYP_DESC',
             'VIC_AGE_GROUP', 'VIC_RACE', 'VIC_SEX',
             'SUSP_AGE_GROUP', 'SUSP_RACE', 'SUSP_SEX',
             'Latitude', 'Longitude']]
    return df


def nn_pipeline(df: pd.DataFrame):
    """Pipeline for neural network to predict 'SUSP_RACE', 'SUSP_AGE_GROUP' and 'SUSP_SEX'"""

    # replace unknowns
    for col in ['VIC_RACE', 'VIC_AGE_GROUP', 'VIC_SEX',
                'SUSP_RACE', 'SUSP_AGE_GROUP', 'SUSP_SEX']:
        df.loc[df[col].isin(['UNKNOWN', 'U']), col] = np.NaN

    # get only rows with labels
    for col in ['SUSP_RACE', 'SUSP_AGE_GROUP', 'SUSP_SEX']:
        df = df[df[col].notna()]

    # convert categorical to dummy (binary list)
    for col in ['VIC_RACE', 'VIC_AGE_GROUP', 'VIC_SEX', 'BORO_NM', 'PREM_TYP_DESC']:
        df = categorical_to_dummy(df, col)

    # extract hour and month from time
    df['HOUR'] = df['CMPLNT_FR'].dt.hour
    df['MONTH'] = df['CMPLNT_FR'].dt.month
    df = df.drop(columns=['CMPLNT_FR'])

    # extract target
    target = df[['SUSP_AGE_GROUP', 'SUSP_RACE', 'SUSP_SEX']].copy()
    for col in ['SUSP_AGE_GROUP', 'SUSP_RACE', 'SUSP_SEX']:
        df = df.drop(columns=col)

    # convert ordinal to int
    mapper_age = {'<18': 0, '18-24': 1, '25-44': 2, '45-64': 3, '65+': 4}
    target['SUSP_AGE_GROUP'] = target['SUSP_AGE_GROUP'].replace(mapper_age)

    # convert categorical to dummy (binary list)
    for col in ['SUSP_RACE', 'SUSP_SEX']:
        target = categorical_to_dummy(target, col)

    return df, target


def serial_pipeline(df: pd.DataFrame):
    data_imputation(df)
    # convert categorical to dummy (binary list)
    for col in ['VIC_RACE', 'VIC_AGE_GROUP', 'VIC_SEX',
                'SUSP_AGE_GROUP', 'SUSP_RACE', 'SUSP_SEX',
                'BORO_NM', 'PREM_TYP_DESC']:
        df = categorical_to_dummy(df, col)

    # extract hour and month from time
    df['HOUR'] = df['CMPLNT_FR'].dt.hour
    df['MONTH'] = df['CMPLNT_FR'].dt.month
    df = df.drop(columns=['CMPLNT_FR'])

    # extract target
    target = df[['SUSP_AGE_GROUP', 'SUSP_RACE', 'SUSP_SEX']].copy()
    for col in ['SUSP_AGE_GROUP', 'SUSP_RACE', 'SUSP_SEX']:
        df = df.drop(columns=col)

    return df, target
