import argparse
import os
import time
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nsml
from nsml import DATASET_PATH


def body_index(data, version):
    '''
    version=1: BMI만 사용, version=2: Ht, Wt만 사용
    '''
    if 'BMI' in data.columns:
        pass
    else:
        DROP_COLS = ['Ht', 'Wt']
        data1 = data.drop(columns=DROP_COLS).copy()
        return data1

    ht=data['Ht']
    wt=data['Wt']
    bmi=data['BMI']
    DROP_COLS = []
    # X = data.drop(columns=DROP_COLS).copy()
    if version == 1:
        DROP_COLS = ['Ht', 'Wt']
    elif version == 2:
         DROP_COLS = ['BMI']

    data1 = data.drop(columns=DROP_COLS).copy()
    return data1

def heart_index(data, version):
    '''
    version=1: MAP로 바꿔서 사용하고 나머지 drop
    '''
    sbp=data['SBP']
    dbp=data['DBP']
    pr=data['PR']
    DROP_COLS = []
    if version == 1:
        #MAP = (sbp+2dbp)/3
        data['heart']=(sbp+2*dbp)/3
        DROP_COLS = ['SBP','DBP','PR']

    data1 = data.drop(columns=DROP_COLS).copy()
    return data1

def liver_index(data, version):
    '''
    version=1: hepatic steatosis index(HSI)를 구해서 사용
    '''
    AST=data['AST']
    ALT=data['ALT']
    BMI=data['BMI']
    gender=data['gender_enc']
    DROP_COLS = []

    if version == 1:
        #hepatic steatosis index(HSI) = 8*ALT/AST+BMI+2(if female)
        data['HSI']=8*ALT/AST+BMI+2*gender
        DROP_COLS = ['AST','ALT','BMI']
    data1 = data.drop(columns=DROP_COLS).copy()
    return data1

def kidney_index(data, version):
    '''
    version=1: CrCl만 사용, version2: BUN, Cr을 사용
    '''
    BUN=data['BUN']
    Cr=data['Cr']
    CrCl=data['CrCl']
    DROP_COLS = []
    if version == 1:
        DROP_COLS = ['BUN','Cr']
    elif version == 2:
        DROP_COLS = ['CrCl']
    data1 = data.drop(columns=DROP_COLS).copy()
    return data1


def preproc_data(data, drop_which=[] ,label=None, train=True, val_ratio=0.2, seed=1234):
    if train:
        dataset = dict()

        # NaN 값 0으로 채우기
        data = data.fillna(0)

        # 성별 ['M', 'F'] -> [0, 1]로 변환
        data['gender_enc'] = np.where(data['gender'] == 'M', 0, 1)

        # 날짜 datetime으로 변환
        # df.loc[:, 'date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        DROP_COLS = ['CDMID', 'gender', 'date', 'date_E']
        X = data.drop(columns=DROP_COLS).copy()
        y = label

        # modifying data by jenny
        # drop_which = [body, heart, kidney, liver]
        X = liver_index(X, drop_which[3])
        X = body_index(X, drop_which[0])
        X = heart_index(X, drop_which[1])
        X = kidney_index(X, drop_which[2])


        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                          stratify=y,
                                                          test_size=val_ratio,
                                                          random_state=seed,
                                                          )

        X_train = torch.as_tensor(X_train.values).float()
        y_train = torch.as_tensor(y_train.reshape(-1, 1)).float()
        X_val = torch.as_tensor(X_val.values).float()
        y_val = torch.as_tensor(y_val.reshape(-1, 1)).float()

        dataset['train'] = TensorDataset(X_train, y_train)
        dataset['val'] = TensorDataset(X_val, y_val)

        return dataset

    else:
        # NaN 값 0으로 채우기
        data = data.fillna(0)

        # 성별 ['M', 'F'] -> [0, 1]로 변환
        data['gender_enc'] = np.where(data['gender'] == 'M', 0, 1)

        # 날짜 datetime으로 변환
        # df.loc[:, 'date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        DROP_COLS = ['CDMID', 'gender', 'date', 'date_E']
        data = data.drop(columns=DROP_COLS).copy()

        X_test = torch.as_tensor(data.values).float()

        return X_test
