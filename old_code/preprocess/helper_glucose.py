import os
import glob
import pathlib
import numpy as np
import pandas as pd

## asuming the file you have is glucose.csv
df1 = pd.read_csv("/Users/darpitdave/Downloads/sense/data/cohort1/glucose.csv",delimiter=',')
df1 = df1[df1['Event Type'] == 'EGV'].reset_index(drop = True) 
df1['Glucose Value (mg/dL)'] = df1['Glucose Value (mg/dL)'].str.replace('Low','40')
df1['Glucose Value (mg/dL)'] = df1['Glucose Value (mg/dL)'].str.replace('High','400')

df1 = df1[['Timestamp (YYYY-MM-DDThh:mm:ss)','Glucose Value (mg/dL)']]
df1.columns = ['Timestamp','glucose']

df1['Timestamp'] = pd.to_datetime(df1['Timestamp'],format = '%Y-%m-%dT%H:%M:%S')
df1['glucose'] = df1['glucose'].astype(float)
df1 = df1.sort_values('Timestamp').reset_index(drop = True)

df1.to_pickle('path-to-store-glucose-file/glucose.pkl')
