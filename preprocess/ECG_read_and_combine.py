import pandas as pd
import os
import matplotlib.pyplot as plt

ecg_combined_df = pd.DataFrame()
# Iterate over files in the folder
for file in os.listdir(folder_path):
    if file.endswith('_ECG.csv'):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        ecg_combined_df = pd.concat([combined_df, df])

ecg_combined_df.to_pickle('path-to-store-combined-ECG-file/ECG.pkl')

summary_combined_df = pd.DataFrame()
# Iterate over files in the folder
for file in os.listdir(folder_path):
    if file.endswith('_SummaryEnhanced.csv'):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        summary_combined_df = pd.concat([combined_df, df])

summary_combined_df.to_pickle('path-to-store-combined-summary-file/summary.pkl')
