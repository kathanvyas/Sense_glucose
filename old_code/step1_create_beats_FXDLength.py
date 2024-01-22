
### adopted from /home/grads/k/kathan/Warwick/QT_correction/parallel_process/data_build1_fixed_window.py
### Fixed window beats formation

#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.interpolate import interp1d
from skimage.util.shape import view_as_windows as viewW

c=1
s=1
read_path = f'/mnt/nvme-data1/Kathan/main_data/main_data_separate_files/cohort{c}/c{c}s{s:02d}/'
path = '/mnt/nvme-data1/Kathan/QT_correction/all peaks/'


df_ecg = pd.read_pickle(read_path + 'allraw_ECG.pkl')
df_ecg = df_ecg.reset_index(drop = True)
d = df_ecg['EcgWaveform']
d = np.asarray(d)
idx = np.arange(len(d))
ecg = nk.ecg_clean(d, sampling_rate=250)


# %%
check0 = pd.read_pickle(path + f"c{c}s{s:02d}/c{c}s{s:02d}.pkl")

#%%
check1 = check0
check1['rr'] = check1['time_r'].shift(-1) - check1['time_r']
check1['rr'] = check1['rr'].dt.total_seconds()
#%%
check1 = check1.dropna()
check2 = check1[(check1['rr'] > 0.3) & (check1['rr'] < 1.8)]
check2 = check2.reset_index(drop=True)
# %%

sampling_rate = 250
allbeats = []
beat =[]  #l_temp
rtime_list = []   #Times
ppeak_idx = []    #Ps
qpeak_idx = []    #Qs
rpeak_idx = []    #Rs
speak_idx = []    #Ss
tpeak_idx = []    #Ts
p_onset_idx = []  #P_onsets
t_ofset_idx = []  #T_ofsets
start_idx = []
rr_sec = []
samp_from_start = []
b = 3.5  ##this value should be more default=3   #[2,3,4]
a = 1.5   ##this value should be less default=2
for i in range(check2.shape[0]):
    num_samples_before = 100
    num_samples_after = 100
    start = check2.at[i,'idx_r'] - num_samples_before
    end = check2.at[i,'idx_r'] + num_samples_after
    beat.append(ecg[start:end])
    rtime_list.append(check2.at[i,'time_r'])
    start_idx.append(start)
    rr_sec.append(check2.at[i,'rr'])
    ppeak_idx.append(check2.at[i,'idx_p'])
    rpeak_idx.append(check2.at[i,'idx_r'])
    qpeak_idx.append(check2.at[i,'idx_q'])
    speak_idx.append(check2.at[i,'idx_s'])
    tpeak_idx.append(check2.at[i,'idx_t'])
    p_onset_idx.append(check2.at[i,'idx_p_onset'])
    t_ofset_idx.append(check2.at[i,'idx_t_offset'])
    samp_from_start.append(num_samples_before)
beats = pd.DataFrame(beat) 
beats['start_idx'] = start_idx
beats['start_samp'] = samp_from_start
beats['ppeak_idx'] = ppeak_idx
beats['qpeak_idx'] = qpeak_idx
beats['rpeak_idx'] = rpeak_idx
beats['speak_idx'] = speak_idx
beats['tpeak_idx'] = tpeak_idx
beats['p_onset_idx'] = p_onset_idx
beats['t_ofset_idx'] = t_ofset_idx
beats['rr'] = rr_sec
beats['Time'] = rtime_list
#beats['rpk_index'] = rpks
allbeats.append(beats)
df_ecg_final = pd.concat(allbeats).reset_index(drop = True)
if 'Time' in df_ecg_final.columns:
    print('Time is present')

#%%
df_ecg_final['p'] = df_ecg_final['ppeak_idx'] - df_ecg_final['start_idx']
df_ecg_final['q'] = df_ecg_final['qpeak_idx'] - df_ecg_final['start_idx']
df_ecg_final['r'] = df_ecg_final['rpeak_idx'] - df_ecg_final['start_idx']
df_ecg_final['t'] = df_ecg_final['tpeak_idx'] - df_ecg_final['start_idx']

# %%
df_ecg_final.to_pickle(path + f"c{c}s{s:02d}/c{c}s{s:02d}_beats_fixedwindow.pkl")

#%%
#df_ecg_final.rename({'a': 'X', 'b': 'Y'}, axis=1)
def combine_with_summary_glucose(df_ecg_fnl):
    sum_read_path = f'/mnt/nvme-data1/Kathan/main_data/processed_separates/cohort{c}/c{c}s{s:02d}/'
    summary_df = pd.read_pickle(sum_read_path + '_combinedSUMMARY.pkl')
    summary_df['Time']= pd.to_datetime(summary_df['Time'], format='%d/%m/%Y %H:%M:%S.%f' )
    print('summary and ecg read')
    df_combined = pd.merge_asof(df_ecg_fnl.sort_values('Time'),summary_df.sort_values('Time'),on='Time',tolerance=pd.Timedelta('1 sec'),direction='nearest',allow_exact_matches=True)
    #df_combined = pd.concat([df_combined.iloc[:, :time_col_index], df_combined.iloc[:, -8:]], axis=1)
    df_combined = df_combined[(df_combined['HRConfidence'] >= 100) & (df_combined['ECGNoise'] < 0.001)]
    df_combined = df_combined.sort_values(by='Time', ascending=True)#, na_position='first')

    gl_read_path = f'/mnt/nvme-data1/Kathan/main_data/main_data_separate_files/cohort{c}/'
    gDF = pd.read_csv(gl_read_path + f'glucose_cohort{c}_check_raw.csv' )
    gDF = gDF[['Timestamp','glucose','patient']]
    if c>1:
        gDF = gDF[gDF['patient']==s+5]
    else:
        gDF = gDF[gDF['patient']==s]
    gDF = gDF.drop(columns=['patient']).reset_index(drop=True)
    gDF = gDF.replace({'Low': '40', 'High': '400'}).astype({'glucose': 'float64'})
    gDF.columns = ['time', 'glucose']
    gDF['Time'] = pd.to_datetime(gDF['time'],format = '%Y-%m-%d %H:%M:%S')
    gDF['flag'] = np.arange(1, gDF.shape[0]+1)
    finalDF = pd.merge_asof(df_combined.sort_values('Time'),gDF.sort_values('Time'),on='Time',tolerance=pd.Timedelta('330s'),direction='forward',allow_exact_matches=True)
    ### creating labels 
    f1df = finalDF
    hypothresh = 70
    hyperthresh = 180
    ### Adding hypo labels
    conditions = [(finalDF['glucose'] < hypothresh),(finalDF['glucose'] >= hypothresh) ]
    values = [1,0]
    finalDF['hypo_label'] = np.select(conditions, values)
    conditions = [(finalDF['glucose'] <= hyperthresh),(finalDF['glucose'] > hyperthresh) ]
    values = [1,0]
    finalDF['hypo_flag'] = np.select(conditions, values)
    return finalDF

finaldf = combine_with_summary_glucose(df_ecg_final)

#%%
# Check how many rows have negative values in all four columns separately
rows_with_negative_p = (finaldf['p'] < 0).sum()
rows_with_negative_q = (finaldf['q'] < 0).sum()
rows_with_negative_r = (finaldf['r'] < 0).sum()
rows_with_negative_t = (finaldf['t'] < 0).sum()

# Check how many rows have negative values in all four columns combined
rows_with_negative_all = ((finaldf['p'] < 0) & (finaldf['q'] < 0) & (finaldf['r'] < 0) & (finaldf['t'] < 0)).sum()

# Display the results
print(f"Rows with negative values in column 'p': {rows_with_negative_p}")
print(f"Rows with negative values in column 'q': {rows_with_negative_q}")
print(f"Rows with negative values in column 'r': {rows_with_negative_r}")
print(f"Rows with negative values in column 't': {rows_with_negative_t}")
print(f"Rows with negative values in all four columns: {rows_with_negative_all}")


#%%
finaldf = finaldf.sort_values('Time')
finaldf = finaldf.reset_index(drop=True)
finaldf.to_pickle(path + f"c{c}s{s:02d}/c{c}s{s:02d}_wholeDF_fixedwindow.pkl")



