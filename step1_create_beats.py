
### adopted from /home/grads/k/kathan/Warwick/QT_correction/parallel_process/data_build1.py
### Variable window beats formation
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


#%%
df_ecg = pd.read_pickle(read_path + 'allraw_ECG.pkl')
df_ecg = df_ecg.reset_index(drop = True)
d = df_ecg['EcgWaveform']
d = np.asarray(d)
idx = np.arange(len(d))
ecg = nk.ecg_clean(d, sampling_rate=250)


#%%
_, rpeaks = nk.ecg_peaks(ecg, sampling_rate=250,correct_artifacts=True)
l_peaks = np.unique(rpeaks['ECG_R_Peaks'])   #np.unique(rpeaks['ECG_R_Peaks'])  rpeaks['ECG_R_Peaks']
arr_time = np.array(df_ecg['Time'])
_, waves_peak = nk.ecg_delineate(ecg, np.unique(rpeaks['ECG_R_Peaks']), sampling_rate=250, method="peak")

#%%

########
time_p = []
time_q = []
time_r = []
time_s = []
time_t = []

time_p_onset = []
time_t_offset = []

ecg_p = []
ecg_q = []
ecg_r = []
ecg_s = []
ecg_t = []

ecg_p_onset = []
ecg_t_offset = []

idx_p = []
idx_q = []
idx_r = []
idx_s = []
idx_t = []

idx_p_onset = []
idx_t_offset = []

for i in range(len(l_peaks)):
    time_r.append(arr_time[l_peaks[i]])
    ecg_r.append(ecg[l_peaks[i]])
    idx_r.append(idx[l_peaks[i]])
    try:
        time_t.append(arr_time[waves_peak['ECG_T_Peaks'][i]])
        ecg_t.append(ecg[waves_peak['ECG_T_Peaks'][i]])
        idx_t.append(idx[waves_peak['ECG_T_Peaks'][i]])
    except:
        time_t.append(np.nan)
        ecg_t.append(np.nan)
        idx_t.append(np.nan)
    try:
        time_p.append(arr_time[waves_peak['ECG_P_Peaks'][i]])
        ecg_p.append(ecg[waves_peak['ECG_P_Peaks'][i]])
        idx_p.append(idx[waves_peak['ECG_P_Peaks'][i]])
    except:
        time_p.append(np.nan)
        ecg_p.append(np.nan)
        idx_p.append(np.nan)
    try:
        time_q.append(arr_time[waves_peak['ECG_Q_Peaks'][i]])
        ecg_q.append(ecg[waves_peak['ECG_Q_Peaks'][i]])
        idx_q.append(idx[waves_peak['ECG_Q_Peaks'][i]])
    except:
        time_q.append(np.nan)
        ecg_q.append(np.nan)
        idx_q.append(np.nan)
    try:
        time_s.append(arr_time[waves_peak['ECG_S_Peaks'][i]])
        ecg_s.append(ecg[waves_peak['ECG_S_Peaks'][i]])
        idx_s.append(idx[waves_peak['ECG_S_Peaks'][i]])
    except:
        time_s.append(np.nan)
        ecg_s.append(np.nan)
        idx_s.append(np.nan)

    try:
        time_p_onset.append(arr_time[waves_peak['ECG_P_Onsets'][i]])
        ecg_p_onset.append(ecg[waves_peak['ECG_P_Onsets'][i]])
        idx_p_onset.append(idx[waves_peak['ECG_P_Onsets'][i]])
    except:
        time_p_onset.append(np.nan)
        ecg_p_onset.append(np.nan)
        idx_p_onset.append(np.nan)
    try:
        time_t_offset.append(arr_time[waves_peak['ECG_T_Offsets'][i]])
        ecg_t_offset.append(ecg[waves_peak['ECG_T_Offsets'][i]])
        idx_t_offset.append(idx[waves_peak['ECG_T_Offsets'][i]])
    except:
        time_t_offset.append(np.nan)
        ecg_t_offset.append(np.nan)
        idx_t_offset.append(np.nan)
d = {'r_peaks':ecg_r,
'p_peaks':ecg_p,
'q_peaks':ecg_q,
's_peaks':ecg_s,
't_peaks':ecg_t,
'p_onset':ecg_p_onset,
't_offset':ecg_t_offset,
'time_r':time_r,
'time_p':time_p,
'time_q':time_q,
'time_s':time_s,
'time_t':time_t,
'time_p_onset':time_p_onset,
'time_t_offset':time_t_offset,
'idx_r':idx_r,
'idx_p':idx_p,
'idx_q':idx_q,
'idx_s':idx_s,
'idx_t':idx_t,
'idx_p_onset':idx_p_onset,
'idx_t_offset':idx_t_offset}
check = pd.DataFrame(d)

check.to_pickle(f"/mnt/nvme-data1/Kathan/ECGContext/all peaks/c{c}s{s:02d}/c{c}s{s:02d}.pkl")

# %%
check0 = pd.read_pickle(f"/mnt/nvme-data1/Kathan/ECGContext/all peaks/c{c}s{s:02d}/c{c}s{s:02d}.pkl")

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
    num_samples_before = int(np.ceil((check2.at[i,'rr']/b)*sampling_rate))
    num_samples_after = int(np.ceil((check2.at[i,'rr']/a)*sampling_rate))
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
df_ecg_final.to_pickle(f"/mnt/nvme-data1/Kathan/ECGContext/all peaks/c{c}s{s:02d}/c{c}s{s:02d}_beats.pkl")

#%%
row_index = 350000
valp = df_ecg_final.at[row_index,'p']
valq = df_ecg_final.at[row_index,'q']
valr = df_ecg_final.at[row_index,'r']
valt = df_ecg_final.at[row_index,'t']
# Plot line chart for the selected row
plt.plot(df_ecg_final.iloc[row_index,:200])
plt.scatter(valp,df_ecg_final.at[row_index,valp],c='black',s=20)
plt.scatter(valq,df_ecg_final.at[row_index,valq],c='black',s=20)
plt.scatter(valr,df_ecg_final.at[row_index,valr],c='black',s=20)
plt.scatter(valt,df_ecg_final.at[row_index,valt],c='black',s=20)
# Plot scatter plot with black dots of size 20

#%%
#df_ecg_final.rename({'a': 'X', 'b': 'Y'}, axis=1)
def combine_with_summary_glucose(df_ecg_fnl):
    sum_read_path = f'/mnt/nvme-data1/Kathan/main_data/processed_separates/cohort{c}/c{c}s{s:02d}/'
    summary_df = pd.read_pickle(sum_read_path + '_combinedSUMMARY.pkl')
    summary_df['Time']= pd.to_datetime(summary_df['Time'], format='%d/%m/%Y %H:%M:%S.%f' )
    print('summary and ecg read')
    df_combined = pd.merge_asof(df_ecg_fnl.sort_values('Time'),summary_df.sort_values('Time'),on='Time',tolerance=pd.Timedelta('1 sec'),direction='nearest',allow_exact_matches=True)
    time_col_index = df_combined.columns.get_loc('Time')
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
finaldf = pd.concat([finaldf.iloc[:, :250], finaldf.iloc[:, -27:]], axis=1)

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
#finaldf = finaldf[(finaldf['p'] >= 0) & (finaldf['q'] >= 0) & (finaldf['r'] >= 0) & (finaldf['t'] >= 0)]
finaldf = finaldf[(finaldf['r'] >= 0) & (finaldf['t'] >= 0)]
finaldf = finaldf.sort_values('Time')
finaldf = finaldf.reset_index(drop=True)
finaldf.to_pickle(f"/mnt/nvme-data1/Kathan/ECGContext/all peaks/c{c}s{s:02d}/c{c}s{s:02d}_wholeDF.pkl")


final_hypo = finaldf[finaldf['hypo_label'] == 1]
final_hypo.to_pickle(f"/mnt/nvme-data1/Kathan/ECGContext/all peaks/c{c}s{s:02d}/c{c}s{s:02d}_wholeDF_hypo.pkl")
final_normal = finaldf[finaldf['hypo_label'] == 0]
final_normal.to_pickle(f"/mnt/nvme-data1/Kathan/ECGContext/all peaks/c{c}s{s:02d}/c{c}s{s:02d}_wholeDF_normal.pkl")



#%%
