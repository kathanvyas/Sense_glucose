import os
import time
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt

def align_ecg(row_data, target_r_peak, sampling_rate=250):
    r_peak = row_data['r']
    displacement = target_r_peak - r_peak

    ecg_data = np.array(row_data[:sampling_rate])
    aligned_ecg_data = np.roll(ecg_data, displacement)
    if displacement > 0:
        aligned_ecg_data[:displacement] = 0
    elif displacement < 0:
        aligned_ecg_data[displacement:] = 0

    aligned_p = int(row_data['p'] + displacement)
    aligned_q = int(row_data['q'] + displacement)
    aligned_r = int(row_data['r'] + displacement)
    aligned_t = int(row_data['t'] + displacement)
    aligned_time = row_data['Time']
    aligned_glucose = row_data['glucose']
    aligned_flag = row_data['flag']
    aligned_hypo_label = row_data['hypo_label']

    # append the aligned data to the dataframe
    aligned_row = aligned_ecg_data.tolist()
    aligned_row.extend([aligned_p, aligned_q, aligned_r, aligned_t, aligned_time, aligned_glucose, aligned_flag, aligned_hypo_label])

    return aligned_row

def load_checkfile(ecg_df, out_path, regenerate=False):

    ecg_df = ecg_df.reset_index(drop=True)
    d = ecg_df['EcgWaveform']
    d = np.asarray(d)
    idx =  np.arange(len(d))
    # clean an ECG signal to remove noise and improve peak-detection accuracy
    ecg = nk.ecg_clean(d, sampling_rate=250)

    if regenerate:
        _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=250, correct_artifacts=True) # Find R-peaks using the default method ("neurokit").
        
        l_peaks = np.unique(rpeaks['ECG_R_Peaks'])
        arr_time = np.array(ecg_df['Time'])
        print("Generating waves_peak ~")
        start_time = time.time()
        # waves_peak: dictionary containing additional information. For derivative method, the dictionary contains the samples at which P-peaks, Q-peaks, S-peaks, T-peaks, P-onsets and T-offsets occur
        _, waves_peak = nk.ecg_delineate(ecg, l_peaks, sampling_rate=250, method="peak")
        print("--- {:.3f} seconds ---".format(time.time() - start_time))


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
        check.to_pickle(out_path)

    check = pd.read_pickle(out_path)

    # calculate rr interval
    check['time_r'] = pd.to_datetime(check['time_r'], dayfirst=True)
    check['rr'] = check['time_r'].shift(-1) - check['time_r']
    check['rr'] = check['rr'].dt.total_seconds()

    check = check.dropna()
    check = check[(check['rr'] > 0.3) & (check['rr'] < 1.8)]
    check = check.reset_index(drop=True)

    return ecg, check

def generate_beats(ecg, df, sampling_rate = 250):
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

    for i in range(df.shape[0]):
        num_samples_before = int(np.ceil((df.at[i,'rr']/b)*sampling_rate))
        num_samples_after = int(np.ceil((df.at[i,'rr']/a)*sampling_rate))
        start = df.at[i,'idx_r'] - num_samples_before
        end = df.at[i,'idx_r'] + num_samples_after
        beat.append(ecg[start:end])
        rtime_list.append(df.at[i,'time_r'])
        start_idx.append(start)
        rr_sec.append(df.at[i,'rr'])
        ppeak_idx.append(df.at[i,'idx_p'])
        rpeak_idx.append(df.at[i,'idx_r'])
        qpeak_idx.append(df.at[i,'idx_q'])
        speak_idx.append(df.at[i,'idx_s'])
        tpeak_idx.append(df.at[i,'idx_t'])
        p_onset_idx.append(df.at[i,'idx_p_onset'])
        t_ofset_idx.append(df.at[i,'idx_t_offset'])
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

    return df_ecg_final

def plot_beat(beats, row_index, out_path, num_samples=200):
    # Plot the ECG beat
    print("Show sample ECG beat ...")
    # Get the values from the selected row
    valp = beats.at[row_index,'p']
    valq = beats.at[row_index,'q']
    valr = beats.at[row_index,'r']
    valt = beats.at[row_index,'t']
    print("Row: {}, p: {}, q: {}, r: {}, t: {}".format(row_index, valp, valq, valr, valt))
    # Plot line chart for the selected row
    # create color dictionary
    color_dict = dict({'p':'red', 'q':'blue', 'r':'green', 's':'orange', 't':'purple'})
    plt.plot(beats.iloc[row_index,:num_samples])
    if valp >= 0 and valp < num_samples:
        plt.scatter(valp, beats.at[row_index,valp], c=color_dict['p'], s=20, label='p')
    if valq >= 0 and valq < num_samples:
        plt.scatter(valq, beats.at[row_index,valq], c=color_dict['q'], s=20, label='q')
    if valr >= 0 and valr < num_samples:
        plt.scatter(valr, beats.at[row_index,valr], c=color_dict['r'], s=20, label='r')
    if valt >= 0 and valt < num_samples:
        plt.scatter(valt, beats.at[row_index,valt], c=color_dict['t'], s=20, label='t')
    plt.legend()
    plt.title("Sample ECG beat: Row {}".format(row_index))
    plt.savefig(out_path)

def plot_alignment_beats(beats, row_indices, out_path, num_samples=200):
    # Plot the ECG beat
    print("Show sample ECG beat ...")
    # create color dictionary
    color_dict = dict({'p':'red', 'q':'blue', 'r':'green', 's':'orange', 't':'purple'})
    valps = []
    valqs = []
    valrs = []
    valts = []
    for row_index in row_indices:
        # Get the values from the selected row
        valp = beats.at[row_index,'p']
        valq = beats.at[row_index,'q']
        valr = beats.at[row_index,'r']
        valt = beats.at[row_index,'t']
        print("Row: {}, p: {}, q: {}, r: {}, t: {}".format(row_index, valp, valq, valr, valt))
        # Plot line chart for the selected row
        plt.plot(beats.iloc[row_index,:num_samples])
        if valp >= 0 and valp < num_samples:
            valps.append([valp, beats.at[row_index,valp]])
        if valq >= 0 and valq < num_samples:
            valqs.append([valq, beats.at[row_index,valq]])
        if valr >= 0 and valr < num_samples:
            valrs.append([valr, beats.at[row_index,valr]])
        if valt >= 0 and valt < num_samples:
            valts.append([valt, beats.at[row_index,valt]])
    
    valps = np.array(valps)
    valqs = np.array(valqs)
    valrs = np.array(valrs)
    valts = np.array(valts)

    plt.scatter(valps[:,0], valps[:,1], c=color_dict['p'], s=20, label='p')
    plt.scatter(valqs[:,0], valqs[:,1], c=color_dict['q'], s=20, label='q')
    plt.scatter(valrs[:,0], valrs[:,1], c=color_dict['r'], s=20, label='r')
    plt.scatter(valts[:,0], valts[:,1], c=color_dict['t'], s=20, label='t')

    plt.legend()
    plt.title("Sample Aligned ECG beats")
    plt.savefig(out_path)


def combine_with_summary_glucose(ecg_df_final, summary_df, glucose_df):

    summary_df['Time']= pd.to_datetime(summary_df['Time'], format='%d/%m/%Y %H:%M:%S.%f' )
    df_combined = pd.merge_asof(ecg_df_final.sort_values('Time'),summary_df.sort_values('Time'),on='Time',tolerance=pd.Timedelta('1 sec'),direction='nearest',allow_exact_matches=True)
    df_combined = df_combined[(df_combined['HRConfidence'] >= 100) & (df_combined['ECGNoise'] < 0.001)]
    df_combined = df_combined.sort_values(by='Time', ascending=True)

    # create add Time and flag to glucose_df
    glucose_df.columns = ['time', 'glucose']
    glucose_df['Time'] = pd.to_datetime(glucose_df['time'],format='%Y-%m-%d %H:%M:%S')
    glucose_df['flag'] = np.arange(1, glucose_df.shape[0]+1)

    final_df = pd.merge_asof(df_combined.sort_values('Time'), glucose_df.sort_values('Time'), on='Time', tolerance=pd.Timedelta('330s'), direction='forward', allow_exact_matches=True)

    ### creating labels 
    hypothresh = 70
    hyperthresh = 180
    ### Adding hypo labels
    conditions = [(final_df['glucose'] < hypothresh),(final_df['glucose'] >= hypothresh) ]
    values = [1,0]
    final_df['hypo_label'] = np.select(conditions, values)
    conditions = [(final_df['glucose'] <= hyperthresh),(final_df['glucose'] > hyperthresh) ]
    values = [1,0]
    final_df['hypo_flag'] = np.select(conditions, values)

    # find the index of "start_idx" column
    start_idx_col = list(final_df.columns).index('start_idx')
    final_df = pd.concat([final_df.iloc[:, :250], final_df.iloc[:, start_idx_col:]], axis=1)

    return final_df

def clean_negative_values(final_df, verbose=True):
    num_rows = final_df.shape[0]    
    # Check how many rows have negative values in all four columns separately
    rows_with_negative_p = (final_df['p'] < 0).sum()
    rows_with_negative_q = (final_df['q'] < 0).sum()
    rows_with_negative_r = (final_df['r'] < 0).sum()
    rows_with_negative_t = (final_df['t'] < 0).sum()

    # Check how many rows have negative values in all four columns combined
    rows_with_negative_all = ((final_df['p'] < 0) & (final_df['q'] < 0) & (final_df['r'] < 0) & (final_df['t'] < 0)).sum()

    final_df = final_df[(final_df['r'] >= 0) & (final_df['t'] >= 0)]
    final_df = final_df.sort_values('Time')
    final_df = final_df.reset_index(drop=True)
    # Display the results
    if verbose:
        print("Total number of rows removed (if r or t is smaller than 0): {}".format(num_rows - final_df.shape[0]))
        print(" - rows with negative values in column 'p': {}".format(rows_with_negative_p))
        print(" - rows with negative values in column 'q': {}".format(rows_with_negative_q))
        print(" - rows with negative values in column 'r': {}".format(rows_with_negative_r))
        print(" - rows with negative values in column 't': {}".format(rows_with_negative_t))
        print(" - rows with negative values in all four columns: {}".format(rows_with_negative_all))

    return final_df