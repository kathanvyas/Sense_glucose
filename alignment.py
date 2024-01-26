import os
import tqdm
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def show_aligned_demo(row_data, row_data_aligned, sampling_rate=250):
    ecg_data = np.array(row_data[:sampling_rate])
    aligned_ecg_data = np.array(row_data_aligned[:sampling_rate])

    plt.figure(figsize=(12, 6))  # Adjust figure size
    plt.plot(np.arange(ecg_data.shape[0]), ecg_data, label="ECG Original", color='grey')
    plt.plot(np.arange(aligned_ecg_data.shape[0]), aligned_ecg_data, label="ECG Aligned", color='cyan')

    # Scatter plots with different markers and colors
    plt.scatter(row_data['p'], ecg_data[int(row_data['p'])], color='green', marker='o', label="P Original")
    plt.scatter(row_data['q'], ecg_data[int(row_data['q'])], color='blue', marker='x', label="Q Original")
    plt.scatter(row_data['r'], ecg_data[int(row_data['r'])], color='red', marker='^', label="R Original")
    plt.scatter(row_data['t'], ecg_data[int(row_data['t'])], color='purple', marker='s', label="T Original")

    # Similar for aligned data with different marker styles
    plt.scatter(row_data_aligned['p'], aligned_ecg_data[int(row_data_aligned['p'])], color='green', marker='P', label="P Aligned")
    plt.scatter(row_data_aligned['q'], aligned_ecg_data[int(row_data_aligned['q'])], color='blue', marker='X', label="Q Aligned")
    plt.scatter(row_data_aligned['r'], aligned_ecg_data[int(row_data_aligned['r'])], color='red', marker='v', label="R Aligned")
    plt.scatter(row_data_aligned['t'], aligned_ecg_data[int(row_data_aligned['t'])], color='purple', marker='*', label="T Aligned")

    # Mark the R peak vertical line
    plt.axvline(x=row_data['r'], color='r', linestyle='--', label="R Peak Original")
    plt.axvline(x=row_data_aligned['r'], color='b', linestyle='--', label="R Peak Aligned")

    plt.legend(loc='upper right')  # Adjust legend position
    plt.title("ECG Peak: Original({}) vs Aligned({})".format(row_data['r'], row_data_aligned['r']))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig("./demo/alignment.png", dpi=300)  # Save with high DPI    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("This is the script for aligning the R peaks for each ECG signal")
    parser.add_argument('--ecg', type=str, help='path to the processed ecg data')
    parser.add_argument('--out_folder', default="/mnt/data2/mtseng/dataset/SeNSE/TCH_aligned", type=str, help='path to the output aligned R peak folder')
    args = parser.parse_args()

    print("Reading data from {}".format(args.ecg))
    df = pd.read_pickle(args.ecg)
    # I think its already sorted, but just in case
    df.sort_values('Time', inplace=True) 

    # remove NaN
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column].fillna(pd.Timestamp('1900-01-01'), inplace=True)  # Or pd.NaT
        elif pd.api.types.is_numeric_dtype(df[column]):
            df[column].fillna(0, inplace=True)
        else:
            df[column].fillna('Missing', inplace=True)


    filename = os.path.basename(args.ecg).split('.')[0]

    if not os.path.exists(args.out_folder): 
        os.mkdir(args.out_folder)
    out_dir = os.path.join(args.out_folder, filename)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # get the R peak's position
    aligned_r_peak_pos = int(df['r'].to_numpy().mean())
    print("R peak's avg position: ", aligned_r_peak_pos)

    
    # create an empty dataframe to store the aligned R peaks, only keep the "glucose", "time" columns that are needed
    sampling_rate = 250
    columns = list(np.arange(sampling_rate))
    columns.extend(['p', 'q', 'r', 't', 'Time', 'glucose', 'flag', 'hypo_label'])
    df_aligned = pd.DataFrame(columns=columns)

    # iterate through each row in df
    for i in tqdm.tqdm(range(len(df))):
        row_data = df.iloc[i]
        r_peak = df.iloc[i]['r']
        displacement = aligned_r_peak_pos - r_peak

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

        df_aligned.loc[i] = aligned_row

    # save the aligned dataframe
    df_aligned.to_pickle(os.path.join(out_dir, "{}_aligned.pkl".format(filename)))

    # pick a random row to show the alignment
    random_row = np.random.randint(0, len(df_aligned))
    show_aligned_demo(df.iloc[random_row], df_aligned.iloc[random_row])

