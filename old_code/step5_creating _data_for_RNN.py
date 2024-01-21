
### adopted from /home/grads/k/kathan/Warwick/QT_correction/modelling/auto_regV2.py
### takes in flaten output from CNN model and uses interpolation to feed in missing values. 
### Applies that to all dimensions and also to rr time series and time and creates data for RNN models.

#%%
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from numpy import save


c=2
s=5


dtp = ['train','test']
#dat_type = 'test'
for dat_type in dtp:
    for fld in range(1, 6):
        print(f'Processing fld: {fld}')
        # read_path = '/mnt/nvme-data1/Kathan/QT_correction/embeds/'
        read_path = '/mnt/nvme-data1/Kathan/QT_correction/embeds_hyper/'
        # Load data
        tr_embed_df = pd.read_pickle(read_path + f'c{c}s{s:02d}__{dat_type}_flatten30_{fld}_.pkl')
        print(f'Embeds shape: {tr_embed_df.shape}')
        # rp = '/mnt/nvme-data1/Kathan/QT_correction/{dat_type}s/'
        rp = f'/mnt/nvme-data1/Kathan/QT_correction/{dat_type}s_hyper/'
        trdf = pd.read_pickle(rp + f'c{c}s{s:02d}_{dat_type}_{fld-1}.pkl')
        print(f'tr data shape: {trdf.shape}')
        df = pd.concat([tr_embed_df.reset_index(drop=True), trdf.iloc[:, -31:].reset_index(drop=True)], axis=1)
        print(f'Concatenated data: {df.shape}')

        groups = df.groupby('flag')

        # Initialize an empty list to store the imputed dataframes for each group
        imputed_dfs = []

        # Imputation loop
        for group_name, group_df in groups:
            avg_rr = group_df['rr'].mean()
            group_df['rr_shifted'] = group_df['rr'].shift(1)
            group_df['rr_ratio'] = group_df['rr'] / avg_rr
            missing_rows = group_df[group_df['rr_ratio'] > 1.2][1:-1]
            missing_rows['n_missing'] = np.floor(missing_rows['rr_ratio']).astype(int)

            imputed_rows = pd.DataFrame()  # Initialize DataFrame to store imputed rows for current group

            # Step 3-6: Iterate over each group in missing_rows and impute missing rows
            for _, missing_group in missing_rows.iterrows():
                n_missing = missing_group['n_missing']
                try:
                    # Exclude first and last rows when performing linear interpolation
                    start_row = group_df.loc[missing_group.name - 1]
                    end_row = group_df.loc[missing_group.name]

                    # Perform linear interpolation for each column
                    interpolated_rows = pd.DataFrame()
                    for column in group_df.columns:
                        if column == 'rr':  # Skip the 'rr' column as it is used for identification
                            continue

                        # Determine the column data type and handle interpolation accordingly
                        column_data_type = group_df[column].dtype

                        if column_data_type == np.float64 or column_data_type == np.int64:
                            interpolated_values = np.linspace(start_row[column], end_row[column], num=n_missing + 2)[1:-1]
                        elif column_data_type == np.datetime64:
                            interpolated_values = pd.date_range(start_row[column], end_row[column], periods=n_missing + 2)[1:-1]
                        else:
                            continue  # Skip the column if it is not float, int, or timestamp

                        interpolated_rows[column] = interpolated_values

                    imputed_rows = imputed_rows.append(interpolated_rows)
                except:
                    continue

            imputed_dfs.append(imputed_rows)  # Append imputed rows for current group to imputed_dfs

        # Step 7: Merge imputed DataFrame with original DataFrame
        imputed_df = pd.concat([df] + imputed_dfs)

        # Step 8: Sort the merged DataFrame by index
        imputed_df.sort_index(inplace=True)

        # Preprocessing loop
        time_steps = 200
        processed_data = []
        rr_data = []
        rpk_data = []
        labels = []
        timestamp_data = []
        for _, group in groups:
            group_data = group.iloc[:, :192].values  # Select first 192 columns
            group_rr = group['rr'].values.reshape(-1, 1)  # Extract 'rr' column values
            #group_labels = group['hypo_label'].values  # Extract labels
            group_labels = group['hyper_label'].values  # Extract labels
            group_rpk = group['rpeak_idx'].values.reshape(-1, 1)  # Extract 'r peak' column values
            group_timestamp = group['Time'].values  # Extract 'timestamp' column values
            group_timestamp = np.reshape(group_timestamp, (-1, 1))  # reshape it to (-1, 1)

            for i in range(len(group_data) - time_steps, -1, -time_steps):
                if i + time_steps <= len(group_data):
                    sequence = group_data[i:i + time_steps]
                    rr_sequence = group_rr[i:i + time_steps]  # Get 'rr' values for the sequence
                    rpk_sequence = group_rpk[i:i + time_steps]  # Get 'r peaks' values for the sequence
                    timestamp_sequence = group_timestamp[i:i + time_steps]  # Get 'timestamp' values for the sequence
                    label = group_labels[i + time_steps - 1]  # Get label for the last row in the sequence

                    processed_data.append(sequence)
                    rr_data.append(rr_sequence)
                    rpk_data.append(rpk_sequence)
                    timestamp_data.append(timestamp_sequence)
                    labels.append(label)

        processed_data = np.array(processed_data)
        rr_data = np.array(rr_data)
        rpk_data = np.array(rpk_data)
        timestamp_data = np.array(timestamp_data)
        labels = np.array(labels).reshape(-1, 1)

        print(processed_data.shape)
        print(rr_data.shape)
        print(rpk_data.shape)
        print(timestamp_data.shape)
        print(labels.shape)

        # Save processed data
        #path = '/mnt/nvme-data1/Kathan/QT_correction/rnn_data/'
        path = '/mnt/nvme-data1/Kathan/QT_correction/rnn_data_hyper/'
        save(path + f'c{c}s{s:02d}_{dat_type}_{fld}_X.npy', processed_data)
        save(path + f'c{c}s{s:02d}_{dat_type}_{fld}_RR.npy', rr_data)
        save(path + f'c{c}s{s:02d}_{dat_type}_{fld}_RPEAKS.npy', rpk_data)
        save(path + f'c{c}s{s:02d}_{dat_type}_{fld}_Y.npy', labels)
        save(path + f'c{c}s{s:02d}_{dat_type}_{fld}_TIMESTAMP.npy', timestamp_data)


        print(f'Processing for fld {fld} completed.')






        # for _, group in groups:
        #     group_data = group.iloc[:, :192].values  # Select first 192 columns
        #     group_rr = group['rr'].values.reshape(-1, 1)  # Extract 'rr' column values
        #     group_labels = group['hypo_label'].values  # Extract labels
        #     group_rpk = group['rpeak_idx'].values.reshape(-1, 1)  # Extract 'r peak' column values

        #     for i in range(len(group_data) - time_steps, -1, -time_steps):
        #         if i + time_steps <= len(group_data):
        #             sequence = group_data[i:i + time_steps]
        #             rr_sequence = group_rr[i:i + time_steps]  # Get 'rr' values for the sequence
        #             rpk_sequence = group_rpk[i:i + time_steps] # Get 'r peaks' values for the sequence
        #             label = group_labels[i + time_steps - 1]  # Get label for the last row in the sequence

        #             processed_data.append(sequence)
        #             rr_data.append(rr_sequence)
        #             rpk_data.append(rpk_sequence)
        #             labels.append(label)

        # processed_data = np.array(processed_data)
        # rr_data = np.array(rr_data)
        # rpk_data = np.array(rpk_data)
        # labels = np.array(labels).reshape(-1, 1)

        # print(processed_data.shape)
        # print(rr_data.shape)
        # print(rpk_data.shape)
        # print(labels.shape)

        # # Save processed data
        # save(f'/mnt/nvme-data1/Kathan/QT_correction/rnn_data/c{c}s{s:02d}_{dat_type}_{fld}_X.npy', processed_data)
        # save(f'/mnt/nvme-data1/Kathan/QT_correction/rnn_data/c{c}s{s:02d}_{dat_type}_{fld}_RR.npy', rr_data)
        # save(f'/mnt/nvme-data1/Kathan/QT_correction/rnn_data/c{c}s{s:02d}_{dat_type}_{fld}_RPEAKS.npy', rpk_data)
        # save(f'/mnt/nvme-data1/Kathan/QT_correction/rnn_data/c{c}s{s:02d}_{dat_type}_{fld}_Y.npy', labels)

        # print(f'Processing for fld {fld} completed.')

print('All fld values processed.')

# %%
