
## adopted from /home/grads/k/kathan/Warwick/QT_correction/modelling/PCA_data_creation.py
## uses the PCA filtered beats and then does padding to make it fixed lengths for CNN

#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.interpolate import interp1d
from skimage.util.shape import view_as_windows as viewW

def create_filtered_dfs(df,HrRt,ttype='hypo'):
    HR1 = HrRt-5
    HR2 = HrRt+5
    print(f'extremes are {HR1} and {HR2}')
    df__ = df[(df['HR'] <= HR2) & (df['HR'] > HR1)]
    print(df__.shape)
    df__hypo = df__[df__[f'{ttype}_label'] == 1]
    df__normal = df__[df__[f'{ttype}_label'] == 0]
    return df__,df__hypo,df__normal

def interpolate_array_scipy(arr, new_length):
    old_length = len(arr)
    old_indices = np.arange(old_length)
    new_indices = np.linspace(0, old_length - 1, new_length)
    f = interp1d(old_indices, arr, kind='linear')
    new_arr = f(new_indices)
    return new_arr

def interpolate_dataframe(df, target_length=200):
    interpolated_rows = []
    for _, row in df.iterrows():
        non_na_values = row.dropna().values
        if len(non_na_values) > 0:
            interpolated_row = interpolate_array_scipy(non_na_values, target_length)
            interpolated_rows.append(interpolated_row)
    return pd.DataFrame(interpolated_rows)

def df_add_zero_padding_to_start(df, value):
    padding_map = {
        125: 40,
        115: 35,
        105: 35,
        95: 30,
        85: 25,
        75: 20,
        65: 10
    }
    padding = padding_map.get(value, 0)
    return df.applymap(lambda x: np.pad(x, (padding,), 'constant'))

def shift_r_peak(df, shift_array, max_shift=100):
    def shift_row(row, shift, max_shift):
        return [0] * (max_shift - shift) + list(row) #+ [0] * shift
    # Create an empty list to hold the shifted rows
    shifted_rows = []
    for idx, row in df.iterrows():
        if idx > df.shape[0]-1:
            continue
        shift = shift_array[idx]
        shifted_row = shift_row(row, shift, max_shift)
        shifted_rows.append(shifted_row)
    return pd.DataFrame(shifted_rows)

def strided_indexing_roll(a, r):
    # Concatenate with sliced to cover all rolls
    a_ext = np.concatenate((a,a[:,:-1]),axis=1)
    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = a.shape[1]
    return viewW(a_ext,(1,n))[np.arange(len(r)), (n-r)%n,0]


#%%
c=1
s=3
read_path = f'/mnt/nvme-data1/Kathan/QT_correction/PCA data/c{c}s{s:02d}'
#read_path = f'/mnt/nvme-data1/Kathan/QT_correction/PCA data_hyper/c{c}s{s:02d}'
# read_path_modified = f'/mnt/nvme-data1/Kathan/QT_correction/PCA data/c{c}s{s:02d}'
ecgdf = pd.read_pickle(read_path + '_filtered_df.pkl')
#ecgdf = pd.read_pickle(read_path_modified + '_filtered_df.pkl')
ecgdf = ecgdf.reset_index(drop=True)

old_columns = ecgdf.columns[:200]
new_columns = list(range(200))
column_mapping = dict(zip(old_columns, new_columns))
ecgdf = ecgdf.rename(columns=column_mapping)
#%%
seedDF = ecgdf.iloc[:,:-32] 

d25,d25_hypo,d25_normal = create_filtered_dfs(ecgdf,HrRt=25)
d35,d35_hypo,d35_normal = create_filtered_dfs(ecgdf,HrRt=35)
d45,d45_hypo,d45_normal = create_filtered_dfs(ecgdf,HrRt=45)
d55,d55_hypo,d55_normal = create_filtered_dfs(ecgdf,HrRt=55)
d65,d65_hypo,d65_normal = create_filtered_dfs(ecgdf,HrRt=65)
d75,d75_hypo,d75_normal = create_filtered_dfs(ecgdf,HrRt=75)
d85,d85_hypo,d85_normal = create_filtered_dfs(ecgdf,HrRt=85)
d95,d95_hypo,d95_normal = create_filtered_dfs(ecgdf,HrRt=95)
d105,d105_hypo,d105_normal = create_filtered_dfs(ecgdf,HrRt=105)
d115,d115_hypo,d115_normal = create_filtered_dfs(ecgdf,HrRt=115)
d125,d125_hypo,d125_normal = create_filtered_dfs(ecgdf,HrRt=125)
d135,d135_hypo,d135_normal = create_filtered_dfs(ecgdf,HrRt=135)
d145,d145_hypo,d145_normal = create_filtered_dfs(ecgdf,HrRt=145)
d155,d155_hypo,d155_normal = create_filtered_dfs(ecgdf,HrRt=155)
d165,d165_hypo,d165_normal = create_filtered_dfs(ecgdf,HrRt=165)

#%%
#######################
fig, axs = plt.subplots(15, 1, figsize=(10, 15), dpi=300, gridspec_kw={'height_ratios': [3, 3, 3, 3, 3, 3, 3,3,3,3,3,3,3,3,3]})

# plot mean values and shade the area within one standard deviation
nm = [25,35,45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155,165]
for i, df in enumerate([d25,d35,d45, d55, d65, d75, d85, d95, d105, d115, d125, d135, d145, d155, d165]):
    mean_vals = df.iloc[:,:-32].mean(axis=0)
    std_vals = df.iloc[:,:-32].std(axis=0)
    axs[i].plot(mean_vals, color='blue')
    axs[i].fill_between(range(len(mean_vals)), mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color='blue')
    axs[i].set_title(f'{nm[i]-5} to {nm[i]+5}')
    axs[i].set_ylabel('Voltage (mV)')
    #axs[i].text(0.02, 0.9, f'Total Beats: {df.shape[0]}', transform=axs[i].transAxes, fontsize=10, verticalalignment='top')
axs[-1].set_xlabel('')
# remove vertical spacing between subplots
fig.subplots_adjust(hspace=0)
fig.suptitle('Average Beat after interpolation')
# set bottom and top margins to zero
fig.subplots_adjust(bottom=0, top=1)
# adjust the spacing between subplots
fig.tight_layout()
#fig.text(0.5, 0.00001, 'ECG Sample ID', ha='center', fontsize=24)
#fig.text(0.001, 0.5, 'ECG Sample Amplitude', va='center', rotation='vertical', fontsize=24)
plt.xlabel('Time (s)')
# show the plot
plt.show()


#%%
### Interpolation
seed_interpolate = seedDF

df25__interpolate200 = interpolate_dataframe(d25.iloc[:,25:175],200)
df35__interpolate200 = interpolate_dataframe(d35.iloc[:,25:175],200)
df45__interpolate200 = interpolate_dataframe(d45.iloc[:,25:175],200)
df55__interpolate200 = interpolate_dataframe(d55.iloc[:,25:175],200)
df65__interpolate200 = interpolate_dataframe(d65.iloc[:,25:175],200)
df75__interpolate200 = interpolate_dataframe(d75.iloc[:,25:175],200)
df85__interpolate200 = interpolate_dataframe(d85.iloc[:,25:175],200)
df95__interpolate200 = interpolate_dataframe(d95.iloc[:,25:175],200)
df105__interpolate200 = interpolate_dataframe(d105.iloc[:,25:175],200)
df115__interpolate200 = interpolate_dataframe(d115.iloc[:,25:175],200)
df125__interpolate200 = interpolate_dataframe(d125.iloc[:,25:175],200)
df135__interpolate200 = interpolate_dataframe(d135.iloc[:,25:175],200)
df145__interpolate200 = interpolate_dataframe(d145.iloc[:,25:175],200)
df155__interpolate200 = interpolate_dataframe(d155.iloc[:,25:175],200)
df165__interpolate200 = interpolate_dataframe(d165.iloc[:,25:175],200)

#%%
interpolates = [df25__interpolate200,df35__interpolate200,df45__interpolate200,df55__interpolate200,df65__interpolate200,df75__interpolate200,df85__interpolate200,
                df95__interpolate200,df105__interpolate200,df115__interpolate200,df125__interpolate200,
                df135__interpolate200,df145__interpolate200,df155__interpolate200,df165__interpolate200]
final_df__interpolate200 = pd.concat(interpolates)

#%%
# final_df__interpolate200 = interpolate_dataframe(seed_interpolate,200)
final_df__interpolate200_all = pd.concat([final_df__interpolate200.reset_index(drop=True),ecgdf.iloc[:,-31:]],axis=1)

#%%

# df25,df25_hypo,df25_normal = create_filtered_dfs(final_df__interpolate200_all,HrRt=25)
# df35,df35_hypo,df35_normal = create_filtered_dfs(final_df__interpolate200_all,HrRt=35)
# df45,df45_hypo,df45_normal = create_filtered_dfs(final_df__interpolate200_all,HrRt=45)
# df55,df55_hypo,df55_normal = create_filtered_dfs(final_df__interpolate200_all,HrRt=55)
# df65,df65_hypo,df65_normal = create_filtered_dfs(final_df__interpolate200_all,HrRt=65)
# df75,df75_hypo,df75_normal = create_filtered_dfs(final_df__interpolate200_all,HrRt=75)
# df85,df85_hypo,df85_normal = create_filtered_dfs(final_df__interpolate200_all,HrRt=85)
# df95,df95_hypo,df95_normal = create_filtered_dfs(final_df__interpolate200_all,HrRt=95)
# df105,df105_hypo,df105_normal = create_filtered_dfs(final_df__interpolate200_all,HrRt=105)
# df115,df115_hypo,df115_normal = create_filtered_dfs(final_df__interpolate200_all,HrRt=115)
# df125,df125_hypo,df125_normal = create_filtered_dfs(final_df__interpolate200_all,HrRt=125)
# df135,df135_hypo,df135_normal = create_filtered_dfs(final_df__interpolate200_all,HrRt=135)
# df145,df145_hypo,df145_normal = create_filtered_dfs(final_df__interpolate200_all,HrRt=145)
# df155,df155_hypo,df155_normal = create_filtered_dfs(final_df__interpolate200_all,HrRt=155)
# df165,df165_hypo,df165_normal = create_filtered_dfs(final_df__interpolate200_all,HrRt=165)

# fig, axs = plt.subplots(15, 1, figsize=(10, 15), dpi=300, gridspec_kw={'height_ratios': [3, 3, 3, 3, 3, 3, 3,3,3,3,3,3,3,3,3]})

# # plot mean values and shade the area within one standard deviation
# nm = [25,35,45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155,165]
# for i, df in enumerate([df25,df35,df45, df55, df65, df75, df85, df95, df105, df115, df125, df135, df145, df155, df165]):
#     mean_vals = df.iloc[:,:-31].mean(axis=0)
#     std_vals = df.iloc[:,:-31].std(axis=0)
#     axs[i].plot(mean_vals, color='blue')
#     axs[i].fill_between(range(len(mean_vals)), mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color='blue')
#     axs[i].set_title(f'{nm[i]-5} to {nm[i]+5}')
#     axs[i].set_ylabel('Voltage (mV)')
#     #axs[i].text(0.02, 0.9, f'Total Beats: {df.shape[0]}', transform=axs[i].transAxes, fontsize=10, verticalalignment='top')
# axs[-1].set_xlabel('')
# # remove vertical spacing between subplots
# fig.subplots_adjust(hspace=0)
# fig.suptitle('Average Beat after interpolation')
# # set bottom and top margins to zero
# fig.subplots_adjust(bottom=0, top=1)
# # adjust the spacing between subplots
# fig.tight_layout()
# #fig.text(0.5, 0.00001, 'ECG Sample ID', ha='center', fontsize=24)
# #fig.text(0.001, 0.5, 'ECG Sample Amplitude', va='center', rotation='vertical', fontsize=24)
# plt.xlabel('Time (s)')
# # show the plot
# plt.show()


#%%
## Zero padding to right
seedrp = seedDF
final_df_rightpad = seedrp.fillna(0)
final_df_rightpad_all = pd.concat([final_df_rightpad,ecgdf.iloc[:,-31:]],axis=1)

#%%

dfrp25,df25_hypo,df25_normal = create_filtered_dfs(final_df_rightpad_all,HrRt=25)
dfrp35,df35_hypo,df35_normal = create_filtered_dfs(final_df_rightpad_all,HrRt=35)
dfrp45,df45_hypo,df45_normal = create_filtered_dfs(final_df_rightpad_all,HrRt=45)
dfrp55,df55_hypo,df55_normal = create_filtered_dfs(final_df_rightpad_all,HrRt=55)
dfrp55,df55_hypo,df55_normal = create_filtered_dfs(final_df_rightpad_all,HrRt=55)
dfrp55,df55_hypo,df55_normal = create_filtered_dfs(final_df_rightpad_all,HrRt=55)
dfrp65,df65_hypo,df65_normal = create_filtered_dfs(final_df_rightpad_all,HrRt=65)
dfrp75,df75_hypo,df75_normal = create_filtered_dfs(final_df_rightpad_all,HrRt=75)
dfrp85,df85_hypo,df85_normal = create_filtered_dfs(final_df_rightpad_all,HrRt=85)
dfrp95,df95_hypo,df95_normal = create_filtered_dfs(final_df_rightpad_all,HrRt=95)
dfrp105,df105_hypo,df105_normal = create_filtered_dfs(final_df_rightpad_all,HrRt=105)
dfrp115,df115_hypo,df115_normal = create_filtered_dfs(final_df_rightpad_all,HrRt=115)
dfrp125,df125_hypo,df125_normal = create_filtered_dfs(final_df_rightpad_all,HrRt=125)
dfrp135,df135_hypo,df135_normal = create_filtered_dfs(final_df_rightpad_all,HrRt=135)
dfrp145,df145_hypo,df145_normal = create_filtered_dfs(final_df_rightpad_all,HrRt=145)
dfrp155,df155_hypo,df155_normal = create_filtered_dfs(final_df_rightpad_all,HrRt=155)
dfrp165,df165_hypo,df165_normal = create_filtered_dfs(final_df_rightpad_all,HrRt=165)

fig, axs = plt.subplots(15, 1, figsize=(10, 15), dpi=300, gridspec_kw={'height_ratios': [3, 3, 3, 3, 3, 3, 3,3,3,3,3,3,3,3,3]})

# plot mean values and shade the area within one standard deviation
nm = [25,35,45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155,165]
for i, df in enumerate([dfrp25,dfrp35,dfrp45, dfrp55, dfrp65, dfrp75, dfrp85, dfrp95, dfrp105, dfrp115, dfrp125, dfrp135, dfrp145, dfrp155, dfrp165]):
    mean_vals = df.iloc[:,:-32].mean(axis=0)
    std_vals = df.iloc[:,:-32].std(axis=0)
    axs[i].plot(mean_vals, color='blue')
    axs[i].fill_between(range(len(mean_vals)), mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color='blue')
    axs[i].set_title(f'{nm[i]-5} to {nm[i]+5}')
    axs[i].set_ylabel('Voltage (mV)')
    #axs[i].text(0.02, 0.9, f'Total Beats: {df.shape[0]}', transform=axs[i].transAxes, fontsize=10, verticalalignment='top')
axs[-1].set_xlabel('')
# remove vertical spacing between subplots
fig.subplots_adjust(hspace=0)
fig.suptitle('Average Beat after interpolation')
# set bottom and top margins to zero
fig.subplots_adjust(bottom=0, top=1)
# adjust the spacing between subplots
fig.tight_layout()
#fig.text(0.5, 0.00001, 'ECG Sample ID', ha='center', fontsize=24)
#fig.text(0.001, 0.5, 'ECG Sample Amplitude', va='center', rotation='vertical', fontsize=24)
plt.xlabel('Time (s)')
# show the plot
plt.show()

# #%%
# max_shift = 100
# shifts = np.array(final_df_rightpad_all['start_samp'])
# r = max_shift-shifts
# out = pd.DataFrame(strided_indexing_roll(np.pad(final_df_rightpad.to_numpy(),
#                                                 ((0, 0), (0, r.max()))),r),index=final_df_rightpad.index)

# #%%
# final_df_leftpad_all = pd.concat([out,final_df_rightpad_all.iloc[:,-32:]],axis=1)

# #%%
# dflp25,dflp25_hypo,dflp25_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=25)
# dflp35,dflp35_hypo,dflp35_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=35)
# dflp45,dflp45_hypo,dflp45_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=45)
# dflp55,dflp55_hypo,dflp55_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=55)
# dflp65,dflp65_hypo,dflp65_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=65)
# dflp75,dflp75_hypo,dflp75_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=75)
# dflp85,dflp85_hypo,dflp85_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=85)
# dflp95,dflp95_hypo,dflp95_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=95)
# dflp105,dflp105_hypo,dflp105_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=105)
# dflp115,dflp115_hypo,dflp115_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=115)
# dflp125,dflp125_hypo,dflp125_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=125)
# dflp135,dflp135_hypo,dflp135_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=135)
# dflp145,dflp145_hypo,dflp145_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=145)
# dflp155,dflp155_hypo,dflp155_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=155)
# dflp165,dflp165_hypo,dflp165_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=165)

# fig, axs = plt.subplots(15, 1, figsize=(10, 15), dpi=300, gridspec_kw={'height_ratios': [3, 3, 3, 3, 3, 3, 3,3,3,3,3,3,3,3,3]})

# # plot mean values and shade the area within one standard deviation
# nm = [25,35,45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155,165]
# for i, df in enumerate([dflp25,dflp35,dflp45, dflp55, dflp65, dflp75, dflp85, dflp95, dflp105, dflp115, dflp125, dflp135, dflp145, dflp155, dflp165]):
#     mean_vals = df.iloc[:,:-32].mean(axis=0)
#     std_vals = df.iloc[:,:-32].std(axis=0)
#     axs[i].plot(mean_vals, color='blue')
#     axs[i].fill_between(range(len(mean_vals)), mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color='blue')
#     axs[i].set_title(f'{nm[i]-5} to {nm[i]+5}')
#     axs[i].set_ylabel('Voltage (mV)')
#     #axs[i].text(0.02, 0.9, f'Total Beats: {df.shape[0]}', transform=axs[i].transAxes, fontsize=10, verticalalignment='top')
# axs[-1].set_xlabel('')
# # remove vertical spacing between subplots
# fig.subplots_adjust(hspace=0)
# fig.suptitle('Average Beat after interpolation')
# # set bottom and top margins to zero
# fig.subplots_adjust(bottom=0, top=1)
# # adjust the spacing between subplots
# fig.tight_layout()
# #fig.text(0.5, 0.00001, 'ECG Sample ID', ha='center', fontsize=24)
# #fig.text(0.001, 0.5, 'ECG Sample Amplitude', va='center', rotation='vertical', fontsize=24)
# plt.xlabel('Time (s)')
# # show the plot
# plt.show()




#%%
wrp = '/mnt/nvme-data1/Kathan/QT_correction/pca_int_pad_data/'
#wrp = '/mnt/nvme-data1/Kathan/QT_correction/pca_int_pad_data_hyper/'
final_df__interpolate200_all.to_pickle(wrp + f'c{c}s{s:02d}_df_interpolate.pkl')
final_df_rightpad_all.to_pickle(wrp + f'c{c}s{s:02d}_df_pad.pkl')


#%%
#####################################################################################################


# #%%
# fig, axs = plt.subplots(8, 1, figsize=(10, 15), dpi=300, gridspec_kw={'height_ratios': [3, 3, 3, 3, 3, 3, 3, 3]})

# # Define the sampling rate
# sampling_rate = 250

# # Plot mean values and shade the area within one standard deviation
# nm = [125, 115, 105, 95, 85, 75, 65, 55]
# for i, df in enumerate([df125, df115, df105, df95, df85, df75, df65, df55]):
#     mean_vals = df.iloc[:, :-31].mean(axis=0)
#     std_vals = df.iloc[:, :-31].std(axis=0)

#     # Calculate time values
#     time = list(range(len(mean_vals)))
#     time = [(t - 60) / sampling_rate for t in time]

#     axs[i].plot(time, mean_vals, color='blue')
#     axs[i].fill_between(time, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color='blue')
#     axs[i].set_title(f'{nm[i] - 5} to {nm[i] + 5}')
#     axs[i].set_ylabel('Voltage (mV)')

# axs[-1].set_xlabel('Time (s)')
# fig.subplots_adjust(hspace=0)
# fig.suptitle('Average Beat after interpolation')
# fig.subplots_adjust(bottom=0, top=1)
# fig.tight_layout()

# # Show the plot
# plt.show()


# #%%

# ### HYPONORMAL HYPONORMAL HYPONORMAL HYPONORMAL HYPONORMAL HYPONORMAL HYPONORMAL HYPONORMAL HYPONORMAL HYPONORMAL HYPONORMAL 
# fig, axs = plt.subplots(8, 1, figsize=(10, 15), dpi=300, gridspec_kw={'height_ratios': [3, 3, 3, 3, 3, 3, 3, 3]})

# # Define the two lists of dataframes
# df_hypo = [df125_hypo, df115_hypo, df105_hypo, df95_hypo, df85_hypo, df75_hypo, df65_hypo, df55_hypo]
# df_normal = [df125_normal, df115_normal, df105_normal, df95_normal, df85_normal, df75_normal, df65_normal, df55_normal]

# # plot mean values and shade the area within one standard deviation for each set of dataframes
# nm = [125, 115, 105, 95, 85, 75, 65, 55]
# for i, (df_h, df_n) in enumerate(zip(df_hypo, df_normal)):
#     mean_vals_h = df_h.iloc[:, :-17].mean(axis=0)
#     std_vals_h = df_h.iloc[:, :-17].std(axis=0)
#     mean_vals_n = df_n.iloc[:, :-17].mean(axis=0)
#     std_vals_n = df_n.iloc[:, :-17].std(axis=0)
    
#     # Plot the first set of dataframes
#     axs[i].plot(mean_vals_h, color='blue')
#     axs[i].fill_between(range(len(mean_vals_h)), mean_vals_h - std_vals_h, mean_vals_h + std_vals_h, alpha=0.2, color='blue')
    
#     # Plot the second set of dataframes
#     axs[i].plot(mean_vals_n, color='red')
#     axs[i].fill_between(range(len(mean_vals_n)), mean_vals_n - std_vals_n, mean_vals_n + std_vals_n, alpha=0.2, color='red')
#     axs[i].set_ylabel('Voltage (mV)')
#     #axs[i].text(0.02, 0.9, f'Total Beats: {df.shape[0]}', transform=axs[i].transAxes, fontsize=10, verticalalignment='top')
#     axs[i].set_title(f'{nm[i]-5} to {nm[i]+5}')
#     axs[i].text(0.02, 0.9, f'Total Beats (Hypo): {df_h.shape[0]/(df_h.shape[0]+df_n.shape[0]):.2f}', transform=axs[i].transAxes, fontsize=10, verticalalignment='top')
#     #axs[i].text(0.02, 0.8, f'Total Beats (Normal): {df_n.shape[0]}', transform=axs[i].transAxes, fontsize=10, verticalalignment='top')

# axs[-1].set_xlabel('')
# fig.subplots_adjust(hspace=0)
# fig.suptitle('Average Beat after interpolation')
# fig.subplots_adjust(bottom=0, top=1)
# fig.tight_layout()
# #fig.text(0.5, 0.00001, 'ECG Sample ID', ha='center', fontsize=24)
# #fig.text(0.001, 0.5, 'ECG Sample Amplitude', va='center', rotation='vertical', fontsize=24)
# plt.xlabel('Time (s)')
# plt.show()

# #%%
# fig, axs = plt.subplots(8, 1, figsize=(10, 15), dpi=300, gridspec_kw={'height_ratios': [3, 3, 3, 3, 3, 3, 3, 3]})

# # Define the two lists of dataframes
# df_hypo = [df125_hypo, df115_hypo, df105_hypo, df95_hypo, df85_hypo, df75_hypo, df65_hypo, df55_hypo]
# df_normal = [df125_normal, df115_normal, df105_normal, df95_normal, df85_normal, df75_normal, df65_normal, df55_normal]

# # Define the sampling rate
# sampling_rate = 250

# # Plot mean values and shade the area within one standard deviation for each set of dataframes
# nm = [125, 115, 105, 95, 85, 75, 65, 55]
# for i, (df_h, df_n) in enumerate(zip(df_hypo, df_normal)):
#     mean_vals_h = df_h.iloc[:, :-31].mean(axis=0)
#     std_vals_h = df_h.iloc[:, :-31].std(axis=0)
#     mean_vals_n = df_n.iloc[:, :-31].mean(axis=0)
#     std_vals_n = df_n.iloc[:, :-17].std(axis=0)

#     # Calculate time values
#     time = list(range(len(mean_vals_h)))
#     time = [(t - 60) / sampling_rate for t in time]

#     # Plot the first set of dataframes
#     axs[i].plot(time, mean_vals_h, color='red', label='Hypo')
#     axs[i].fill_between(time, mean_vals_h - std_vals_h, mean_vals_h + std_vals_h, alpha=0.2, color='red')

#     # Plot the second set of dataframes
#     axs[i].plot(time, mean_vals_n, color='blue', label='Normal')
#     axs[i].fill_between(time, mean_vals_n - std_vals_n, mean_vals_n + std_vals_n, alpha=0.2, color='blue')
#     axs[i].set_ylabel('Voltage (mV)')
#     axs[i].set_title(f'{nm[i] - 5} to {nm[i] + 5}')
#     axs[i].text(0.02, 0.9, f'Total Beats (Hypo): {df_h.shape[0]/(df_h.shape[0]+df_n.shape[0]):.2f}', transform=axs[i].transAxes, fontsize=10, verticalalignment='top')

#     # Add legend
#     axs[i].legend()

# axs[-1].set_xlabel('Time (s)')
# fig.subplots_adjust(hspace=0)
# fig.suptitle('Average Beat after interpolation')
# fig.subplots_adjust(bottom=0, top=1)
# fig.tight_layout()

# # Show the plot
# plt.show()


#%%
#fig, axs = plt.subplots(8, 1, figsize=(10, 15), dpi=300, gridspec_kw={'height_ratios': [3, 3, 3, 3, 3, 3,3,3]})

# # plot mean values and shade the area within one standard deviation
# nm = [125,115,105,95,85,75,65,55]
# for i, df in enumerate([dfrp125, dfrp115, dfrp105, dfrp95, dfrp85, dfrp75, dfrp65, dfrp55]):
#     mean_vals = df.iloc[:,:-31].mean(axis=0)
#     std_vals = df.iloc[:,:-31].std(axis=0)
#     axs[i].plot(mean_vals, color='blue')
#     axs[i].fill_between(range(len(mean_vals)), mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color='blue')
#     axs[i].set_title(f'{nm[i]-5} to {nm[i]+5}')
#     axs[i].set_ylabel('Voltage (mV)')
#     #axs[i].text(0.02, 0.9, f'Total Beats: {df.shape[0]}', transform=axs[i].transAxes, fontsize=10, verticalalignment='top')
# axs[-1].set_xlabel('')
# # remove vertical spacing between subplots
# fig.subplots_adjust(hspace=0)
# fig.suptitle('Average Beat after Zero pad on right')
# # set bottom and top margins to zero
# fig.subplots_adjust(bottom=0, top=1)
# # adjust the spacing between subplots
# fig.tight_layout()
# #fig.text(0.5, 0.00001, 'ECG Sample ID', ha='center', fontsize=24)
# #fig.text(0.001, 0.5, 'ECG Sample Amplitude', va='center', rotation='vertical', fontsize=24)
# axs[-1].set_xlabel('Time (s)')
# # show the plot
# plt.show()

# #%%

# ### HYPONORMAL HYPONORMAL HYPONORMAL HYPONORMAL HYPONORMAL HYPONORMAL HYPONORMAL HYPONORMAL HYPONORMAL HYPONORMAL HYPONORMAL 
# fig, axs = plt.subplots(8, 1, figsize=(10, 15), dpi=300, gridspec_kw={'height_ratios': [3, 3, 3, 3, 3, 3, 3, 3]})

# # Define the two lists of dataframes
# df_hypo = [dfrp125_hypo, dfrp115_hypo, dfrp105_hypo, dfrp95_hypo, dfrp85_hypo, dfrp75_hypo, dfrp65_hypo, dfrp55_hypo]
# df_normal = [dfrp125_normal, dfrp115_normal, dfrp105_normal, dfrp95_normal, dfrp85_normal, dfrp75_normal, dfrp65_normal, dfrp55_normal]

# # plot mean values and shade the area within one standard deviation for each set of dataframes
# nm = [125, 115, 105, 95, 85, 75, 65, 55]
# for i, (df_h, df_n) in enumerate(zip(df_hypo, df_normal)):
#     mean_vals_h = df_h.iloc[:, :-17].mean(axis=0)
#     std_vals_h = df_h.iloc[:, :-17].std(axis=0)
#     mean_vals_n = df_n.iloc[:, :-17].mean(axis=0)
#     std_vals_n = df_n.iloc[:, :-17].std(axis=0)
    
#     # Plot the first set of dataframes
#     axs[i].plot(mean_vals_h, color='red')
#     axs[i].fill_between(range(len(mean_vals_h)), mean_vals_h - std_vals_h, mean_vals_h + std_vals_h, alpha=0.2, color='red')
    
#     # Plot the second set of dataframes
#     axs[i].plot(mean_vals_n, color='blue')
#     axs[i].fill_between(range(len(mean_vals_n)), mean_vals_n - std_vals_n, mean_vals_n + std_vals_n, alpha=0.2, color='blue')
    
#     axs[i].set_title(f'{nm[i]-5} to {nm[i]+5}')
#     # axs[i].text(0.02, 0.9, f'Total Beats (Hypo): {df_h.shape[0]}', transform=axs[i].transAxes, fontsize=10, verticalalignment='top')
#     # axs[i].text(0.02, 0.8, f'Total Beats (Normal): {df_n.shape[0]}', transform=axs[i].transAxes, fontsize=10, verticalalignment='top')

# axs[-1].set_xlabel('')
# fig.subplots_adjust(hspace=0)
# fig.suptitle('Average Beat after interpolation')
# fig.subplots_adjust(bottom=0, top=1)
# fig.tight_layout()
# fig.text(0.5, 0.00001, 'ECG Sample ID', ha='center', fontsize=24)
# fig.text(0.001, 0.5, 'ECG Sample Amplitude', va='center', rotation='vertical', fontsize=24)

# plt.show()

# %%
# dflrp55,dflrp55_hypo,dflrp55_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=55)
# dflrp65,dflrp65_hypo,dflrp65_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=65)
# dflrp75,dflrp75_hypo,dflrp75_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=75)
# dflrp85,dflrp85_hypo,dflrp85_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=85)
# dflrp95,dflrp95_hypo,dflrp95_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=95)
# dflrp105,dflrp105_hypo,dflrp105_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=105)
# dflrp115,dflrp115_hypo,dflrp115_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=115)
# dflrp125,dflrp125_hypo,dflrp125_normal = create_filtered_dfs(final_df_leftpad_all,HrRt=125)

# #%%
# # ###################################################################################
# fig, axs = plt.subplots(8, 1, figsize=(10, 15), dpi=300, gridspec_kw={'height_ratios': [3, 3, 3, 3, 3, 3,3,3]})

# # plot mean values and shade the area within one standard deviation
# nm = [125,115,105,95,85,75,65,55]
# for i, df in enumerate([dflrp125, dflrp115, dflrp105, dflrp95, dflrp85, dflrp75, dflrp65, dflrp55]):
#     mean_vals = df.iloc[:,:250].mean(axis=0)
#     std_vals = df.iloc[:,:250].std(axis=0)
#     axs[i].plot(mean_vals, color='blue')
#     axs[i].fill_between(range(len(mean_vals)), mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color='blue')
#     axs[i].set_title(f'{nm[i]-5} to {nm[i]+5}')
#     #axs[i].text(0.02, 0.9, f'Total Beats: {df.shape[0]}', transform=axs[i].transAxes, fontsize=10, verticalalignment='top')
#     axs[i].set_ylabel('Voltage (mV)')
# axs[-1].set_xlabel('')
# # remove vertical spacing between subplots
# fig.subplots_adjust(hspace=0)
# fig.suptitle('Average Beat after Zero pad on right')
# # set bottom and top margins to zero
# fig.subplots_adjust(bottom=0, top=1)
# # adjust the spacing between subplots
# fig.tight_layout()
# #fig.text(0.5, 0.00001, 'ECG Sample ID', ha='center', fontsize=24)
# #fig.text(0.001, 0.5, 'ECG Sample Amplitude', va='center', rotation='vertical', fontsize=24)
# axs[-1].set_xlabel('Time (s)')
# # show the plot
# plt.show()

# #%%

# #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# fig, axs = plt.subplots(8, 1, figsize=(10, 15), dpi=300, gridspec_kw={'height_ratios': [3, 3, 3, 3, 3, 3,3,3]})

# # plot mean values and shade the area within one standard deviation
# nm = [125,115,105,95,85,75,65,55]
# for i, df in enumerate([dflrp125, dflrp115, dflrp105, dflrp95, dflrp85, dflrp75, dflrp65, dflrp55]):
#     mean_vals = df.iloc[:,50:300].mean(axis=0)
#     std_vals = df.iloc[:,50:300].std(axis=0)
#     axs[i].plot(mean_vals, color='blue')
#     axs[i].fill_between(range(50, 300), mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color='blue')
#     axs[i].set_title(f'{nm[i]-5} to {nm[i]+5}')
#     axs[i].text(0.02, 0.9, f'Total Beats: {df.shape[0]}', transform=axs[i].transAxes, fontsize=10, verticalalignment='top')
# axs[-1].set_xlabel('')
# # remove vertical spacing between subplots
# fig.subplots_adjust(hspace=0)
# fig.suptitle('Average Beat after Zero pad on right')
# # set bottom and top margins to zero
# fig.subplots_adjust(bottom=0, top=1)
# # adjust the spacing between subplots
# fig.tight_layout()
# fig.text(0.5, 0.00001, 'ECG Sample ID', ha='center', fontsize=24)
# fig.text(0.001, 0.5, 'ECG Sample Amplitude', va='center', rotation='vertical', fontsize=24)
# axs[-1].set_xlabel('Time (s)')
# # show the plot
# plt.show()

# #%%
# #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# fig, axs = plt.subplots(8, 1, figsize=(10, 15), dpi=300, gridspec_kw={'height_ratios': [3, 3, 3, 3, 3, 3,3,3]})

# # plot mean values and shade the area within one standard deviation
# nm = [125,115,105,95,85,75,65,55]
# for i, df in enumerate([dflrp125, dflrp115, dflrp105, dflrp95, dflrp85, dflrp75, dflrp65, dflrp55]):
#     mean_vals = df.iloc[:,50:300].mean(axis=0)
#     std_vals = df.iloc[:,50:300].std(axis=0)
#     # convert sample IDs to time in seconds
#     sample_frequency = 250
#     time_in_seconds = (np.arange(len(mean_vals)) - 50) / sample_frequency
#     axs[i].plot(time_in_seconds, mean_vals, color='blue')
#     axs[i].fill_between(time_in_seconds, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color='blue')
#     axs[i].set_title(f'{nm[i]-5} to {nm[i]+5}')
#     axs[i].set_ylabel('Voltage (mV)')
#     #axs[i].text(0.02, 0.9, f'Total Beats: {df.shape[0]}', transform=axs[i].transAxes, fontsize=10, verticalalignment='top')
# axs[-1].set_xlabel('')
# # remove vertical spacing between subplots
# fig.subplots_adjust(hspace=0)
# fig.suptitle('Average Beat after Zero pad on right')
# # set bottom and top margins to zero
# fig.subplots_adjust(bottom=0, top=1)
# # adjust the spacing between subplots
# fig.tight_layout()
# # set x-axis ticks and labels
# xticks = np.arange(-0.2, 0.8, 0.2)
# xticklabels = [f'{t:.1f}' for t in xticks]
# for ax in axs:
#     ax.set_xticks(xticks)
#     ax.set_xticklabels(xticklabels, fontsize=16)
# #fig.text(0.5, 0.00001, 'Time (s)', ha='center', fontsize=24)
# #fig.text(0.001, 0.5, 'ECG Sample Amplitude', va='center', rotation='vertical', fontsize=24)
# axs[-1].set_xlabel('Time (s)')
# # show the plot
# plt.show()

# #%%
# fig, axs = plt.subplots(8, 1, figsize=(10, 15), dpi=300, gridspec_kw={'height_ratios': [3, 3, 3, 3, 3, 3, 3, 3]})

# # Define the two lists of dataframes
# df_hypo = [dflrp125_hypo, dflrp115_hypo, dflrp105_hypo, dflrp95_hypo, dflrp85_hypo, dflrp75_hypo, dflrp65_hypo, dflrp55_hypo]
# df_normal = [dflrp125_normal, dflrp115_normal, dflrp105_normal, dflrp95_normal, dflrp85_normal, dflrp75_normal, dflrp65_normal, dflrp55_normal]

# # plot mean values and shade the area within one standard deviation for each set of dataframes
# nm = [125, 115, 105, 95, 85, 75, 65, 55]
# for i, (df_h, df_n) in enumerate(zip(df_hypo, df_normal)):
#     mean_vals_h = df_h.iloc[:, 50:300].mean(axis=0)
#     std_vals_h = df_h.iloc[:, 50:300].std(axis=0)
#     mean_vals_n = df_n.iloc[:, 50:300].mean(axis=0)
#     std_vals_n = df_n.iloc[:, 50:300].std(axis=0)
#     # convert sample IDs to time in seconds
#     sample_frequency = 250
#     time_in_seconds = (np.arange(len(mean_vals_h)) - 50) / sample_frequency
    
#     # Plot the first set of dataframes
#     axs[i].plot(time_in_seconds, mean_vals_h, color='red', label='Hypo')
#     axs[i].fill_between(time_in_seconds, mean_vals_h - std_vals_h, mean_vals_h + std_vals_h, alpha=0.2, color='red')
    
#     # Plot the second set of dataframes
#     axs[i].plot(time_in_seconds, mean_vals_n, color='blue', label='Normal')
#     axs[i].fill_between(time_in_seconds, mean_vals_n - std_vals_n, mean_vals_n + std_vals_n, alpha=0.2, color='blue')
#     axs[i].set_ylabel('Voltage (mV)')
#     axs[i].set_title(f'{nm[i]-5} to {nm[i]+5}')
#     axs[i].text(0.02, 0.9, f'Total Beats (Hypo): {df_h.shape[0]/(df_h.shape[0]+df_n.shape[0]):.2f}', transform=axs[i].transAxes, fontsize=10, verticalalignment='top')
#     #axs[i].text(0.02, 0.8, f'Total Beats (Normal): {df_n.shape[0]}', transform=axs[i].transAxes, fontsize=10, verticalalignment='top')
#     axs[i].legend()
# axs[-1].set_xlabel('')
# fig.subplots_adjust(hspace=0)
# fig.suptitle('Average Beat after Zero pad on right')
# fig.subplots_adjust(bottom=0, top=1)
# fig.tight_layout()
# axs[-1].set_xlabel('Time (s)')
# xticks = np.arange(-0.2, 0.8, 0.2)
# xticklabels = [f'{t:.1f}' for t in xticks]
# for ax in axs:
#     ax.set_xticks(xticks)
#     ax.set_xticklabels(xticklabels, fontsize=16)

# #fig.text(0.5, 0.00001, 'Time (s)', ha='center', fontsize=24)
# #fig.text(0.001, 0.5, 'ECG Sample Amplitude', va='center', rotation='vertical', fontsize=24)
# plt.show()

# %%
