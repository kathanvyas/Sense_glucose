
## adopted from /home/grads/k/kathan/Warwick/QT_correction/modelling/PC_MAHAL_datacreate.py
## uses the wholeDF beats obtained from step1 and uses HRgroups to filter noisy beats using PCA and mahlanobis distance

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows as viewW
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#df = df.fillna(0)
def remove_na_by_columns(df, columns_to_consider):
    return df.dropna(subset=columns_to_consider)
columns_to_consider = ["time", "glucose",'flag']

def create_filtered_dfs(df,HrRt,ttype):
    HR1 = HrRt-5
    HR2 = HrRt+5
    print(f'extremes are {HR1} and {HR2}')
    df__ = df[(df['HR'] <= HR2) & (df['HR'] >= HR1)]
    print(df__.shape)
    df__hypo = df__[df__[f'{ttype}_label'] == 1]
    df__normal = df__[df__[f'{ttype}_label'] == 0]
    return df__,df__hypo,df__normal

def strided_indexing_roll(a, r):
    # Concatenate with sliced to cover all rolls
    a_ext = np.concatenate((a,a[:,:-1]),axis=1)
    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = a.shape[1]
    return viewW(a_ext,(1,n))[np.arange(len(r)), (n-r)%n,0]

def calculate_distance_based_on_PCA(data,title,silent_plot=1):
    x = data.iloc[:, :]
    ecg = x.iloc[:, :]

    # Perform PCA
    pca = PCA()
    pc = pca.fit_transform(ecg)
    v = pca.components_.T
    e = pca.explained_variance_

    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(explained_variance >= 0.90) + 1
    print(f'Number of PCs that cover ~100% variance: {num_components}')

    coff = num_components
    v = v[:, :coff]
    pc = pc[:, :coff]
    e = e[:coff]
    if silent_plot==0:
        # Plot the first and second principal components
        plt.figure()
        plt.plot(pc[:, 0], pc[:, 1], '.')
        plt.xlabel('Principal component 1')
        plt.ylabel('Principal component 2')
        plt.title('Non-normalised PC1 vs PC2')
    #mu = np.mean(ecg, axis=0)
    # normalise teh principal components
    sqrt_e = np.sqrt(e).reshape(1, -1)
    replicated_sqrt_e = np.repeat(sqrt_e, pc.shape[0], axis=0)
    pcn = pc / replicated_sqrt_e
    d = np.sqrt(np.sum(pcn ** 2, axis=1) / coff)
    ecg['d'] = d

    ecgx = ecg.sort_values('d')
    if silent_plot==0:
    # Plot the normalized principal components
        plt.figure()
        plt.plot(pcn[:, 0], pcn[:, 1], '.')
        plt.xlabel('Principal component 1')
        plt.ylabel('Principal component 2')
        plt.title('Normalised PC1 vs PC2')

    # d1, ix = np.sort(d), np.argsort(d)

    # Plot the sorted distances
    plt.figure()
    plt.plot(np.sort(d))
    plt.title(f'Sorted distances for {title}')
    plt.xlabel('Beat Index')
    plt.ylabel('Distance')

    ecg1 = ecgx.iloc[:,:-1]
    ecg1 = ecg1.reset_index(drop=True)

    return ecg,ecg1

def create_final_df(e,dfs):
    e   = e.reset_index(drop=True)
    dfs = dfs.reset_index(drop=True)
    final = pd.concat([e,dfs],axis=1)
    return final

def perform_regression(fdf,test='RT',scale=0):
    selected_columns = ['rr', 'RT', 'QT', 'glucose']
    df_selected = fdf[selected_columns]
    if scale==0:
        df_selected['rr'] = df_selected.iloc[:]['rr']*1000
        df_selected['RT'] = df_selected.iloc[:]['RT']*1000
        df_selected['QT'] = df_selected.iloc[:]['QT']*1000

    # Splitting into independent variables (X) and dependent variable (y)
    X = df_selected[['rr', 'glucose']]
    if test=='RT':
        y = df_selected['RT']
    else:
        y = df_selected['QT']

    # Running linear regression
    lm1 = LinearRegression()
    lm1.fit(X, y)

    # Printing the results
    print('Intercept:\n', lm1.intercept_)
    print('Coefficients:\n', lm1.coef_)

def perform_regression1(fdf, test='RT', scale=0):
    selected_columns = ['rr', 'RT', 'QT', 'glucose']
    df_selected = fdf[selected_columns].copy()  # Create a copy of the selected columns

    if scale == 0:
        df_selected[['rr', 'RT', 'QT']] *= 1000  # Scale the columns in-place

    # Splitting into independent variables (X) and dependent variable (y)
    X = df_selected[['rr', 'glucose']]
    y = df_selected[test] if test == 'RT' else df_selected['QT']

    # Running linear regression
    lm1 = LinearRegression()
    lm1.fit(X, y)

    # Printing the results
    print('Intercept:\n', lm1.intercept_)
    print('Coefficients:\n', lm1.coef_)


#%%
# Load data from file
c=2
s=5
#path = '/mnt/nvme-data1/Kathan/ECGContext/all peaks/'  #For hypo
path = '/mnt/nvme-data1/Kathan/QT_correction/all peaks_hyper/'     #For hyper
df = pd.read_pickle(path + f"c{c}s{s:02d}/c{c}s{s:02d}_wholeDF.pkl")
df = remove_na_by_columns(df, columns_to_consider)
df['QT_samples'] = df['t'] - df['q']
df['QT'] = df['QT_samples']/250
df['RT_samples'] = df['t'] - df['r']
df['RT'] = df['RT_samples']/250

#perform_regression1(df,test='RT')


#%%
df25,df25_hypo,df25_normal = create_filtered_dfs(df,HrRt=25,ttype='hyper')
df35,df35_hypo,df35_normal = create_filtered_dfs(df,HrRt=35,ttype='hyper')
df45,df45_hypo,df45_normal = create_filtered_dfs(df,HrRt=45,ttype='hyper')
df55,df55_hypo,df55_normal = create_filtered_dfs(df,HrRt=55,ttype='hyper')
df65,df65_hypo,df65_normal = create_filtered_dfs(df,HrRt=65,ttype='hyper')
df75,df75_hypo,df75_normal = create_filtered_dfs(df,HrRt=75,ttype='hyper')
df85,df85_hypo,df85_normal = create_filtered_dfs(df,HrRt=85,ttype='hyper')
df95,df95_hypo,df95_normal = create_filtered_dfs(df,HrRt=95,ttype='hyper')
df105,df105_hypo,df105_normal = create_filtered_dfs(df,HrRt=105,ttype='hyper')
df115,df115_hypo,df115_normal = create_filtered_dfs(df,HrRt=115,ttype='hyper')
df125,df125_hypo,df125_normal = create_filtered_dfs(df,HrRt=125,ttype='hyper')
df135,df135_hypo,df135_normal = create_filtered_dfs(df,HrRt=135,ttype='hyper')
df145,df145_hypo,df145_normal = create_filtered_dfs(df,HrRt=145,ttype='hyper')
df155,df155_hypo,df155_normal = create_filtered_dfs(df,HrRt=155,ttype='hyper')
df165,df165_hypo,df165_normal = create_filtered_dfs(df,HrRt=165,ttype='hyper')


dfs_list = [df25,df35,df45,df55,df65,df75,df85,df95,df105,df115,df125,df135,df145,df155,df165]
dfs = {}
dataF = 25
for i in dfs_list:
    cdf = i.iloc[:,:-31]
    max_shift = 100
    shifts = np.array(i['r'].values)
    r = max_shift-shifts
    if not cdf.empty:
        out = pd.DataFrame(strided_indexing_roll(np.pad(cdf.to_numpy(),((0, 0), (0, r.max()))),r),index=i.index)
        dfs[dataF] = pd.concat([out.iloc[:,30:230],i.iloc[:,-31:]],axis=1)
    else:
        dfs[dataF] = i
    dataF = dataF+10

#%%
# e25,e1_25 = calculate_distance_based_on_PCA(dfs[25].iloc[:,:200].fillna(0),'25')
# f25 = create_final_df(e25,dfs[25].iloc[:,-31:].fillna(0))
# e35,e1_35 = calculate_distance_based_on_PCA(dfs[35].iloc[:,:200].fillna(0),'35')
# f35 = create_final_df(e35,dfs[35].iloc[:,-31:].fillna(0))
e45,e1_45 = calculate_distance_based_on_PCA(dfs[45].iloc[:,:200].fillna(0),'45')
f45 = create_final_df(e45,dfs[45].iloc[:,-31:].fillna(0))
e55,e1_55 = calculate_distance_based_on_PCA(dfs[55].iloc[:,:200].fillna(0),'55')
f55 = create_final_df(e55,dfs[55].iloc[:,-31:].fillna(0))
e65,e1_65 = calculate_distance_based_on_PCA(dfs[65].iloc[:,:200].fillna(0),'65')
f65 = create_final_df(e65,dfs[65].iloc[:,-31:].fillna(0))
e75,e1_75 = calculate_distance_based_on_PCA(dfs[75].iloc[:,:200].fillna(0),'75')
f75 = create_final_df(e75,dfs[75].iloc[:,-31:].fillna(0))
e85,e1_85 = calculate_distance_based_on_PCA(dfs[85].iloc[:,:200].fillna(0),'85')
f85 = create_final_df(e85,dfs[85].iloc[:,-31:].fillna(0))
e95,e1_95 = calculate_distance_based_on_PCA(dfs[95].iloc[:,:200].fillna(0),'95')
f95 = create_final_df(e95,dfs[95].iloc[:,-31:].fillna(0))
e105,e1_105 = calculate_distance_based_on_PCA(dfs[105].iloc[:,:200].fillna(0),'105')
f105 = create_final_df(e105,dfs[105].iloc[:,-31:].fillna(0))
e115,e1_115 = calculate_distance_based_on_PCA(dfs[115].iloc[:,:200].fillna(0),'115')
f115 = create_final_df(e115,dfs[115].iloc[:,-31:].fillna(0))
e125,e1_125 = calculate_distance_based_on_PCA(dfs[125].iloc[:,:200].fillna(0),'125')
f125 = create_final_df(e125,dfs[125].iloc[:,-31:].fillna(0))
e135,e1_135 = calculate_distance_based_on_PCA(dfs[135].iloc[:,:200].fillna(0),'135')
f135 = create_final_df(e135,dfs[135].iloc[:,-31:].fillna(0))
e145,e1_145 = calculate_distance_based_on_PCA(dfs[145].iloc[:,:200].fillna(0),'145')
f145 = create_final_df(e145,dfs[145].iloc[:,-31:].fillna(0))
e155,e1_155 = calculate_distance_based_on_PCA(dfs[155].iloc[:,:200].fillna(0),'155')
f155 = create_final_df(e155,dfs[155].iloc[:,-31:].fillna(0))
e165,e1_165 = calculate_distance_based_on_PCA(dfs[165].iloc[:,:200].fillna(0),'165')
f165 = create_final_df(e165,dfs[165].iloc[:,-31:].fillna(0))


#%%
final_dfs = [f45,f55,f65,f75,f85,f95,f105,f115,f125,f135,f145,f155,f165]
f = pd.concat(final_dfs)

# f25_filtered = f25[f25['d'] < 2.5].copy()
# f35_filtered = f35[f35['d'] < 2.5].copy()
f45_filtered = f45[f45['d'] < 2.5].copy()
f55_filtered = f55[f55['d'] < 2.5].copy()
f65_filtered = f65[f65['d'] < 2.5].copy()
f75_filtered = f75[f75['d'] < 2.5].copy()
f85_filtered = f85[f85['d'] < 2.5].copy()
f95_filtered = f95[f95['d'] < 2.5].copy()
f105_filtered = f105[f105['d'] < 2.5].copy()
f115_filtered = f115[f115['d'] < 2.5].copy()
f125_filtered = f125[f125['d'] < 2.5].copy()
f135_filtered = f135[f135['d'] < 2.5].copy()
f145_filtered = f145[f145['d'] < 2.5].copy()
f155_filtered = f155[f155['d'] < 2.5].copy()
f165_filtered = f165[f165['d'] < 2.5].copy()


#f25_filtered,f35_filtered,
final_filtered_dfs = [f45_filtered,f55_filtered,f65_filtered,f75_filtered,f85_filtered,
                      f95_filtered,f105_filtered,f115_filtered,f125_filtered,f135_filtered,f145_filtered,
                      f155_filtered,f165_filtered]
f_filtered = pd.concat(final_filtered_dfs)


#%%
fig, axs = plt.subplots(12, 1, figsize=(10, 15), dpi=300, gridspec_kw={'height_ratios': [3, 3, 3, 3, 3, 3,3,3,3,3,3,3]})

# plot mean values and shade the area within one standard deviation
nm = [45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155]
for i, df in enumerate([f45_filtered, f55_filtered, f65_filtered,f75_filtered, f85_filtered, 
                        f95_filtered, f105_filtered, f115_filtered,f125_filtered, f135_filtered, 
                        f145_filtered, f155_filtered]):
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
f.to_pickle(f'/mnt/nvme-data1/Kathan/QT_correction/PCA data_hyper/c{c}s{s:02d}_f_df.pkl')
f_filtered.to_pickle(f'/mnt/nvme-data1/Kathan/QT_correction/PCA data_hyper/c{c}s{s:02d}_filtered_df.pkl')


#%%

# perform_regression1(f,test='RT')
# perform_regression1(f_filtered,test='RT')



# #%%
# import statsmodels.formula.api as smf
# d1 = f
# d1[['rr', 'RT', 'QT']] *= 1000
# model = smf.mixedlm("RT ~ rr + glucose", d1, groups=d1["hypo_label"])
# result = model.fit()
# print(result.summary())

# d2 = f_filtered
# d2[['rr', 'RT', 'QT']] *= 1000
# model1 = smf.mixedlm("RT ~ rr + glucose", d2, groups=d2["hypo_label"])
# result1 = model1.fit()
# print(result1.summary())



# #%%

# f.to_pickle(f'/mnt/nvme-data1/Kathan/QT_correction/PCA data/c{c}s{s:02d}_f_df.pkl')
# f_filtered.to_pickle(f'/mnt/nvme-data1/Kathan/QT_correction/PCA data/c{c}s{s:02d}_filtered_df.pkl')

# #%%

# f1 = f[['rr','RT','QT','glucose','hypo_label']]
# f_filtered1 = f_filtered[['rr','RT','QT','glucose','hypo_label']]

# f1.to_csv(f'/home/grads/k/kathan/Warwick/QT_correction/filtered_dfs/c{c}s{s:02d}_f_df.csv',index=False)
# f_filtered1.to_csv(f'/home/grads/k/kathan/Warwick/QT_correction/filtered_dfs/c{c}s{s:02d}_filtered_df.csv',index=False)


# #%%
# ###### Linear Regression
# perform_regression1(f95,test='RT')
# perform_regression1(f95_filtered,test='RT')

# #%%

# #%%
# #f_filtered = f_filtered[f_filtered['RT'] <=0.5]
# red_data = f_filtered[f_filtered['hypo_label'] == 1]['RT']
# blue_data = f_filtered[f_filtered['hypo_label'] == 0]['RT']

# # Create a histogram with color coding
# plt.hist(red_data, color='red', alpha=0.4, label='Hypo', bins=35, edgecolor='black')
# plt.hist(blue_data, color='blue', alpha=0.4, label='Normal', bins=35, edgecolor='black')
# # plt.axvline(mean_0,c='blue')
# # plt.axvline(mean_1,c='red')
# # Add labels and title
# plt.xlabel('RT')
# plt.ylabel('Frequency')
# plt.title('Distribution of RT')
# plt.legend()

# # Show the histogram
# plt.show()

# #%%
# #f = f[f['RT'] <=0.5]
# red_data = f[f['hypo_label'] == 1]['RT']
# blue_data = f[f['hypo_label'] == 0]['RT']

# # Create a histogram with color coding
# plt.hist(red_data, color='red', alpha=0.4, label='Hypo', bins=35, edgecolor='black')
# plt.hist(blue_data, color='blue', alpha=0.4, label='Normal', bins=35, edgecolor='black')
# # plt.axvline(mean_0,c='blue')
# # plt.axvline(mean_1,c='red')
# # Add labels and title
# plt.xlabel('RT')
# plt.ylabel('Frequency')
# plt.title('Distribution of RT')
# plt.legend()

# # Show the histogram
# plt.show()

# #%%
# mean_vals = f_filtered.iloc[:,:200].mean(axis=0)
# std_vals = f_filtered.iloc[:,:200].std(axis=0)

# x_values = np.arange(len(mean_vals))  # Generate x-axis values
# # Plotting
# #plt.title(f'Average ECG beat and STD for {hr-5}<=HR<{hr+5} (10,000 samples)')
# plt.ylabel('ECG amplitude (mV)')
# plt.xlabel('Time (s)')
# plt.plot(x_values, mean_vals, color='blue')
# plt.fill_between(x_values, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color='blue')
# plt.show()



# #%%
# perform_regression(df,test='RT')
# #%%
# # %%
# from IPython import display
# import time
# for i, row in e1_105.iterrows():
#     if i%50==0:
#         # Create a new figure and axis
#         fig, ax = plt.subplots()

#         # Plot the row data
#         ax.plot(row)

#         # Set appropriate labels and title
#         ax.set_xlabel('X-axis')
#         ax.set_ylabel('Y-axis')
#         ax.set_title(f'Row {i+1}')

#         # Display the plot
#         display.display(fig)
#         display.clear_output(wait=True)

#         # Pause for 0.2 seconds
#         time.sleep(0.000001)
#         plt.close(fig)
#     else:
#         continue
# # %%
# fig, ax = plt.subplots()

# # Plot the row data
# ax.plot(e1_95.iloc[5,:])

# # Set appropriate labels and title
# ax.set_xlabel('ECG samples')
# ax.set_ylabel('ECG-Amplitude')


# # %%
# plt.figure()
# plt.plot(np.sort(e105['d']))
# plt.title(f'Sorted distances')
# plt.xlabel('Beat Index')
# plt.ylabel('Distance')
# # %%
# e105['log_d'] = np.log2(e105['d'])
# # %%
# plt.figure()
# plt.plot(np.sort(e105['log_d']))
# plt.title(f'Sorted distances')
# plt.xlabel('Beat Index')
# plt.ylabel('Distance')
# # %%
