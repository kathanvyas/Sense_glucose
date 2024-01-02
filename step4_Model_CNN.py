
## adopted from /home/grads/k/kathan/Warwick/QT_correction/modelling/CNN_eval_pad_v2.py
## Uses the PCA filtered and padded beats as input, creates 4 bootsrap folds and performs CNN model presented in the code below.
## saves the second last layer of CNN and names it flatten output 

#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import math
import random
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, Dense, Input, MaxPooling1D, LSTM, Dropout,Lambda, BatchNormalization, LayerNormalization,concatenate, Flatten
from tensorflow.keras.layers import TimeDistributed,Activation
#from tensorflow.keras.layers.layer import TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import MaxPooling1D, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.metrics import roc_curve,auc,classification_report,accuracy_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import time
from tensorflow import random
from numpy.random import seed
from sklearn.model_selection import KFold
import gc
import sys

sd = 42
seed(sd)
random.set_seed(sd)
gpu_num = 3  #2,3
c=2
s=5
t_type='hypo' #hyper  #hypo
st = time.time()

wrp = '/mnt/nvme-data1/Kathan/QT_correction/pca_int_pad_data/'
#wrp = '/mnt/nvme-data1/Kathan/QT_correction/pca_int_pad_data_hyper/'
# int_df = pd.read_pickle(wrp + f'c{c}s{s:02d}_df_interpolate.pkl')
pad_df = pd.read_pickle(wrp + f'c{c}s{s:02d}_df_pad.pkl')    
bsize = 512



#%%
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
config = tf.compat.v1.ConfigProto(device_count = {'GPU':gpu_num}) #max no of GPUs = 1, CPUs =4
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6


def og_build_model(n_rows, n_cols, rr_feats):
    ecg_input = Input(shape=(n_cols, n_rows), name='ecg_signal')
    print('model_input shape:', ecg_input.shape)
    rr_input = Input(shape=(rr_feats), name='BlockB')
    print("BlockB CNN shape:", rr_input.shape)
    nkernels = 32
    ksize = 5
    c = ecg_input
    for i in range(1, 11):
        c = Conv1D(nkernels, ksize, name=f'conv_{i}', padding='same', kernel_initializer="glorot_uniform",
                   bias_initializer=initializers.Zeros(), kernel_regularizer=l2(0.001))(c)
        b = BatchNormalization(name=f'BN_{i}')(c)
        a = Activation('relu')(b)
        if i % 2 == 0:
            p = MaxPooling1D(pool_size=2)(a)
            c = p
        else:
            c = a
    # After the convolutional layers
    fl = Flatten(name='fl')(c)
    
    den = Dense(30, activation='relu', kernel_regularizer=l2(0.001))(fl)
    den = Dropout(0.5)(den)  # Add dropout for regularization
    output = Dense(1, activation='sigmoid')(den)
    opt = Adam(learning_rate=1e-4)
    model = Model(inputs=ecg_input, outputs=output, name='model')
    extractor = Model(inputs=ecg_input, outputs=model.get_layer('fl').output)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model, extractor


def split_and_create_sets(df):
        unique_flags = df['flag'].unique()
        sorted_flags = sorted(unique_flags, key=lambda x: df[df['flag'] == x]['Time'].min())

        train_test_sets = []
        kf = KFold(n_splits=5)
        for train_indices, test_indices in kf.split(sorted_flags):
            train_flags = [sorted_flags[i] for i in train_indices]
            test_flags = [sorted_flags[i] for i in test_indices]
            train_data = df[df['flag'].isin(train_flags)]
            test_data = df[df['flag'].isin(test_flags)]
            train_test_sets.append((train_data, test_data))
        return train_test_sets

    
# Function to split a dataframe and create train-test sets
def create_train_test_sets(dataframe):
    np.random.seed(42)  # Set a seed for reproducibility

    # Split the dataframe into two based on hypo_label
    df_hypo0 = dataframe[dataframe['hypo_label'] == 0]
    df_hypo1 = dataframe[dataframe['hypo_label'] == 1]

    # Create train-test sets for both dataframes
    train_test_sets_hypo0 = split_and_create_sets(df_hypo0)
    train_test_sets_hypo1 = split_and_create_sets(df_hypo1)

    # Combine the corresponding train and test sets
    final_train_test_sets = []
    for sets_hypo0, sets_hypo1 in zip(train_test_sets_hypo0, train_test_sets_hypo1):
        train_data_combined = pd.concat([sets_hypo0[0], sets_hypo1[0]])
        test_data_combined = pd.concat([sets_hypo0[1], sets_hypo1[1]])

        final_train_test_sets.append((train_data_combined, test_data_combined))

    return final_train_test_sets

#%%

int_train_test_data = create_train_test_sets(pad_df)
# %%
stdoutOrigin=sys.stdout 

for fld in range(5):
    trd, tsd = int_train_test_data[fld]
    trd.to_pickle(f'/mnt/nvme-data1/Kathan/QT_correction/trains_hyper/c{c}s{s:02d}_train_{fld}.pkl')
    tsd.to_pickle(f'/mnt/nvme-data1/Kathan/QT_correction/tests_hyper/c{c}s{s:02d}_test_{fld}.pkl')
    train_data_labels = trd[t_type + '_label'].values
    xtrain,xvalid,ytrain,yvalid = train_test_split(trd, train_data_labels, test_size=0.3, random_state=42, shuffle=True, stratify=train_data_labels)

    train_data = xtrain.iloc[:,:200].values
    train_data=train_data[...,None]
    train_rr = xtrain.iloc[:]['rr'].values
    train_rr = train_rr.reshape(-1, 1)

    val_data = xvalid.iloc[:,:200].values
    val_data=val_data[...,None]
    val_rr = xvalid.iloc[:]['rr'].values
    val_rr = val_rr.reshape(-1, 1)

    test_data = tsd.iloc[:,:200].values
    test_data=test_data[...,None]
    test_rr = tsd.iloc[:]['rr'].values
    test_rr = test_rr.reshape(-1, 1)
    test_labels = tsd[t_type + '_label'].values

    data = pd.concat([trd, tsd]).iloc[:,:200].values
    data=data[...,None]

    n_cols = data.shape[2]
    n_rows = data.shape[1]
    rr_feats = data.shape[2]

    del data
    gc.collect()

    sys.stdout = open(f"/mnt/nvme-data1/Kathan/QT_correction/logs/cnn_hyper/c{c}s{s:02d}_CNNPAD_log{fld+1}.txt", "w")

    batch_size,verbose , epochs = bsize, 1, 45

    early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 5, mode = 'auto', restore_best_weights = True,start_from_epoch=0)

    model,ex = og_build_model(n_cols,n_rows,rr_feats)

    history = model.fit(x=[train_data, train_rr], y=ytrain, epochs=epochs, verbose=verbose,
                        batch_size=batch_size, validation_data=([val_data, val_rr], yvalid),
                        shuffle=False, callbacks=[early_stopping_callback])

    y_pred = model.predict([test_data, test_rr])
    fpr, tpr, thresholds =roc_curve(test_labels, y_pred)
    roc_auc = auc(fpr,tpr)
    print("AUC: ",roc_auc)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    pred_label = np.where(y_pred > thresh,1,0)
    tsacc = accuracy_score(test_labels, pred_label)

    print(classification_report(test_labels,pred_label))

    # model.save(path + ver + subject + '/' + t_type + mode + '_'+ run + '_cnn_model')
    # ex.save(path + ver + subject + '/' + t_type + mode + '_'+ run + '_cnn_extractor')
    

    print('original train data:',trd.shape)
    idd = 1
    trdfs={}
    for i in range(0,trd.shape[0],20000):
        #print(i)
        data = trd[i:i+20000]
        ttrain = data.iloc[:,:200].values
        ttrain = ttrain[...,None]
        tr_rr = data.iloc[:]['rr'].values
        tr_rr = train_rr.reshape(-1, 1)
        tr_feat = ex([ttrain,tr_rr])
        feats = np.array(tr_feat)
        trdfs[idd] = pd.DataFrame(feats)
        # d = d.reset_index(drop=True)
        # trdfs[idd] = pd.concat([d,data.iloc[:,200:]],axis = 1)
        idd = idd+1
        i=i+20000 
    ddf = pd.concat(trdfs, ignore_index=True)
    print('Flattened Train output:', ddf.shape)
    ddf.to_csv(f'/mnt/nvme-data1/Kathan/QT_correction/embeds_hyper/c{c}s{s:02d}__train_flatten30_{fld+1}_.csv')
    ddf.to_pickle(f'/mnt/nvme-data1/Kathan/QT_correction/embeds_hyper/c{c}s{s:02d}__train_flatten30_{fld+1}_.pkl')

    test_main = tsd
    print('original train data:',test_main.shape)
    tidd = 1
    tsdfs={}
    for i in range(0,test_main.shape[0],20000):
        #print(i)
        tdata = test_main[i:i+20000]
        testt = tdata.iloc[:,:200].values
        testt = testt[...,None]
        ts_rr = tdata.iloc[:]['rr'].values
        ts_rr = ts_rr.reshape(-1, 1)
        ts_feat = ex([testt,ts_rr])
        tfeats = np.array(ts_feat)
        tsdfs[tidd] = pd.DataFrame(tfeats)
        # td = td.reset_index(drop=True)
        # tsdfs[tidd] = pd.concat([td,tdata.iloc[:,200:].reset_index(drop=True)],axis = 1)
        tidd = tidd+1
        i=i+20000 
    tddf = pd.concat(tsdfs, ignore_index=True)
    print('Flattened Train output:', tddf.shape)
    tddf.to_csv(f'/mnt/nvme-data1/Kathan/QT_correction/embeds_hyper/c{c}s{s:02d}__test_flatten30_{fld+1}_.csv')
    tddf.to_pickle(f'/mnt/nvme-data1/Kathan/QT_correction/embeds_hyper/c{c}s{s:02d}__test_flatten30_{fld+1}_.pkl')

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    sys.stdout.close()

sys.stdout=stdoutOrigin
