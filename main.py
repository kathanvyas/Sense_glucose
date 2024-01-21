import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.interpolate import interp1d
from skimage.util.shape import view_as_windows as viewW

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, help='path to folder containing ECG, summary, and glucose files')
    args = parser.parse_args()

    # check if ECG, summary, and glucose files exist
    ecg_path = os.path.join(args.input_folder, "ECG.pkl")
    summary_path = os.path.join(args.input_folder, "summary.pkl")
    glucose_path = os.path.join(args.input_folder, "glucose.pkl")
    if not os.path.exists(ecg_path) or not os.path.exists(summary_path) or not os.path.exists(glucose_path):
        raise ValueError("ECG, summary, or glucose files do not exist, please run libs/preprocess.py first")
    
    # use pickle to load df (faster than csv)
    print("Loading all ECG ~")
    start_time = time.time()   
    ecg_df = pd.read_pickle(ecg_path)
    print("size of df: ", ecg_df.shape)
    print("--- {:.3f} seconds ---".format(time.time() - start_time))

    print("Loading all Summary ~")
    start_time = time.time()
    summary_df = pd.read_pickle(summary_path)
    print("size of df: ", summary_df.shape)
    print("--- {:.3f} seconds ---".format(time.time() - start_time))

    print("Loading glucose file ~")
    start_time = time.time()
    glucose_df = pd.read_pickle(glucose_path)
    print("size of df: ", glucose_df.shape)
    print("--- {:.3f} seconds ---".format(time.time() - start_time))