import time
import random
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from libs.helper import plot_beat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ecg', type=str, help='path to the processed ecg data')
    args = parser.parse_args()

    print("Reading data from {}".format(args.ecg))
    start_t = time.time()
    df = pd.read_pickle(args.ecg)
    print("--- {:2f} seconds ---".format(time.time() - start_t))
    
    # get number of rows where hypo_label is 1
    print("Number of hypo: {}".format(len(df[df['hypo_label'] == 1])))
    # get number of rows where hypo_label is 0
    print("Number of normal: {}".format(len(df[df['hypo_label'] == 0])))
    
    # get the index of the random row
    idx = random.randint(0, len(df))
    print(df.iloc[idx][:200])
    plot_beat(df, idx, "./demo/ecg.png")

   

