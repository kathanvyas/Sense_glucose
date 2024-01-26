import time
import random
import argparse
import pandas as pd
import numpy as np
from libs.helper import plot_beat, plot_alignment_beats

def demo_main(ecg_path):

    print("Reading data from {}".format(ecg_path))
    start_t = time.time()
    df = pd.read_pickle(ecg_path)
    print("--- {:2f} seconds ---".format(time.time() - start_t))
    
    # get number of rows where hypo_label is 1
    print("Number of hypo: {}".format(len(df[df['hypo_label'] == 1])))
    # get number of rows where hypo_label is 0
    print("Number of normal: {}".format(len(df[df['hypo_label'] == 0])))
    
    # get the index of the random row
    idx = random.randint(0, len(df))
    print(df.iloc[idx][:200])
    plot_beat(df, idx, "./demo/ecg.png")


def demo_aligned(ecg_aligned_path):
    print("Reading data from {}".format(ecg_aligned_path))
    start_t = time.time()
    df = pd.read_pickle(ecg_aligned_path)
    print("--- {:2f} seconds ---".format(time.time() - start_t))

    # get number of rows where hypo_label is 1
    print("Number of hypo: {}".format(len(df[df['hypo_label'] == 1])))
    # get number of rows where hypo_label is 0
    print("Number of normal: {}".format(len(df[df['hypo_label'] == 0])))
    
    # random select 5 rows to show
    indices = random.sample(np.arange(len(df)).tolist(), 5)
    plot_alignment_beats(df, indices, "./demo/ecg_aligned.png")
   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ecg', default=None, type=str, help='path to the processed ecg data')
    parser.add_argument('--ecg_aligned', default=None, type=str, help='path to the processed ecg data')
    args = parser.parse_args()


    if args.ecg is not None:
        demo_main(args.ecg)

    if args.ecg_aligned is not None:
        demo_aligned(args.ecg_aligned)