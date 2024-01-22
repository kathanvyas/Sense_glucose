import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
from libs.helper import load_checkfile, generate_beats, plot_beat, combine_with_summary_glucose, clean_negative_values
from scipy.interpolate import interp1d
from skimage.util.shape import view_as_windows as viewW

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, help='path to folder containing ECG, summary, and glucose files')
    parser.add_argument('--out_folder', default="/mnt/data2/mtseng/dataset/SeNSE/TCH_processed", type=str, help='path to the output preprocessed folder')
    args = parser.parse_args()

    cohort_id = int(args.input_folder.split('/')[-2].replace('cohort', ''))
    subject_id = int(args.input_folder.split('/')[-1].replace('s', ''))
    
    out_folder = os.path.join(args.out_folder, "c{}s{:02d}".format(cohort_id, subject_id))
    demo_folder = os.path.join("./", "demo")
    if not os.path.exists(args.out_folder):
        os.mkdir(args.out_folder)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    if not os.path.exists(demo_folder):
        os.mkdir(demo_folder)

    print("===========================================")
    print("Start processing:")
    print("Cohort ID: ", cohort_id)
    print("Subject ID: ", subject_id)
    print("===========================================")

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

    # ask if need to regenerate the check file
    checkfile_path = os.path.join(out_folder, "checkfile.pkl".format(cohort_id, subject_id))
    if os.path.exists(checkfile_path):
        print("Check file exists, do you want to regenerate it? (y/n)")
        ans = input()
        if ans == 'y':
            print("Regenerating check file ...")
            ecg, check = load_checkfile(ecg_df, checkfile_path, regenerate=True)
        else:
            ecg, check = load_checkfile(ecg_df, checkfile_path, regenerate=False)
    else:
        print("Generating check file ...")
        ecg, check = load_checkfile(ecg_df, checkfile_path, regenerate=True)

    print("===========================================")
    print("Start generating beats:")
    beats_path = os.path.join(out_folder, "c{}s{:02d}_beats.pkl".format(cohort_id, subject_id))
    start_time = time.time()
    ecg_df_final = generate_beats(ecg, check)
    ecg_df_final.to_pickle(os.path.join(out_folder, "beats.pkl"))
    print("--- {:.3f} seconds ---".format(time.time() - start_time))
    random_row = np.random.randint(0, len(ecg_df_final))
    plot_beat(ecg_df_final, random_row, demo_folder)

    print("===========================================")
    print("Combining with ecg, summary, and glucose")
    final_df = combine_with_summary_glucose(ecg_df_final, summary_df, glucose_df)
    
    print("===========================================")
    print("Clean the rows that have r, t out of range")
    cleaned_final_df = clean_negative_values(final_df)

    print("===========================================")
    print("Saving all the files:")
    # save the final_df, final_hypo_df, and final_normal_df
    final_whole_path = os.path.join(out_folder, "c{}s{:02d}.pkl".format(cohort_id, subject_id))
    print(" - saving whole_df to {}".format(final_whole_path))
    cleaned_final_df.to_pickle(final_whole_path)

    final_hypo_path = os.path.join(out_folder, "c{}s{:02d}_hypo.pkl".format(cohort_id, subject_id))
    final_hypo_df = cleaned_final_df[cleaned_final_df['hypo_label'] == 1]
    print(" - saving hypo_df to {}".format(final_hypo_path))
    final_hypo_df.to_pickle(final_hypo_path)

    final_normal_path = os.path.join(out_folder, "c{}s{:02d}_normal.pkl".format(cohort_id, subject_id))
    final_normal_df = cleaned_final_df[cleaned_final_df['hypo_label'] == 0]
    print(" - saving normal_df to {}".format(final_normal_path))
    final_normal_df.to_pickle(final_normal_path)

    print("===========================================")
    print("Done!")