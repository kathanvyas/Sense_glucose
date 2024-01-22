# ECG Processing

## Get Started
### Installation
1. Clone the repository
```
git clone https://github.com/Morris88826/Sense_glucose.git
cd Sense_glucose
```
2. Install dependent packages
```
conda create --name ecg python=3.11
conda activate ecg
pip install -r requirements.txt
```
### Download the Data
We uses the data in the **TCH: Cohort 1 Data** and **TCH: Cohort 2 Data** folder from the **[SeNSE TAMU](https://drive.google.com/drive/folders/1Pts4PLTFIYqpPU53k8ZE4H-J4zZNH-WY?usp=drive_link)** team drive.
* Please reach out to Professor [Gutierrez-Osuna](mailto:rgutier@cse.tamu.edu) at the PSI Lab in the Department of Computer Science & Engineering at Texas A&M University if you wish to access the ECG data.

There should be five subject folders (S01-S05) in both **TCH: Cohort 1 Data** and **TCH: Cohort 2 Data** respectively. We only need the ***zephyr*** and ***cgm*** folders from each.
* For zephyr: download all folders inside
* For cgm: There should be one Clarity_report_..._.csv file

Download all the data and put them in the same folder. For example:

- SeNSE
  - TCH
    - cohort1
      - s01
        - 2022_06_08-13_32_45
        - 2022_06_08-17_47_46
        - ...
        - Clarity_Export_C01S01_2022-07-05.csv
      - s02
    - cohort2

## Preprocessing
After downloading all the data and placed in the files structure mentioned above, please run the following command for data preprocessing.

This reads all \*_ECG.csv/\*_SummaryEnhanced.csv files in the folder and combines them in one single pickle file respectively (ECG.pkl, summary.pkl). It also convert the raw glucose metadata file into the desired glucose file format (glucose.pkl).

```
python libs/preprocess.py --folder_path <folder_path> --glucose_path <glucose_path>
```
- **folder_path**: The folder path containing all the downloaded ECG and summary files for each subject. 
    - Ex: "./SeNSE/TCH/cohort1/s01".
- **glucose_path**: The path towards the raw glucose metadata file, i.e. the clarity report. 
    - Ex: "./SeNSE/TCH/cohort1/s01/Clarity_Export_C01S01_2022-07-05.csv"
- out_folder: Optional. If not provided, it be the same as the folder_path. This is the where the preprocessed **ECG.pkl**, **summary.pkl**, and **glucose.pkl** are saved.

## Process the raw data into ECG beats 
After the preprocessing step is complete, we can begin processing the data by running the command:

```
python main.py --input_folder <input_folder> --out_folder <out_folder>
```
- **input_folder**: The folder where the ECG.pkl, summary.pkl, and glucose.pkl are saved, i.e. the out_folder path in the preprocessing step.
    - Ex: “./SeNSE/TCH/cohort1/s01”.
- **out_folder**: This is the folder where all the final extracted ECG beat data is stored. It will create the cohort and subject folder automatically based on your input_folder.
    - Ex: Create a new folder, **TCH_processed**, under the "SeNSE" folder, i.e. “./SeNSE/TCH_processed”.

The processed ecg beats data (**c1s01.pkl**, **c1s01_hypo.pkl**, **c1s01_normal.pkl**) will be stored under "./SeNSE/TCH_processed/c1s01" for the example above.

Here is an example of the beat extracted.
<p align="center">
    <img src="https://hackmd.io/_uploads/Hkre912ta.png"  width="50%">
</p>
