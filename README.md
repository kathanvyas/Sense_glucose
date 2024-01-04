# Zephyr_ECG_Summary_Glucose data processing

## Overview
This project is a comprehensive suite for extracting ECG morphology, RR, and statistical features from Zephyr devices. It is structured in a series of steps, each encapsulated in individual Python scripts, to provide a modular and easy-to-understand approach to creating datasets to feed to DL networks. The repository contains code for the TAMU-Sense Project. Contains code from reading the ECG-Sumamry file from the Zephyr folder and then processing it with the reading glucose file. The code contains all necessary functions, from reading to creating a dataset before feeding it to CNN and RNN networks.

## Repository Structure

### preprocessing data
Zephyr BioHarness provides the following files: 
1) record_timestamp_ECG.csv
2) record_timestamp_SummaryEnhanced.csv

You should also have one glucose file. Ensure the glucose file contains no missing values and only numerical values in the glucose column. you can use preprocessing/helper_glucose.py

Run the file preprocessing/ECG_read_and_combine.py -> This reads all ECG.csv files and combines them in one single pickle file.


### `step1_create_beats.py`
- **Description**: The ECG.pkl (combined all ecg files), summary.pkl (combined all summary files) and glucose.pkl file. The Python file creates separate beats from all the input files. it checks for HRConfidence, and ECGnoise and also develops variable length beats. This will generate a hypo_label that corresponds to the hypo vs normal glucose classification problem
- **Usage**: `python step1_create_beats.py [arguments]`
- **Key Functions**: [List key functions, e.g., "create_beats", "filter_noise"]

### `step1_create_beats_hyper.py`
- **Description**: The ECG.pkl (combined all ecg files), summary.pkl (combined all summary files) and glucose.pkl file. The Python file creates separate beats from all the input files. it checks for HRConfidence, and ECGnoise and also develops variable length beats. This will generate a hyper_label that corresponds to the hyper vs normal glucose classification problem
- **Usage**: `python step1_create_beats_hyper.py [arguments]`

### `step2_PCA_mahlanobis_filtering.py`
- **Description**: [Details of PCA and Mahalanobis filtering, e.g., "Applies PCA for dimensionality reduction and Mahalanobis distance for anomaly detection."]
- **Usage**: `python step2_PCA_mahlanobis_filtering.py [arguments]`

### `step3_PCA_PAD_beats_creation.py`
- **Description**: [Integration of PCA and PAD in beat creation, e.g., "Combines PCA with PAD for optimized beat creation."]
- **Usage**: `python step3_PCA_PAD_beats_creation.py [arguments]`

### `step4_Model_CNN.py`
- **Description**: [CNN model implementation, e.g., "Implements a Convolutional Neural Network for pattern recognition."]
- **Usage**: `python step4_Model_CNN.py [arguments]`

### `step4_model_cnn_fixedwindow.py`
- **Description**: [CNN with a fixed window, e.g., "CNN model adapted to a fixed window size for uniform input processing."]
- **Usage**: `python step4_model_cnn_fixedwindow.py [arguments]`

### `step5_creating_data_for_RNN.py`
- **Description**: [Data preparation for RNN, e.g., "Prepares and structures data for Recurrent Neural Network processing."]
- **Usage**: `python step5_creating_data_for_RNN.py [arguments]`

## Installation
Provide details on how to set up the environment for running these scripts. Include instructions for installing dependencies, setting up virtual environments, etc.

## How to Contribute
Instructions for how others can contribute to this project. Include guidelines for coding standards, pull request process, etc.

## License
Specify the license under which this project is released.

## Contact
Provide your contact information or a way for users to reach out with questions or contributions.

---

### Enhancing the README with Graphics

1. **Flowchart**: Include a flowchart at the start to visually depict the flow of the scripts.
2. **Screenshots**: Add screenshots of the script outputs or visualizations they produce.
3. **Code Snippets**: Use markdown code blocks to highlight important code snippets or usage examples.

---

