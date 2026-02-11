# LogBoost: Boost Log Anomaly Detection by Cherry-Picking Log Sequences

## Introduction

LogBoost, a lightweight framework to boost log-based anomaly detection by automatically reducing redundant log templates. Based on our proposed similarity measurement, it effectively sorts the importance of log templates and identifies templates that are ineffective in anomaly detection. By filtering out these "noise" templates, LogBoost optimizes the training data, thereby improving the efficiency and accuracy of downstream anomaly detection models.

## Project Structure

The project directory is organized as follows:

```
LogBoost/
├── data/                   # Stores datasets (raw, processed, and boosted)
├── demo/                   # Executable scripts for boosting and model running
│   ├── boostlog.py         # Script to run the LogBoost data optimization process
│   ├── deeplog.py          # Entry point for training/testing the DeepLog model
│   └── ...                 # Other model demonstrations (e.g., loganomaly, robustlog)
├── logboost/               # Core library package
│   ├── boost/              # LogBoost core algorithms (template analysis & filtering)
│   ├── dataGenerator/      # Data loaders, sampling (sliding/session window), and preprocessing
│   ├── models/             # Implementations of anomaly detection models (e.g., LSTM)
│   └── utils/              # Utilities for training, prediction, evaluation, and visualization
├── requirements.txt        # Python dependency list
└── README.md               # Project documentation

```

## Key Files and Configuration

### 1. Data Optimization: `demo/boostlog.py`

This script executes the core LogBoost algorithm. It analyzes the dataset to identify log templates that do not contribute positively to anomaly detection and generates a new "boosted" dataset by filtering them out.

* **Configuration**: Modify the `options` dictionary at the top of `demo/boostlog.py` to customize the boosting process.
* `datapath`: Path to the input dataset.
* `data_prefix`: Prefix of the data files (e.g., `hdfs_`).
* `savepath`: Directory where the optimized (boosted) datasets will be saved.
* `alpha`: The similarity threshold for grouping log sequences.
* `topk`: The number of "ineffective" templates to filter out.
* `mode`: Dataset mode (`deep` for DeepLog-style HDFS, `swiss` for others).



### 2. Model Execution: `demo/deeplog.py`

This is the main entry point for training, predicting, and evaluating the DeepLog model. It supports both original and boosted datasets via Command Line Arguments (CLI).

* **Configuration**: Global hyperparameters are defined in the `options` dictionary within `demo/deeplog.py`.
* **Model Params**: `input_size`, `hidden_size`, `num_layers`.
* **Training Params**: `batch_size`, `lr` (learning rate), `max_epoch`.
* **Sampling**: `window_size` (for sliding window), `sample` type.



## Environment Requirements

* **Python Version**: 3.9.13 (Recommended)
* **Dependencies**: Listed in `requirements.txt`. Key libraries include:
* `torch==1.13.1`
* `numpy==1.23.3`
* `pandas==1.5.1`
* `scikit_learn==1.2.1`
* `xgboost==1.7.3`



## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/LogBoost.git
cd LogBoost

```


2. **Install dependencies**:
```bash
pip install -r requirements.txt

```



## Usage

### Step 1: Data Preparation & Boosting

Before training the model, you can optionally run LogBoost to optimize your dataset. This process generates new training and testing files with the prefix `boost_`.

1. Open `demo/boostlog.py` and ensure the `options` correspond to your target dataset (e.g., HDFS).
2. Run the boosting script:
```bash
cd demo
python boostlog.py

```


*Output*: You will see logs indicating the "Recommended filter template" and confirmation that `boost_hdfs_train`, `boost_hdfs_test_normal`, etc., have been saved to the specified `savepath`.

### Step 2: Model Training

You can train the model on either the **original** dataset or the **boosted** dataset using the CLI.

**Syntax**:

```bash
python deeplog.py [mode] [data_source] [target_type] [data_version] [device]

```

**Examples**:

* **Train on Original HDFS Data**:
```bash
python deeplog.py train hdfs deep origin cpu

```


* **Train on Boosted HDFS Data**:
```bash
python deeplog.py train hdfs deep boost cpu

```



### Step 3: Model Prediction

After training, generate prediction results on the test set.

* **Predict on Boosted HDFS Data**:
```bash
python deeplog.py predict hdfs deep boost cpu

```


*Output*: This will save prediction results (e.g., `res_normal_abnormal.npz`) in the result directory specified in `options['save_dir']`.

### Step 4: Evaluation

Evaluate the model's performance (Precision, Recall, F1-measure) based on the prediction results.

* **Evaluate Boosted HDFS Data**:
```bash
python deeplog.py evaluation hdfs deep boost cpu

```



## Customization & Extension

* **Adding New Models**: Place your model architecture in `logboost/models/`. You can reference `deeplog.py` to create a new runner script in the `demo/` folder.
* **New Datasets**: Ensure your data parsing logic is compatible with `logboost/dataGenerator/sample.py`. If using a new log format, you may need to adjust the `read_file` or windowing functions.
