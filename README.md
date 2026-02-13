# LogBoost: Boost Log Anomaly Detection by Cherry-Picking Log Sequences

> **LogBoost, a lightweight framework to boost log-based anomaly detection by automatically reducing redundant log templates. Based on our proposed similarity measurement, it effectively sorts the importance of log templates and identifies templates that are ineffective in anomaly detection. By filtering out these "noise" templates, LogBoost optimizes the training data, thereby improving the efficiency and accuracy of downstream anomaly detection models.**

---

## 📝 Introduction

In modern distributed systems, logs are generated at an unprecedented rate, often containing a vast amount of redundant information. Traditional anomaly detection models often struggle with this noise, leading to high computational costs and lower accuracy.

**LogBoost** addresses this by introducing a "Cherry-Picking" mechanism. It evaluates the contribution of different log templates to the anomaly detection task using a semantic similarity metric. By selectively preserving high-value log sequences and discarding redundant "noise," LogBoost acts as a universal enhancer for various downstream models, including Deep Learning (e.g., DeepLog, LogAnomaly) and Machine Learning (e.g., SVM, XGBoost) approaches.

---

## ✨ Key Features

* **🍒 Smart Cherry-Picking**: Automatically identifies and filters out redundant log templates based on semantic similarity measurements, optimizing the quality of training data.
* **🚀 Performance Boosting**: By reducing the dimensionality and noise of input data, LogBoost significantly reduces training time while maintaining or improving detection accuracy.
* **📚 Comprehensive Model Zoo**:
* **Deep Learning**: Implementations of LSTM-based models (DeepLog) and Semantic-based models (LogAnomaly).
* **Machine Learning**: Wrappers for Random Forest, XGBoost, SVM, and Logistic Regression.
* **Sequence Matching**: Includes robust sequence matching algorithms like RobustLog.


* **🛠️ End-to-End Pipeline**: Provides a complete workflow from log parsing (Drain/Spell) and feature extraction to model training and evaluation.

---

## 🏗 Architecture & Structure

```text
LogBoost/
├── logboost/
│   ├── boost/                 # Core logic for Data Boosting & Cherry-Picking
│   ├── dataGenerator/         # Data preprocessing and vectorization
│   │   ├── tensor.py          # PyTorch Tensor generation
│   │   ├── sample.py          # Sliding window & negative sampling
│   │   └── ...                # Specific processors for HDFS/Spark
│   ├── models/                # Model Implementations
│   │   ├── lstm.py            # Deep Learning models (DeepLog, LogAnomaly)
│   │   └── ml.py              # Machine Learning wrappers (XGB, RF, SVM)
│   └── utils/                 # Utilities
│       ├── train.py           # Training loops
│       ├── predict.py         # Inference logic
│       └── visualize.py       # Result visualization tools
├── demo/                      # Entry points for running experiments
│   ├── boostlog.py            # Main demo for the LogBoost algorithm
│   ├── deeplog.py             # Baseline: DeepLog
│   ├── loganomaly.py          # Baseline: LogAnomaly
│   ├── robustlog.py           # Baseline: RobustLog
│   ├── xgb.py                 # Baseline: XGBoost
│   └── ...
├── data/                      # Raw datasets (Zipped)
├── logparse_result/           # Intermediate parsing results (Templates/Vectors)
└── requirements.txt           # Dependency list

```

---

## 📦 Prerequisites & Installation

### Environment

* **Python**: 3.8+
* **PyTorch**: 1.8+ (CUDA recommended for Deep Learning models)
* **Scikit-learn**: For Machine Learning models

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-repo/LogBoost.git
cd LogBoost

```


2. **Install dependencies**:
```bash
pip install -r requirements.txt

```



---

## 🚀 Quick Start

### 1. Data Preparation

The repository contains compressed datasets and parsing results to save space. You **must** unzip them before running any models.

```bash
# Unzip raw data
cd data
unzip spark.zip
# (Optional: unzip other datasets if available)

# Unzip intermediate parsing results (Templates & Vectors)
cd ../logparse_result
unzip results.zip
cd ..

```

### 2. Running Anomaly Detection

You can run different models using the scripts provided in the `demo/` folder. The scripts require specific arguments to select the dataset, target setting, and whether to use the original data or LogBoost-enhanced data.

#### A. Deep Learning Models (DeepLog, LogAnomaly, RobustLog)

**Command Syntax:**

```bash
python demo/<model>.py <mode> <dataset> <target> <boost_type> <device>

```

**Arguments:**

* `model`: `deeplog`, `loganomaly`, or `robustlog`
* `mode`:
* `train`: Train the model.
* `predict`: Run inference (prediction).
* `evaluation`: Evaluate model performance (not available for RobustLog).


* `dataset`: `hdfs` or `spark`
* `target`:
* `deep`: Target HDFS-A dataset setting.
* `swiss`: Target HDFS-B dataset setting.
* `spark`: Target Spark dataset setting.


* `boost_type`:
* `origin`: Use original (baseline) data.
* `boost`: Use LogBoost enhanced data.


* `device`: `cpu` or `cuda`

**Examples:**

1. **Run DeepLog on HDFS-A (Original vs. Boosted):**
```bash
# Train original DeepLog on CPU
python demo/deeplog.py train hdfs deep origin cpu

# Train LogBoost-enhanced DeepLog on CPU
python demo/deeplog.py train hdfs deep boost cpu

```


2. **Run LogAnomaly on Spark:**
```bash
# Train original LogAnomaly
python demo/loganomaly.py train spark spark origin cpu

# Predict with LogBoost-enhanced LogAnomaly
python demo/loganomaly.py predict spark spark boost cpu

```


3. **Run RobustLog on HDFS-B:**
```bash
python demo/robustlog.py train hdfs swiss boost cpu

```



#### B. Machine Learning Models (XGBoost, RandomForest)

**Command Syntax:**

```bash
python demo/<model>.py <dataset> <feature_type> <target> <boost_type>

```

**Arguments:**

* `model`: `xgb` or `randomforest`
* `dataset`: `hdfs` or `spark`
* `feature_type`:
* `seq`: Use Sequence Vectors.
* `frq`: Use Frequency Vectors.


* `target`: `deep` (HDFS-A), `swiss` (HDFS-B), or `spark`
* `boost_type`: `origin` or `boost`

**Examples:**

1. **Run XGBoost on HDFS-A (Sequence Vector):**
```bash
# Baseline
python demo/xgb.py hdfs seq deep origin

# Boosted
python demo/xgb.py hdfs seq deep boost

```


2. **Run RandomForest on Spark (Frequency Vector):**
```bash
# Baseline
python demo/randomforest.py spark frq spark origin

# Boosted
python demo/randomforest.py spark frq spark boost

```



#### C. Running the Boosting Algorithm

To generate the boosted datasets yourself (performing the "Cherry-Picking" analysis), you can run:

```bash
python demo/boostlog.py

```

*(Note: You may need to modify the `options` dictionary inside `demo/boostlog.py` to select the specific dataset and parameters you wish to process.)*

---

## 🧪 Supported Models

LogBoost framework supports and compares against the following state-of-the-art models:

| Model | Type | Key Technology | Description |
| --- | --- | --- | --- |
| **LogBoost** | **Ours** | **Cherry-Picking** | **Filters noise templates to boost downstream model performance.** |
| **DeepLog** | Deep Learning | LSTM | Models log patterns as a natural language sequence. |
| **LogAnomaly** | Deep Learning | LSTM + Semantics | Utilizes template semantic vectors to handle new log patterns. |
| **RobustLog** | Sequence Match | Attention/Matching | Robust against parsing errors and noise. |
| **XGBoost** | Machine Learning | Gradient Boosting | High-performance classifier based on log count vectors. |
| **RandomForest** | Machine Learning | Ensemble | Baseline classifier using decision trees. |
| **SVM** | Machine Learning | Hyperplane | Standard baseline for linear classification tasks. |

---

## 📊 Datasets

The framework is optimized for standard log anomaly detection datasets:

* **HDFS**: Distributed file system logs.
* **Spark**: Large-scale data processing engine logs. [Spark-SDA](https://github.com/limin5120/Spark-SDA)
* *(And other custom datasets processed via the `dataGenerator` module)*
