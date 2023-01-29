# LogBoost

## Run Test
This project is an implementation of LogBoost. 

- All example code is placed in the `demo` folder, and the commands to run the tests are at the bottom of each file. 
- The code to run the feature optimization is in `boostlog.py`, and the parameters for each dataset optimization are in the comment area at the top of the file.
- About the running requirements, with python 3.9.13 and run:
  ```bash
  pip install -r requirements.txt
  ```

## Datasets
All data files, including raw data, optimization data, and semantic feature vectors, are placed in the `data` folder.

## Other
The core implementation code of logboost is located in `logboost/boost/boost.py`

The feature data generation code is located in `logboost/dataGenerator`

The kernel code for models is located in the `logboost/models`

Model calling, data reading, visualization, and statistical analysis code are located in `logboost/utils`