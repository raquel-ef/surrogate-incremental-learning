# Surrogate-assisted multi-objective evolutionary feature selection of generation-based fixed evolution control for time series forecasting with LSTM networks


## Overview
This repository contains the implementation of the research paper "Surrogate-assisted multi-objective evolutionary feature selection of generation-based fixed evolution control for time series forecasting with LSTM networks." The study focuses on improving feature selection (FS) for LSTM-based time series forecasting using surrogate-assisted multi-objective evolutionary algorithms (MOEAs).

The main goal of this project is to reduce the excessive computational time required by traditional wrapper-based FS while maintaining or improving predictive accuracy. The approach involves using surrogate models with incremental learning (IL) and conventional machine learning methods to update the surrogate model. In order to address this challenge, this project introduces: 

- **Surrogate-Assisted MOEA**. Implements a MOEA to optimize FS.
- **Incremental Learning vs. Conventional Learning**. Compares different approaches for updating the surrogate model.
- **LSTM for Time Series Forecasting**. Evaluates FS effectiveness for LSTMs.
- **Multiple Surrogate Models**. Utilizes SGD and MLP for IL, and RF and SGD for conventional approaches.


## Project Structure
``` plaintext
📂 config/                      # Configuration file
📂 data/                        # Dataset storage
📂 models/                      # Trained models
📂 notebooks/                   # Jupyter Notebooks with examples
 ├── Incremental learning surrogate-assisted.ipynb
 ├── Non-incremental surrogate-assisted.ipynb
 ├── Surrogate model creation.ipynb
📂 problems/                    # Problem-specific implementations
 ├── AttributeSelection.py
📂 src/                         # Source code
 ├── data_processing.py             # Preprocessing and cleaning
 ├── evaluation.py                  # Model evaluation scripts
 ├── utils.py                       # Utility functions
📂 variables/                   # Variable storage
📜 requirements.txt             # Dependencies
```

## Installation
Ensure you have Python 3.10 installed and install the following dependencies:
```sh
pip install -r requirements.txt
```


## Usage
1. **Prepare the dataset**. Place data in `.arff` format in the [data](/data/) directory. Note that the data must be previously transformed using a sliding window method (see function `lags` in [utils](/src/utils.py) for this transformation).
2. **Train the surrogate models**. Train the surrogate models for each approach as in the example notebook [Surrogate model creation](/notebooks/Surrogate%20model%20creation.ipynb). The parameters should be previously configured in [config](/config/config.py).
3. **Comparisons**. Run and compare IL and non-incremental approaches for FS as in the example notebooks ([Incremental learning surrogate-assisted](/notebooks/Incremental%20learning%20surrogate-assisted.ipynb) and  [Non-incremental surrogate-assisted](/notebooks/Non-incremental%20surrogate-assisted.ipynb)).



## Citation
If you use this software in your work, please include the following citation:
```
@article{espinosa2024surrogate,
  title={Surrogate-assisted multi-objective evolutionary feature selection of generation-based fixed evolution control for time series forecasting with LSTM networks},
  author={Espinosa, Raquel and Jim{\'e}nez, Fernando and Palma, Jos{\'e}},
  journal={Swarm and Evolutionary Computation},
  volume={88},
  pages={101587},
  year={2024},
  publisher={Elsevier}
}
```

## License
[MIT License](/LICENSE)

