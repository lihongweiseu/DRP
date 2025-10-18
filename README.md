## Codes and data for the research:

📝 Exploiting small-scale neural network models for dynamic response prediction problems

## ⚙️ Requirements
- Python 3.11+
- Pytorch 2.4.1+

## 🗂️ Project Structure

```text
├── 1. Bouc-Wen/                       # Case 1
│   ├── BW_RNN.py ~ BW_TCN.py          # Main training scripts for five neural networks
│   ├── BW_data.mat                    # Training and testing datasets
│   ├── saved_models                   # Checkpoints for trained models
│   ├── saved_data                     # Saved data (e.g., loss evolutions) for checkpoints
│
├── 2. Wiener Hammerstein/ - 5. Nonlinear system/ are similar to 1. Bouc-Wen/
│
├── RNN_tools.py, LSTM_tools.py, DSNN_tools.py, AUNN_tools.py, and TCN_tools.py
│   present tools to define and train the five neural networks
│
├── general_tools.py                  # time display and training early stop function.
│
├── README.md                         # Project overview and usage instructions
├── LICENSE.txt                       # MIT license
└── .gitattributes                    # Language and encoding declaration
```
