# Weather RNNs, LSTMs & GRUs

A time-series forecasting project exploring **vanilla RNNs, LSTMs, and GRUs** for Melbourne daily temperature forecasting.

The project focuses not only on model performance, but also on an important practical question:

> **Does a more complex recurrent architecture actually produce better forecasts than a simpler model?**

The experiments compare recurrent architectures, multi-step forecasting strategies, simple statistical baselines, and different sequence lengths.

---

## Project Structure

```text
Weather_LSTMs_GRUs/
│
├── Notebooks/
│   ├── Melbourne_Weather_EDA.ipynb
│   ├── RNN_LSTM_GRU_Training.ipynb
│   ├── Multi_Step_Predictions(Recursive_Direct).ipynb
│   └── Sequence_Length_Sensitivity.ipynb
│
└── README.md
```

### Notebooks

| Notebook                                         | Purpose                                                                                    |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| `Melbourne_Weather_EDA.ipynb`                    | Data exploration, cleaning, missing-date handling, interpolation, and temperature analysis |
| `RNN_LSTM_GRU_Training.ipynb`                    | Training and comparison of vanilla RNN, LSTM, and GRU models                               |
| `Multi_Step_Predictions(Recursive_Direct).ipynb` | Multi-step forecasting using recursive and direct strategies                               |
| `Sequence_Length_Sensitivity.ipynb`              | Investigating how the amount of historical context affects forecasting performance         |

---

## Dataset

The project uses daily weather observations from **Melbourne, Australia**, with temperature as the primary forecasting variable.

The time series is converted into supervised learning examples using sliding windows.

For a sequence length \(S\):

```text
Past S observations
        ↓
   Recurrent model
        ↓
Future temperature
```

For example, with `S = 14`:

```text
14 previous days → predict the next day
```

For multi-step forecasting with horizon `H = 7`:

```text
14 previous days → predict the next 7 days
```

---

## Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* PyTorch
* Jupyter / Google Colab

---

## Reproducibility

The experiments are organized into separate notebooks so that each stage can be inspected independently:

```text
EDA
 ↓
Model Training
 ↓
Multi-Step Forecasting
 ↓
Sequence-Length Sensitivity
```

Models are evaluated using time-aware train/validation/test splits, with preprocessing fitted only on the training period.
