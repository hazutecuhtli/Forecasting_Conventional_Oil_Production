#  📊 Forecasting Mexican Oil Production Using Deep Learning

This project demonstrates the use of deep learning techniques—specifically CNNs, LSTM, and GRU—for forecasting conventional oil production in Mexico. The goal is to explore and compare different architectures for time series prediction using TensorFlow, focusing on the practical application of sliding windows, cyclical time features, and uncertainty estimation via Monte Carlo Dropout.

This is a practice excersice, but it is important to understand the beahiovur of conventional oil production from PEMEX, Mexico’s state-owned oil company. It is an analysis based on conventional oil production since is not considering additional sources such as condensanted oil.

## Introduction

The production of hydrocarbons remains a critical topic for various types of analysis, as it is closely tied to several key industries such as energy, plastics, and automotive. Nonetheless, although several approaches have been developed to reduce reliance on this essential resource for energy generation—such as renewable sources including wind, solar, and geothermal—in many countries, like Mexico, hydrocarbons are still considered a strategic asset that supports numerous core productive sectors.

Hence, it is important to understand the behavior of hydrocarbon production in order to analyze its impact on various aspects of a country, such as energy generation or even the financial health of the state-owned company responsible for extracting these resources—as is the case with the selected company, PEMEX. Several factors can affect oil production, such as the depletion of underground reservoirs due to ongoing exploitation, the lack of investment in the exploration and drilling of new potential reservoir locations, among others.

## 📁 Repository Structure

```text
├── Forecasting_oil_production_using_DL.ipynb  # Main notebook with the full implementation
├── utils_forecasting.py                       # Auxiliary functions and model utilities
├── data/                                      # Folder containing PEMEX oil production
└── README.md                                  # You are here! :)
```

## 🛠️ Technologies Used

    🐍 Python 3
    🔮 TensorFlow / Keras
    📊 Pandas, NumPy, Matplotlib, Seaborn
    🔁 TQDM
    🧪 Scikit-learn

## 📈 Models Implemented

| Architecture | Purpose                              | Notes                                 |
|--------------|--------------------------------------|---------------------------------------|
| **CNN**      | Short-term trend detection           | Fast but lacks memory                 |
| **LSTM**     | Long-term dependencies               | More complex, higher temporal reach   |
| **GRU**      | Balanced short/medium-term prediction| Performs best in this notebook        |

## 🔍 Features & Highlights

✅ Custom sliding window implementation

✅ Cyclical encoding for months

✅ Monte Carlo Dropout for uncertainty estimation

✅ Fully modular and reusable codebase

✅ Visual analysis of model behavior and predictions

## 📊 Results Summary

The GRU-based model outperformed CNN and LSTM models in short- and medium-term prediction accuracy. Confidence intervals provide insight into model uncertainty, helping assess reliability of forecasts.

## 🎯 Motivation

This notebook was implemented as a personal exercise to practice and improve the use of TensorFlow for building forecasting models, particularly in the context of time series data. The selected dataset focuses on oil production due to a personal interest in the oil & gas industry, where time series modeling plays a crucial role across various stages in the life cycle of reservoirs and wells, especially in upstream operations such as exploration, drilling, completion, and production.

## 🙏🏽 Acknowledgements

Special thanks to my father and sister for inspiring me to explore such an important and fascinating field as the oil and gas industry—especially the study of hydrocarbon reservoir exploitation and production. I would also like to thank ChatGPT for all the support during this project, helping speed up development and improve the overall quality of the final result.


![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?logo=python&logoColor=white)
![Built with TensorFlow](https://img.shields.io/badge/Built%20with-TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Powered by Pandas](https://img.shields.io/badge/Powered%20by-Pandas-150458?logo=pandas&logoColor=white)
![Runs on Jupyter](https://img.shields.io/badge/Runs%20on-Jupyter-F37626?logo=jupyter&logoColor=white)
![Helped by GPT](https://img.shields.io/badge/Helped%20by-ChatGPT-10a37f?logo=openai&logoColor=white)

