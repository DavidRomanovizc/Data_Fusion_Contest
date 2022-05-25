# Data Fusion 2022 Contest

8th place solution for [Data Fusion 2022 Contest](https://ods.ai/competitions/data-fusion2022-main-challenge).

| Rank       | Public      | Private      |
|------------|-------------|--------------|
| Matching   | 6           | 8            |


## Used technology
![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)
![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![scikit_learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Problem solving

1. The main part of the task was the development of functions
Before we start analyzing transactional data, we need to create useful features based on the transaction_dttm and transaction_amt columns.
This will allow you to get more information in the context of various measurements in the future (such as time of day, days of the week, etc.), as well as use the obtained functions in machine learning models.

2. Training: 
   - CatBoostRanker with YetiRank loss with 9000 iterations,
   - Ensembling of 2 catboost models with different parameters.


## Data
1. General data for all tasks in a tabular `.csv` format: `transactions.zip, clicstream.zip` and the target variable `train_matching.csv`
2. Common accompanying data for all tasks in tabular `.csv` format: `mcc_codes.csv`, `click_categories.csv` and `currency_rk.csv`
3. Baselines and examples of solutions for a container Matching problem: random solution `sample_submission.zip` and `baseline_catboost.zip` with an example of a solution based on the catboost library using GPU
