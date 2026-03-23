# Data Quality and Robustness in Tabular Machine Learning

This repository contains the code and experiments for my master's thesis in Computer Science at Gisma University of Applied Sciences.

I studied how incorrect training labels, missing values and unavailable features affect Logistic Regression and LightGBM. The experiments use hospital readmission data and US employment data, testing both predictive performance and the reliability of predicted probabilities on different populations.

## Explore the project

- [Diabetes experiments](Diabetic.ipynb)
- [Employment experiments](ACSEmployment.ipynb)
- [Experiment code](experiments/)
- [Results](results/)

The notebooks explain the experiments step by step and use the shared code in `experiments/`.

## Run the experiments

The reported results used Python 3.14.5 on macOS. From the repository root:

```bash
python3 -m pip install -r requirements.txt
bash reproduce.sh
```

This regenerates the results, tables and figures. You can also run the experiments through the notebooks.

The UCI Diabetes data are included in `raw_data/` (DOI: 10.24432/C5230J).
Folktables downloads the 2018 California and Texas employment data on the first run.
