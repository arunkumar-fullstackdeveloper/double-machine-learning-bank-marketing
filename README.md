# Double Machine Learning (DML) – Treatment Effect Estimation
### Bank Marketing Dataset – Causal Inference Project

This project implements the Double Machine Learning (DML) framework to estimate the causal effect of repeated marketing contacts on the likelihood of a customer subscribing to a term deposit.  
The method corrects for high-dimensional confounding using cross-fitting and machine learning models.

---

## 1. Project Structure

project/
│── README.md  
│── REPORT.md  
│── requirements.txt  
│  
├── Data/  
│    └── bank.csv  
│  
└── Plot/  
     ├── outcome_distribution.png  
     ├── cate_distribution.png  
     └── dml_results.csv  

- **Data/** contains the Bank Marketing dataset  
- **Plot/** contains all generated visualizations and exported results  
- **REPORT.md** contains the full technical explanation of the pipeline  

---

## 2. Objective

The goal is to estimate:

- **Average Treatment Effect (ATE)**  
- **Conditional Average Treatment Effect (CATE)**  

Treatment variable:  
`campaign > 1` → interpreted as “more than one marketing contact”.

Outcome variable:  
`deposit` → whether the customer subscribed (1) or not (0).

---

## 3. Methods Overview

### Nuisance Models
- **Outcome model:** LassoCV  
- **Propensity model:** RandomForestClassifier  

### Double Machine Learning
- Used 5-fold cross-fitting  
- Residualized outcome and treatment  
- Orthogonalized score for ATE estimation  

### CATE Estimation
- Trained RandomForestRegressor on pseudo-outcomes  
- Produces customer-level heterogeneous treatment effects  

---

## 4. Key Results

### ATE
- Approximately **–0.0325**  
- Interpretation: repeated calls lower subscription chances by ~3%  
- Effect is statistically significant (low p-value and tight CI)

### CATE
- Most individuals show minor negative effects  
- Some demographic groups show larger reactions (positive or negative)

---

## 5. Generated Outputs

Located inside the **Plot/** folder:

- `outcome_distribution.png`  
- `cate_distribution.png`  
- `dml_results.csv` → includes CATE values for every row  

---

## 6. Running the Project

### Install dependencies
pip install -r requirements.txt


### Run the script (Colab recommended)
python dml_script.py

Or run each cell manually in Google Colab.

Dataset should be located at:
Data/bank.csv

---

## 7. Notes

- Dataset is observational; unmeasured confounding may exist.  
- Treatment variable is a proxy for marketing intensity, not a perfect measure.  
- Hyperparameters can be tuned for more stable CATE values.  
- DML is used because it handles complex confounding better than OLS.

---



