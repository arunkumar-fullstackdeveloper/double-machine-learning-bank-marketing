# Double Machine Learning (DML) for Causal Treatment Effect Estimation  
### Bank Marketing Dataset – Final Project

---

## 📘 Project Summary

This project implements **Double Machine Learning (DML)** to estimate the **Average Treatment Effect (ATE)** and **Conditional Average Treatment Effect (CATE)** using the Bank Marketing Dataset.

DML provides **robust causal inference** by:

- Using machine learning models (Lasso & Random Forest) to estimate nuisance functions  
- Correcting confounding bias using **orthogonalization**  
- Enhancing generalization using **cross-fitting**  
- Producing reliable ATE & CATE estimates even in high-dimensional data  

This project follows all required tasks:  
✔ EDA  
✔ DML implementation  
✔ ATE estimation  
✔ CATE estimation  
✔ Visualization  
✔ Interpretation & critique  

---

## 📂 Project Structure

Double_Machine_Learning_DML_Project/
│
├── data/
│   └── bank.csv
│
├── plots/
│   ├── outcome_distribution.png
│   ├── cate_distribution.png
│   └── dml_results.csv
│
├── src/
│   └── dml_analysis.ipynb
│
├── README.md
└── REPORT.md

---

## 📊 Dataset Details

**Dataset:** Bank Marketing Dataset  
**Rows:** 11,162  
**Features:** Demographic + financial + campaign info  

### 🔸 Outcome Variable (Y)
`deposit`  
- yes → 1  
- no → 0  

### 🔸 Treatment Variable (T)
`campaign`  
- Original: Number of contacts  
- Converted to binary:
  - `0` → contacted ≤ 1 time  
  - `1` → contacted > 1 time  

### 🔸 Covariates (X)
All remaining categorical + numeric features  
Categorical variables one-hot encoded  

---

## ⚙️ Methodology (DML Pipeline)

1. **K-fold cross-fitting (K=5)**  
2. Estimate nuisance functions:
   - `m(x)` → LassoCV  
   - `g(x)` → RandomForestClassifier  
3. Compute orthogonal residuals:
   \[
   \tilde{Y} = Y - \hat{m}(X)
   \quad\text{and}\quad
   \tilde{T} = T - \hat{g}(X)
   \]
4. Estimate ATE using Neyman orthogonal score  
5. Estimate CATE using Random Forest on pseudo-outcomes  
6. Produce plots + results CSV  

---

## 📈 Key Results

### 🟦 **Average Treatment Effect (ATE)**

ATE: -0.03194
Standard Error: 0.00773
95% CI: (-0.0470, -0.0167)
p-value: 3.62e-05


### Interpretation  
- **Negative effect** → Repeated calls reduce chance of deposit subscription  
- **Strong significance** (p < 0.001)  
- Customers may be **annoyed by multiple contact attempts**  

---

### 🟩 **Conditional Average Treatment Effect (CATE)**  
CATE distribution shows:

- Most customers have negative treatment effects  
- Some segments still show positive values  
- Supports **heterogeneous treatment effects**  
- Useful for **customer personalization**  

---

## 📁 Output Files

All outputs saved in `plots/`:

plots/
│
├── outcome_distribution.png
├── cate_distribution.png
└── dml_results.csv


---

## ▶️ How to Run

1. Upload dataset into `/data/bank.csv`  
2. Open `src/dml_analysis.ipynb` in Google Colab  
3. Mount Drive → Run all cells  
4. Outputs auto-generated in `/plots/`  
5. Review ATE/CATE results & visualizations  

---

## 📦 Requirements

numpy==1.26.4
pandas==2.1.3
scikit-learn==1.3.2
matplotlib==3.8.2
scipy==1.11.4
joblib==1.3.2


---

## 📝 Additional Documentation

Full technical explanation, interpretation, insights, and critique can be found in:

📄 **REPORT.md**

---

## ✔️ Conclusion

- Multiple campaign calls **significantly reduce** subscription probability  
- CATE reveals heterogeneous responses  
- DML effectively removes confounding bias  
- Insights can help design **personalized marketing strategies**  
- Project satisfies all requirements: EDA → DML → ATE → CATE → Interpretation  

---


