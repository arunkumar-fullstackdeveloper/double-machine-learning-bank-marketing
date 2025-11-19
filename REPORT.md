# Double Machine Learning (DML) for Treatment Effect Estimation  
### Bank Marketing Dataset — Final Project Report

---

## 1. Introduction

This project applies **Double Machine Learning (DML)** to estimate both the **Average Treatment Effect (ATE)** and **Conditional Average Treatment Effect (CATE)** using the Bank Marketing Dataset.  
DML provides robust causal inference by removing bias introduced from high-dimensional confounders through:

- Orthogonalization  
- Cross-fitting  
- Separate nuisance estimation for outcome and propensity models  

The goal is to identify whether a marketing intervention (campaign contacts) increases the probability that a customer subscribes to a deposit product.

---

## 2. Dataset Description

The dataset contains **11,162 observations** with demographic, financial, and communication-related features.

- **Outcome (Y):** `deposit`  
  - Encoded: yes = 1, no = 0  
  - Represents whether the customer subscribed to a term deposit.

- **Treatment (T):** `campaign`  
  - Original: number of marketing contacts  
  - Converted to binary:
    - 0 → contacted ≤ 1 time  
    - 1 → contacted more than once  
  - This creates a valid binary intervention indicator for DML.

- **Covariates (X):**  
  - age, job, marital, education, balance, housing, loan, contact type, month, duration, pdays, previous campaign outcomes, etc.  
  - All categorical variables were one-hot encoded.

---

## 3. Exploratory Data Analysis (EDA)

### **Outcome Distribution**
- Majority outcomes:
  - ~5800 customers: did **not** subscribe (`0`)
  - ~5200 customers: subscribed (`1`)
- Outcome is balanced enough for causal modeling.

### **Treatment Assignment**
- In binary form:
  - A large portion had only 1 contact  
  - A smaller portion had multiple contacts  
- Propensity scores were clipped between 0.01 and 0.99 to avoid extreme probabilities.

### **Covariates**
- Mix of categorical and numerical features  
- High-cardinality categorical variables handled using one-hot encoding  
- No missing values after imputation  

---

## 4. DML Implementation

### **Cross-Fitting**
Used 5-fold cross-fitting:

1. Outcome model:  
   - `LassoCV` with scaling and imputation  
2. Treatment model:  
   - RandomForestClassifier (200 trees)  
3. Orthogonal residuals computed:  
   - \( \tilde{Y} = Y - \hat{m}(X) \)  
   - \( \tilde{T} = T - \hat{g}(X) \)

This ensures unbiased estimation of the causal effect in high dimensions.

---

## 5. ATE Estimation Results

The final ATE estimate:

ATE = -0.03194085
Standard Error = 0.00773378
95% CI = (-0.0470, -0.0167)
p-value = 3.62e-05


### **Interpretation**
- **Negative ATE** indicates that giving multiple campaign contacts **reduced** the probability of deposit subscription.  
- Confidence interval does **not** include zero ⇒ statistically significant.  
- Very small p-value (<0.001) ⇒ strong evidence of a causal negative effect.

### **Business Meaning**
This suggests that **over-contacting customers harms marketing performance**, possibly due to annoyance, fatigue, or perceived pressure.

---

## 6. CATE (Heterogeneous Treatment Effects)

CATE was estimated using:

- Pseudo-outcome formulation  
- RandomForestRegressor (300 trees)

### **CATE Distribution Findings**
- Most CATE values lie between **−3 to +3** percentage points.  
- Majority show **negative impact**, confirming ATE result.  
- Small pockets of positive CATE exist → some customer segments still respond well under repeated contact.

### Potential high-response groups:
- Certain job categories  
- Customers with higher balances  
- Customers without previous negative campaign interactions  

***This can guide targeted marketing strategies.***

---

## 7. Comparison to OLS Baseline

Unlike simple OLS:

| Aspect | OLS | DML |
|-------|-----|-----|
| Handles high-dimensional covariates | ❌ | ✔ |
| Orthogonalization to remove bias | ❌ | ✔ |
| Bias from overfitting | High | Low |
| Valid inference with ML models | ❌ | ✔ |

**Reason for choosing DML:**  
It isolates the treatment effect from confounding noise using machine learning for nuisance estimation, giving more reliable causal estimates.

---

## 8. Limitations

1. Treatment converted from continuous to binary may simplify real behavior.  
2. DML assumes no unobserved confounders (ignorability).  
3. CATE estimation depends on Random Forest quality—may vary slightly.  
4. Campaign duration variable is highly predictive and may dominate the model.

---

## 9. Conclusion

- Multiple marketing contacts (treatment) **reduced** deposit subscription probability overall.  
- Strong statistical significance confirms robustness.  
- However, heterogeneity exists — some segments show positive CATE, offering opportunities for personalized marketing.  
- DML proved superior to traditional methods by providing low-bias, high-confidence causal insights.

---

## 10. Files Generated

- `dml_results.csv` – includes m_hat, g_hat, CATE  
- `outcome_dist.png` – outcome distribution  
- `cate_dist.png` – CATE distribution  
- Full Python DML code (included in GitIngest submission)

---

## 11. Reproducibility Instructions

1. Load the dataset  
2. Convert `deposit` → binary  
3. Convert `campaign` → binary treatment  
4. Run DML code in Google Colab  
5. Ensure all libraries installed  
6. Examine ATE and CATE outputs  
7. Use plots for interpretation  
8. Submit results + analysis

---

# End of Report
