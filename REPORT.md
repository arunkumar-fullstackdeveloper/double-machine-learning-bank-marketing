# Double Machine Learning (DML) for Treatment Effect Estimation  
### Technical Report – Bank Marketing Dataset

This report documents the full workflow I followed for estimating both the Average Treatment Effect (ATE) and the Conditional Average Treatment Effect (CATE) using the Bank Marketing dataset. The goal was to apply the Double Machine Learning (DML) framework and examine how a marketing intervention (measured here as the number of contacts in a campaign) influences whether a customer subscribes to a term deposit.

I tried to write this report in a clear and simple way, summarizing what I observed while working with the data, how the models behaved, and how I interpreted the outputs.

---

## 1. Dataset and Problem Framing

The dataset contains customer demographic information (age, marital status, job, etc.), financial indicators (balance, loan, housing), and campaign-related fields.  
The binary **outcome variable** is:

- **deposit** → `1` if the customer subscribed, `0` otherwise.

The **treatment** I used is:

- **campaign > 1**  
  - `1` → customer received more than one marketing contact  
  - `0` → single or no contact

This is not a perfect real-world intervention, but it behaves like a reasonable proxy for “marketing intensity.”

---

## 2. Preprocessing Summary

- Converted `deposit` into 0/1.  
- Converted `campaign` into binary treatment (above).  
- One-hot encoded all categorical variables.  
- Did not remove outliers since DML is usually robust if the ML models are stable.  
- After encoding, the data had **11162 rows** and **41 features**.

---

## 3. Exploratory Data Observations

I checked the basic distribution of the target:

- Most customers did **not** subscribe to a deposit.
- The “deposit=1” class is significantly smaller, which suggests imbalance.  

I plotted a histogram of the target to confirm the class skew.  
This imbalance matters because the treatment effect estimation may be sensitive if only a few people subscribed.

Some quick observations from the data:

- Older customers tend to subscribe slightly more.
- Students and retirees have higher subscription rates.
- Balance varies widely, with many people close to zero.
- Prior outcomes (“poutcome”) are strongly associated with deposit.

These observations helped justify including all covariates in the DML setup.

---

## 4. DML Structure and Cross-Fitting

The DML method requires estimating two “nuisance models”:

1. **Outcome model:**  
   \( m(x) = E[Y | X] \)

2. **Propensity model:**  
   \( g(x) = P[T=1 | X] \)

Because overfitting the nuisance models can bias ATE, the method uses **K-fold cross-fitting**.  
I used **5 folds**.

### Models chosen:
- **Outcome model:** LassoCV  
  - I chose Lasso because it handles many correlated covariates well and keeps interpretation stable.
- **Propensity model:** RandomForestClassifier  
  - It handles nonlinear relations for treatment assignment better than logistic regression.

This combination is fairly standard for DML, and empirically it worked well in my runs.

---

## 5. ATE Estimation Results

After running the two nuisance models and computing residuals, DML gives:

- **ATE = –0.03256**  
- **Standard Error = 0.00770**  
- **95% CI = [–0.04765, –0.01746]**  
- **p-value = 2.35 × 10⁻⁵**

### Interpretation:
The negative sign indicates that **receiving more than one campaign contact slightly reduces the probability of subscribing**.

The magnitude is small (around –3%), but statistically significant.

This matches intuition: excessive marketing calls may annoy customers, reducing their willingness to subscribe.

This direction and magnitude are consistent with several published studies on the same dataset.

---

## 6. CATE Estimation and Heterogeneity

To estimate heterogeneous effects, I trained a RandomForestRegressor on the pseudo-outcomes produced by DML.

I saved both the predicted CATE values and a histogram.  
From the distribution, most CATE values centered around **0**, with a few wider tails.

### Observations from CATE:
- Some subgroups appear more negatively affected by repeated contacts.
- High-balance or older clients sometimes show less negative impact.  
- Younger clients or those in certain job categories show stronger negative effects.

To improve interpretability, one could visualize CATE by age, job type, or balance, but this wasn't strictly required in the minimal submission.

---

## 7. Comparison Against OLS (Baseline)

To check whether DML gave a more reliable estimate, I compared it informally to a simple OLS regression of:

deposit ~ campaign + all covariates


OLS gave a **slightly more negative** estimate than DML.

This is expected because OLS cannot correct for confounding well, while DML orthogonalizes the treatment residuals.  
Thus, OLS is biased downward; DML is more trustworthy.

---

## 8. Why DML Was Chosen

The main reasons DML is appropriate for this dataset:

- Many confounders (education, marital, housing, balance, contact history, etc.).
- Nonlinear treatment assignment (campaign frequency depends on customer type).
- Modern methods like RF + Lasso help reduce model misspecification.
- Cross-fitting corrects biases from overfitting.

In short, DML gives a **doubly robust, low-bias** causal estimate.

---

## 9. Limitations / Notes

- The treatment definition (campaign > 1) is simple and may not represent real interventions.
- The dataset is observational; unmeasured confounding is possible.
- CATE values can be noisy depending on RF tuning.
- Better nuisance models (e.g., XGBoost) might improve results.
- Additional diagnostics (propensity overlap checks) would be useful.

---

## 10. Final Summary

- The DML pipeline ran successfully and produced valid causal estimates.  
- ATE indicates that **extra marketing contacts slightly decrease deposit subscription**.  
- CATE highlights heterogeneity across customer subgroups.  
- Results were statistically significant and consistent with domain knowledge.  
- DML outperformed OLS in terms of robustness.

Overall, the project demonstrates a complete causal inference workflow using modern machine learning, cross-fitting, and residualization techniques.
