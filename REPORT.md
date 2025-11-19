1. Introduction

The goal of this project is to estimate the effect of repeated marketing contact on a customer’s decision to subscribe to a deposit.
I use a Double Machine Learning (DML) approach.
The treatment is the number of contacts.
I convert it into a binary variable:

campaign > 1 → treated

campaign ≤ 1 → control

The outcome variable is whether the customer subscribed (“deposit”).
This is mapped to 1 and 0.

2. Data Preparation and Cleaning

The dataset contains 11,162 rows.
There are demographic, financial, and call-related variables.
Some variables are categorical, so I encoded them using one-hot encoding.
No major missing values were found.
All categorical columns were converted before the DML step.

Main confounders observed:

age

job

balance

previous campaign contact

call duration

education

housing loan

These may influence both treatment and outcome, so they must be included in the model.

3. Exploratory Data Analysis (EDA)
3.1 Outcome Distribution

The subscription rate is low.
The histogram shows most values are 0.
Only a small portion of customers subscribed.

3.2 Treatment Distribution

A large portion of customers received only one contact.
A smaller group received more than one contact.

3.3 Key Patterns

Customers with longer call duration tend to subscribe more.

People contacted many times usually show lower interest.

Previous successful marketing campaigns show strong positive influence.

These observations support the need for a method that handles confounding.

4. DML Implementation

I follow the standard DML structure:

Outcome model (m(x)): Lasso regression

Propensity model (g(x)): Random Forest classifier

Cross-fitting: 5 folds

Residualization: outcome and treatment

Final ATE: ratio between residuals

I used clipping on propensity scores to avoid values near 0 or 1.

5. Average Treatment Effect (ATE)

Results from my run:

ATE: −0.032558

Std Error: 0.0077

95% CI: [ −0.04765 , −0.01746 ]

p-value: 2.35e-05

Interpretation

The estimate is negative.
This means contacting a customer more than once reduces the chance of subscription on average.
The result is statistically significant.
The confidence interval does not cross zero.

This supports the idea that repeated calls may annoy customers.

6. Conditional Average Treatment Effect (CATE)

I trained a Random Forest model on the pseudo-outcome to estimate heterogeneous effects.

CATE observations:

Most CATE values are near zero.

Some customers show strong negative effect.

A small group shows slightly positive effect.

Practical meaning

The treatment effect is not equal across all customers.
The company should not call everyone multiple times.
Targeted calling may be better.

7. Comparison with a Simple Baseline

For reference, I tested a basic OLS model on the raw data.
The effect estimate was close to zero and not significant.
This shows OLS cannot handle confounding well.
DML gives a clearer and more reliable effect.

8. Limitations

Only one dataset was used.

The models used default parameters except for small adjustments.

The pseudo-outcome can be unstable for a few rows.

Business factors outside the dataset are not included.

9. Conclusion

DML provides a structured way to estimate treatment effects in the presence of many confounders.
The analysis shows a negative average effect of repeated marketing calls.
CATE highlights that the effect differs between customer groups.
These results can help improve targeted marketing decisions.
