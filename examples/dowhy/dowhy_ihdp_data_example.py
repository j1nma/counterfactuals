# This code was taken from this page:
# https://hub.gke.mybinder.org/user/microsoft-dowhy-az0mj00z/tree/docs/source/example_notebooks
# for the purpose of testing its example on
# do sampler from DoWhy

# importing required libraries
import os, sys

from dowhy.do_why import CausalModel

sys.path.append(os.path.abspath("../../"))
import dowhy
# from dowhy import CausalModel
import pandas as pd
import numpy as np

# Loading Data
data = pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv",
                   header=None)
col = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1", ]

for i in range(1, 26):
    col.append("x" + str(i))
data.columns = col
data.head()
print(data)

# Model
# Create a causal model from the data and given common causes.
xs = ""
for i in range(1, 26):
    xs += ("x" + str(i) + "+")

model = CausalModel(
    data=data,
    treatment='treatment',
    outcome='y_factual',
    common_causes=xs.split('+')
)

# Identify
# Identify the causal effect
identified_estimand = model.identify_effect()

# Estimate (using different methods)

# 3.1 Using Linear Regression
# Estimate the causal effect and compare it with Average Treatment Effect
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.linear_regression", test_significance=True
                                 )

print(estimate)

print("Causal Estimate is " + str(estimate.value))
data_1 = data[data["treatment"] == 1]
data_0 = data[data["treatment"] == 0]

print("ATE", np.mean(data_1["y_factual"]) - np.mean(data_0["y_factual"]))

# 3.2 Using Propensity Score Matching
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.propensity_score_matching"
                                 )

print("Causal Estimate is " + str(estimate.value))

print("ATE", np.mean(data_1["y_factual"]) - np.mean(data_0["y_factual"]))

# 3.3 Using Propensity Score Stratification
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.propensity_score_stratification",
                                 method_params={'num_strata': 50, 'clipping_threshold': 5}
                                 )

print("Causal Estimate is " + str(estimate.value))
print("ATE", np.mean(data_1["y_factual"]) - np.mean(data_0["y_factual"]))

# 3.4 Using Propensity Score Weighting
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.propensity_score_weighting"
                                 )

print("Causal Estimate is " + str(estimate.value))

print("ATE", np.mean(data_1["y_factual"]) - np.mean(data_0["y_factual"]))

# 4. Refute
# Refute the obtained estimate using multiple robustness checks.
# 4.1 Adding a random common cause

refute_results = model.refute_estimate(identified_estimand, estimate,
                                       method_name="random_common_cause")
print(refute_results)

# 4.2 Using a placebo treatment
res_placebo = model.refute_estimate(identified_estimand, estimate,
                                    method_name="placebo_treatment_refuter", placebo_type="permute")
print(res_placebo)

# 4.3 Data Subset Refuter

res_subset = model.refute_estimate(identified_estimand, estimate,
                                   method_name="data_subset_refuter", subset_fraction=0.9)
print(res_subset)
