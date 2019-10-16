# This code was taken from this page:
# https://github.com/microsoft/dowhy/tree/master/docs/source
# for the purpose of testing its example on
# do sampler from DoWhy

import numpy as np
import pandas as pd

import dowhy
# from dowhy import CausalModel
import dowhy.datasets

# In attempting to estimate the causal effect of some variable X on another Y,
# an instrument is a third variable Z which affects Y only through its effect on X
from dowhy.do_why import CausalModel

data = dowhy.datasets.linear_dataset(beta=10,
                                     num_common_causes=5,
                                     num_instruments=2,
                                     num_samples=10000,
                                     treatment_is_binary=True)
df = data["df"]
print(df.head())
print(data["dot_graph"])
print("\n")
print(data["gml_graph"])

# With graph
model = CausalModel(
    data=df,
    treatment=data["treatment_name"],
    outcome=data["outcome_name"],
    graph=data["gml_graph"]
)

model.view_model()

from IPython.display import Image, display

display(Image(filename="causal_model.png"))

# DoWhy philosophy: Keep identification and estimation separate
# Identification can be achieved without access to the data, acccesing only the graph.
# This results in an expression to be computed.
# This expression can then be evaluated using the available data in the estimation step.
# It is important to understand that these are orthogonal steps.

identified_estimand = model.identify_effect()
print(identified_estimand)

causal_estimate = model.estimate_effect(identified_estimand,
                                        method_name="backdoor.propensity_score_stratification")
print(causal_estimate)
print("Causal Estimate is " + str(causal_estimate.value))

# Refuting the estimate follows in the Jupyter Notebook
