# This code was taken from this page:
# https://hub.gke.mybinder.org/user/microsoft-dowhy-az0mj00z/tree/docs/source/example_notebooks
# for the purpose of testing its example on
# do sampler from DoWhy
import os, sys
import random

sys.path.append(os.path.abspath("../../"))

import numpy as np
import pandas as pd

import dowhy
# from dowhy import CausalModel
from IPython.display import Image, display

# I. Generating dummy data
# We generate some dummy data for three variables: X, Y and Z.
from dowhy.do_why import CausalModel

z = [i for i in range(10)]
random.shuffle(z)
df = pd.DataFrame(data={'Z': z, 'X': range(0, 10), 'Y': range(0, 100, 10)})
print(df)

# II. Loading GML or DOT graphs
# GML format

# With GML string
model = CausalModel(
    data=df,
    treatment='X',
    outcome='Y',
    graph="""graph[directed 1 node[id "Z" label "Z"]  
                    node[id "X" label "X"]
                    node[id "Y" label "Y"]      
                    edge[source "Z" target "X"]    
                    edge[source "Z" target "Y"]     
                    edge[source "X" target "Y"]]"""

)
model.view_model()

display(Image(filename="causal_model_simple_example.png"))
