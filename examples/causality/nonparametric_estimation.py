# This code was taken from this page:
# https://github.com/akelleh/causality
# for the purpose of testing its example on
# Nonparametric Effects Estimation
import numpy
import pandas as pd
from causality.estimation.adjustments import AdjustForDirectCauses
from causality.estimation.nonparametric import CausalEffect
from networkx import DiGraph

g = DiGraph()

g.add_nodes_from(['x1', 'x2', 'x3', 'x4', 'x5'])
g.add_edges_from([('x1', 'x2'), ('x1', 'x3'), ('x2', 'x4'), ('x3', 'x4')])
adjustment = AdjustForDirectCauses()

admissable_set = adjustment.admissable_set(g, ['x2'], ['x3'])

print(admissable_set)

# generate some toy data:
SIZE = 2000
x1 = numpy.random.normal(size=SIZE)
x2 = x1 + numpy.random.normal(size=SIZE)
x3 = x1 + numpy.random.normal(size=SIZE)
x4 = x2 + x3 + numpy.random.normal(size=SIZE)
x5 = x4 + numpy.random.normal(size=SIZE)

# load the data into a dataframe:
X = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5})

# define the variable types: 'c' is 'continuous'.  The variables defined here
# are the ones the search is performed over  -- NOT all the variables defined
# in the data frame.
variable_types = {'x1': 'c', 'x2': 'c', 'x3': 'c', 'x4': 'c', 'x5': 'c'}

effect = CausalEffect(X, ['x2'], ['x3'], variable_types=variable_types, admissable_set=list(admissable_set))

x = pd.DataFrame({'x2': [0.], 'x3': [0.]})

print(effect.pdf(x))
