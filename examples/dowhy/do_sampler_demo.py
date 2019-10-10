# This code was taken from this page:
# https://gist.github.com/akelleh/2a741a57f0a6f75262146ab17b2a6ef3
# for the purpose of testing its example on
# do sampler from DoWhy

import numpy as np
import pandas as pd
import dowhy.api
import seaborn as sns
import matplotlib.pyplot as pp

sns.set(rc={'figure.figsize': (9, 7)})

# Generating Data
# Here, there is no causal relationship between title length and click-through rate.
# For each population, the click-through rate is independent of title length.

N = 100000
n = 30000

# normally, ctr is independent of title_length
title_length = np.random.choice(range(25), size=N) + 1
click_through_rate = np.random.beta(5, 100, size=N)

# but one quirky person prefers titles in a much narrower range, and is especially talented
title_length_2 = np.random.normal(13, 3, size=n).astype(int)
click_through_rate_2 = np.random.beta(10, 100, size=n)

# let's concatenate these together
all_title_lengths = np.array(list(title_length) + list(title_length_2))
all_click_rates = np.array(list(click_through_rate) + list(click_through_rate_2))

df = pd.DataFrame({'click_through_rate': all_click_rates,
                   'title_length': all_title_lengths,
                   'author': [1] * N + [0] * n})

# restrict down to where there's enough data
df = df[df.title_length > 0][df.title_length < 25]

sns.lineplot(data=df, x='title_length', y='click_through_rate')

pp.ylim(0, 0.1)

pp.show(sns)

# You can see now the click-through rate depends on the title length! From the data-generating process,
# we know the dependence isn't causal: it was introduced because one quirky author happened to like 12-16 word titles,
# and happened to be really good at writing titles. If we were to control for author, this effect would go away,
# and we'd recover the underlying causal relationship! Let's use the do-sampler to do the adjustment.

causal_df = df.causal.do('title_length',
                         method='weighting',
                         variable_types={'title_length': 'd',
                                         'click_through_rate': 'c',
                                         'author': 'd'},
                         outcome='click_through_rate',
                         common_causes=['author'])

sns.lineplot(data=causal_df, x='title_length', y='click_through_rate');
pp.ylim(0, 0.1)

pp.show(sns)

# So we see the bump in the middle goes away! We've recovered the true, underlying causal effect
# after removing the confounding by our author's title length preference.
# Go back to the article for more explanation!
