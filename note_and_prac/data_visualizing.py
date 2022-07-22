# %% [markdown]
# ## Data Visualizing with matplotlib and seaborn

# %% [markdown]
# Figures and subplots

# %%
from cProfile import label
from matplotlib.lines import _LineStyle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series, DataFrame

fig = plt.figure(figsize = (18, 12))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
plt.plot(np.random.randn(50).cumsum(), 'k--')
_ = ax1.hist(np.random.randn(100), bins = 20, color= 'k', alpha = 0.3)
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
# %% [markdown]
# construct subplots

# %%
fig, axes = plt.subplots(2, 3)
axes
# %% [markdown]
# adjusting the spacing arond subplots

# %%
fig = plt.figure(figsize = (18, 12))
fig, axes =  plt.subplots(2, 2, sharex = True, sharey = True)
for i in range(2):
    for j in range(2):
        axes[i, j].hist(np.random.randn(500), bins = 50, color = 'k', alpha = 0.5)
plt.subplots_adjust(wspace = 0, hspace = 0)
# %% [markdown]
# colors , markers and line style

# %%
plt.plot(np.random.randn(30).cumsum(), 'o--', color = 'g')

# %%
data = np.random.randn(30).cumsum()
plt.plot(data, '--', color = 'g', label = 'Default')
plt.plot(data, '-', color = 'blue', drawstyle = 'steps-post', label = 'steps-post')
plt.legend(loc = 'best')
# %%
