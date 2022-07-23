# %% [markdown]
# ## 2D plotting

# %% [markdown]
# import packages

# %%
from matplotlib import projections
from more_itertools import sample
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# %%
np.random.seed(123)
y = np.random.randn(500)
x  = range(len(y))
plt.style.use('seaborn-darkgrid')
fig = plt.figure(figsize = (24, 8))
plot1 = fig.add_subplot(1, 2, 1)
plot2 = fig.add_subplot(1, 2, 2)
plt.plot(x, y.cumsum())
_ = plot1.plot(x, y)


# %% [markdown]
# plot with typical labels

# %%
np.random.seed(123)
b = np.random.randn(20)
a = range(len(b))
plt.figure(figsize = (12, 8))
plt.plot(a, b.cumsum(), color = 'blue', lw = 1.5)
plt.plot(a, b.cumsum(), 'ro')
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
# %% [markdown]
# two dimensional datasets

# %%
np.random.seed(123)
y = np.random.standard_normal(size = (20, 2)).cumsum(axis = 0)
plt.style.use('seaborn-darkgrid')
plt.figure(figsize = (12, 8))
plt.plot(y[:, 0], lw = 1.5, label = '1st')
plt.plot(y[:, 1], lw = 1.5, label = '2nd')
plt.plot(y, 'ro')
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.legend(loc = 0)

# %% [markdown]
# Plot the polygon 

# %%
from matplotlib.patches import Polygon
def function(x):
    return 0.5 * np.exp(x) + 1

a, b = 0.5, 1.5
x = np.linspace(0, 2)
y = function(x)

fig, ax = plt.subplots(figsize = (12, 8))
plt.plot(x, y, color = 'blue', lw = 2)
plt.ylim(ymin = 0)

sample_x = np.linspace(a, b)
sample_y = function(sample_x)
vertical = [(a, 0)] + list(zip(sample_x, sample_y)) + [(b, 0)]  ## gray sector
polygon = Polygon(vertical, facecolor = '0.7', edgecolor = '0.5') ## gray sector
ax.add_patch(polygon)

plt.text(0.5 * (a + b), 1, r'$\int_a^b f(x)dx $', horizontalalignment = 'center', fontsize = 20)
plt.figtext(0.9, 0.075, '$x$')
plt.figtext(0.075, 0.9, '$f(x)$')
ax.set_xticks((a, b))
ax.set_xticklabels(('$a$', '$b$'))
ax.set_yticks([function(a), function(b)])
ax.set_yticklabels(('$f(a)$', '$f(b)$'))
plt.grid(True)

# %% [markdown]
# financial data visualizing

# %% 
import mplfinance as mpf
import yfinance as yf

dataframe = yf.download('2330.TW', start = '2022-01-01', end = '2022-07-01')
mc = mpf.make_marketcolors(up = 'red', down = 'green', edge = '', wick = 'inherit', volume = 'inherit')
s = mpf.make_mpf_style(base_mpf_style = 'yahoo', marketcolors = mc)
mpf.plot(dataframe, type = 'candle', style = s, volume = True)

# %% [markdown]
# three dimsenional datasets
# we consider strike values between 50 and 150, times to maturity between 0.5 and 2.5 years
# %%
strike = np.linspace(50, 150, 24)
times_to_maturity = np.linspace(0.5, 2.5, 24)
strike, times_to_maturity = np.meshgrid(strike, times_to_maturity)
implied_volatility = (strike - 100) ** 2 / (100 * strike) / times_to_maturity
from mpl_toolkits.mplot3d import Axes3D
figure = plt.figure(figsize = (9, 6))
ax = figure.gca(projection = '3d')

surf = ax.plot_surface(strike, times_to_maturity, implied_volatility, rstride = 2, cstride = 2, cmap = plt.cm.coolwarm, linewidth = 0.5, antialiased = True)
ax.set_xlabel('strike')
ax.set_ylabel('times - to - matirity')
ax.set_zlabel('implied volatility')

figure.colorbar(surf, shrink = 0.5, aspect = 5)

