# %% [markdown]
# ## Data preprocessing and data processing
# %% [markdown]
# ## 1. Dealing with missing data

# %%
from cmath import nan
from more_itertools import last
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

string_data = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])
print(string_data)
print(string_data.isnull())

string_data[0] = None
print(string_data.isnull())
# %% [markdown]
# dropna is to drop the nan item\
# fillna is to fill space with nan
# %%
from numpy import nan as NA
data = pd.Series([1, NA, 3.5, NA, 7])
print(data)
print(data.dropna())
print(data[data.notnull()])
data2 = pd.DataFrame([[1., 6.5, 3.], [1., NA, NA], [NA, NA, NA], [NA, 6.5, 3.]])
cleaned_data2 = data2.dropna() ## it will drop out any rows with at least a NA
print(f'data2\n', data2)
print(f'cleaned_data2\n', cleaned_data2)
other_cleaned = data2.dropna(how = 'all') ## it will drop out rows that with full NA
print(other_cleaned)
# %% [markdown]
# we can use dropna(how = 'all', axis = 1) to drop out columns

# %%
df = pd.DataFrame(np.random.randn(7, 3))
df.iloc[:4, 1] = NA
df.iloc[:2, 2] = NA
print(df)
print(df.dropna())
print(df.dropna(thresh = 2)) 
# %%
print(df.fillna(0)) ## fill NA with 0
print(df.fillna({1: 0.5, 2: 1.0})) ## fill NA with different columns by using the format of dict
# %% [markdown]
# ## Data transfer

# %%
data3 = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'], 'k2': [1, 1, 2, 3, 3, 4, 4]})
print(data3)
print(data3.duplicated()) ## duplicated will return bollean value that represent the repeat value
print(data.drop_duplicates())
data3['v1'] = range(7)
print(data3.drop_duplicates(['k1'])) ## 對 k1 的重複值做過濾

# transforming data using a function or mapping
food_data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon',
                              'Pastrami', 'corned beef', 'Bacon',
                              'pastrami', 'honey ham', 'nova lox'],
                     'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
print(food_data)

meat_to_animal = {
  'bacon': 'pig',
  'pulled pork': 'pig',
  'pastrami': 'cow',
  'corned beef': 'cow',
  'honey ham': 'pig',
  'nova lox': 'salmon'
}

lowercased = food_data['food'].str.lower() ## str.lower() change to 小寫
print(lowercased)
food_data['animal'] = lowercased.map(meat_to_animal)
print(food_data)

# Replacing Values
data4 = pd.Series([1., -999., 2., -999., -1000., 3.])
print(data4)
print(data4.replace(-999, np.nan))
print(data4.replace([-999, -1000], np.nan))
print(data4.replace([-999, -1000], [np.nan, 0]))
print(data4.replace({-999: np.nan, -1000: 0}))

# Renaming Axis Indexes
data5 = pd.DataFrame(np.arange(12).reshape((3, 4)),
                    index=['Ohio', 'Colorado', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
print(data5)

transform = lambda x: x[:4].upper() ## 把 x 的前四個字母轉換為大寫，並捨棄其他
print(data5.index.map(transform))

data5.index = data5.index.map(transform)
print(data5)
print(data5.rename(index=str.title, columns=str.upper))
data5.rename(index={'OHIO': 'INDIANA'},
            columns={'three': 'THREE'})
data5.rename(index={'OHIO': 'INDIANA'}, inplace=True)
print(data5)
# %%
