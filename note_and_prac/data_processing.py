# %% [markdown]
# ## Data preprocessing and data processing
# %% [markdown]
# ## 1. Dealing with missing data

# %%
from cmath import nan
import py_compile
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

# Discretization and Binning
# When analyzing data, we should deicretization the data or put them into different groups.
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]

# We want to hve groups of 18-25, 26-35, 36-60 and over 61
bins = [18, 25, 35, 60, 100] # (18, 25], (25, 35], ...
cats = pd.cut(ages, bins)
print(cats)
print(cats.categories)
print('amount of every groups\n', pd.value_counts(cats))

# Customize lables
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, labels=group_names)

# What if we want to divide data by the amount of data
data6 = np.random.rand(20) # uniform (0,1)
pd.cut(data, 4, precision = 2) # Divide data into 4 parts, precision = 2 means 小數點第二位

# quartile cut (qcut)
data7 = np.random.randn(1000)
cats2 = pd.qcut(data7, 4)
print(cats2)

# %% [markdown]
# ## Find outliers, sample, dummies

# %%
# find and filt outliers
data8 = pd.DataFrame(np.random.randn(1000, 4))
print(f'statistical analys\n', data8.describe())
column = data8[2]
column[np.abs(column) > 3] # find outliers
data8[(np.abs(data8) > 3).any(1)] # find every outliers

# We can also make the data are always between -3 and 3
data8[np.abs(data8) > 3] = np.sign(data8) * 3 # np.sign() will generate 1 or -1
print(data8.describe())

# Permutation and random sampling
df2 = pd.DataFrame(np.arange(20).reshape((5, 4)))
sampler = np.random.permutation(5) # np.random.permutation(length of axis)
print(sampler)
print(df2.take(sampler))

# Dummy Variables
df3 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                   'data9': range(6)})
pd.get_dummies(df3['key'])

dummies = pd.get_dummies(df3['key'], prefix = 'key') # prefix is 字首
df_with_dummy = df3[['data9']].join(dummies)
print(df_with_dummy)

movie_names = ['movie_id', 'title', 'genres']
movies = pd.read_table('/Users/abnerteng/GitHub/pydata-book/datasets/movielens/movies.dat', sep = '::',
                       header = None, names = movie_names)
print(movies[:10])

# pick out genres
all_genres = []
for x in movies.genres:
  all_genres.extend(x.split('|'))
genres = pd.unique(all_genres)
print(genres)

# one method to construct a DataFrame is to fill the 0-matrix
zero_matrix = np.zeros((len(movies), len(genres)))
dummies = pd.DataFrame(zero_matrix, columns = genres)

#classify index value
gen = movies.genres[0]
gen.split('|')
dummies.columns.get_indexer(gen.split('|'))

for i, gen in enumerate(movies.genres):
  indices = dummies.columns.get_indexer(gen.split('|'))
  dummies.iloc[i, indices] = 1

movies_windic = movies.join(dummies.add_prefix('Genre_'))
print(movies_windic.iloc[0])

# applied in statistic
np.random.seed(123)
values = np.random.rand(100)
print(values)
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
print(pd.get_dummies(pd.cut(values, bins)))


# %%
