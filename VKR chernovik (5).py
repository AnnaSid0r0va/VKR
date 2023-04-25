#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


df1=pd.read_excel('X_bp.xlsx')
df2=pd.read_excel('X_nup.xlsx')


# In[3]:


df1.shape


# In[4]:


df1.head(3)


# In[5]:


df2.shape


# In[6]:


df2.head(3)


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression


# In[8]:


fig, axs = plt.subplots(10, figsize = (10,10))
plt1 = sns.boxplot(df1['Соотношение матрица-наполнитель'], ax = axs[0])
plt2 = sns.boxplot(df1['Плотность, кг/м3'], ax = axs[1])
plt3 = sns.boxplot(df1['модуль упругости, ГПа'], ax = axs[2])
plt4 = sns.boxplot(df1['Количество отвердителя, м.%'], ax = axs[3])
plt5 = sns.boxplot(df1['Содержание эпоксидных групп,%_2'], ax = axs[4])
plt6 = sns.boxplot(df1['Температура вспышки, С_2'], ax = axs[5])
plt7 = sns.boxplot(df1['Поверхностная плотность, г/м2'], ax = axs[6])
plt8 = sns.boxplot(df1['Модуль упругости при растяжении, ГПа'], ax = axs[7])
plt9 = sns.boxplot(df1['Прочность при растяжении, МПа'], ax = axs[8])
plt10 = sns.boxplot(df1['Потребление смолы, г/м2'], ax = axs[9])
plt.tight_layout()


# In[9]:


fig, axs = plt.subplots(3, figsize = (10,3))
plt1 = sns.boxplot(df2['Угол нашивки, град'], ax = axs[0])
plt2 = sns.boxplot(df2['Шаг нашивки'], ax = axs[1])
plt3 = sns.boxplot(df2['Плотность нашивки'], ax = axs[2])
plt.tight_layout()


# In[10]:


df = pd.merge(df1, df2, how="inner")
df.drop(['Unnamed: 0','Температура вспышки, С_2', 'модуль упругости, ГПа', 'Содержание эпоксидных групп,%_2', 'Угол нашивки, град', 'Поверхностная плотность, г/м2'], axis=1, inplace=True)
df.head(3)


# In[11]:


sns.heatmap(df.corr(), annot=True)


# In[12]:


sns.pairplot(df.head(100), diag_kind='kde', palette='cbar')


# In[13]:


df.isnull().sum()


# In[14]:


df.info()


# In[15]:


X = df.copy()
X


# In[16]:


from sklearn.decomposition import PCA
features = [
    'Плотность, кг/м3',
    'Количество отвердителя, м.%',
    'Модуль упругости при растяжении, ГПа',
    "Прочность при растяжении, МПа",
    'Потребление смолы, г/м2',
    'Шаг нашивки',
    'Плотность нашивки'
]

X = df.copy()
y = X.pop("Соотношение матрица-наполнитель")
X = X.loc[:, features]

# Стандартизация
X_stand = (X - X.mean(axis=0)) / X.std(axis=0)

pca = PCA()
X_pca = pca.fit_transform(X_stand)

# Переводим результат в Pandas Dataframe
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

# В loading записываем "смысловую нагрузку" компонентa, их корреляцию с исходными
loadings = pd.DataFrame(
    pca.components_.T,
    columns=component_names,
    index=X.columns,
)

print(loadings)


# In[17]:


fig, axs = plt.subplots(1, 2)
n = pca.n_components_
grid = np.arange(1, n + 1)

evr = pca.explained_variance_ratio_
axs[0].bar(grid, evr)
axs[0].set(xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0))

cv = np.cumsum(evr)
axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
axs[1].set(xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0))

fig.set(figwidth=8, dpi=100)


# In[18]:


sns.histplot(df)


# In[19]:


X = df.copy()
X_stand = (X - X.mean(axis=0)) / X.std(axis=0)
sns.histplot(X_stand)


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X_stand, y, train_size=0.7, test_size=0.3, random_state=0)
X_train.head()


# In[21]:


y_train.head()


# In[22]:


X_stand.describe().transpose()[['mean', 'std']]


# In[34]:


#пробую провести линейную регрессию

from sklearn.metrics import mean_squared_error 

features = [
    'Соотношение матрица-наполнитель',
    'Плотность, кг/м3',
    'Количество отвердителя, м.%',
    'Модуль упругости при растяжении, ГПа',
    'Потребление смолы, г/м2',
    'Шаг нашивки',
    'Плотность нашивки'
]

y = X_stand['Прочность при растяжении, МПа']
X = X_stand[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)
model=LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)
test_data_rmse = mean_squared_error(y_test, y_pred, squared=False)
training_data_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
print(f"Control RMSE = {test_data_rmse:4f}")
print(f"Train RMSE = {training_data_rmse:4f}")


# In[35]:


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


# In[37]:


cv_scores=cross_val_score(
        model,
        X_train, y_train,
        cv=5, scoring="neg_root_mean_squared_error"
    )
print("Results k-fold:")
print("\n".join(f"RSME={s:.4f}" for s in cv_scores))
print(f"Mean RMSE = {np.mean(-cv_scores):.4f}")


# In[72]:


X = df.drop(['Прочность при растяжении, МПа'], axis=1)
X


# In[45]:


from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils


# In[ ]:




