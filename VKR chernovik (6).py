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


# In[23]:


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
model1=LinearRegression()
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
y_train_pred = model1.predict(X_train)
test_data_rmse = mean_squared_error(y_test, y_pred, squared=False)
training_data_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
print(f"Control RMSE = {test_data_rmse:4f}")
print(f"Train RMSE = {training_data_rmse:4f}")


# In[24]:


plt.scatter(y_test, y_pred)
plt.plot([0, y.max()], [0, y.max()], '--', color='gray') 
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Predicted vs Actual values')
plt.show()


# In[25]:


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


# In[26]:


cv_scores=cross_val_score(
        model1,
        X_train, y_train,
        cv=5, scoring="neg_root_mean_squared_error"
    )
print("Results k-fold:")
print("\n".join(f"RSME={s:.4f}" for s in cv_scores))
print(f"Mean RMSE = {np.mean(-cv_scores):.4f}")


# In[27]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# In[29]:


df = df.copy()

X = df.drop(columns=["Прочность при растяжении, МПа"])
y = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model2 = Sequential()
model2.add(Dense(128, input_dim=X_train.shape[1], activation="relu"))
model2.add(Dense(128, activation="relu"))
model2.add(Dense(8, activation="softmax"))

model2.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

model2.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

y_pred = model2.predict(X_test)


# In[30]:


print(model2.summary())


# In[31]:


# Model 1 (Linear Regression)
plt.scatter(y_test, y_pred)
plt.plot(y_test, y_test, color="red")
plt.title("Model 1: Linear Regression")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()

# Model 2 (Neural Network)
y_pred_nn = np.argmax(y_pred, axis=1)
y_test_nn = np.argmax(y_test.to_numpy(), axis=1) # convert to numpy array
plt.scatter(y_test_nn, y_pred_nn)
plt.title("Model 2: Neural Network")
plt.plot(y_test_nn, y_test_nn, color="red")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()


# In[32]:


from sklearn.linear_model import ElasticNet

df = df.copy()

X = df.drop(columns=["Прочность при растяжении, МПа"])
y = df["Прочность при растяжении, МПа"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"RMSE = {rmse:.4f}")

plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red")
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Elastic Net Regression")
plt.show()


# In[33]:


from sklearn.tree import DecisionTreeRegressor

# Initialize the model
dt_reg = DecisionTreeRegressor(random_state=42)

# Train the model
dt_reg.fit(X_train, y_train)

# Predict on the test set
y_pred_dt = dt_reg.predict(X_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score

mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print("Decision Tree Regressor:\nMSE = {:.2f}\nR2 Score = {:.2f}".format(mse_dt, r2_dt))


# In[34]:


plt.scatter(y_test, y_pred_dt)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red")
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Decision Tree Regression")
plt.show()


# In[35]:


from sklearn.ensemble import GradientBoostingRegressor

X = df.drop("Прочность при растяжении, МПа", axis=1)
y = df["Прочность при растяжении, МПа"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

gbm.fit(X_train, y_train)

y_pred = gbm.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[36]:


plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Gradient Boosting Regressor')
plt.show()


# In[37]:


get_ipython().system('pip install pydot')


# In[39]:


import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

# Load the data
data = df
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data into training and testing sets
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# Preprocess the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data for RNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the RNN model
model = tf.keras.Sequential([
    layers.LSTM(64, input_shape=(X_train.shape[1], 1)),
    layers.Dense(1)
])

# Compile the model
model.compile(loss="mean_squared_error", optimizer="adam")

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


# In[42]:


from tensorflow.keras.layers import SimpleRNN

# Load the data
df = df.copy()

# Load the data
data = df

# Set the target variable and predictors
target_var = 'Прочность при растяжении, МПа'
predictors = list(set(data.columns) - set([target_var]))

# Split the data into train and test sets
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Define the function to create the input/output sequences
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, :])
        y.append(data[i, 4])
    X, y = np.array(X), np.array(y)
    return X, y

# Define the sequence length
seq_length = 5

# Create the train and test sequences
X_train, y_train = create_sequences(train_data_scaled, seq_length)
X_test, y_test = create_sequences(test_data_scaled, seq_length)

# Build the RNN model
model = Sequential()
model.add(SimpleRNN(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(SimpleRNN(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Visualize the model
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Make predictions
y_pred = model.predict(X_test)

# Inverse the scaling
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred)

# Calculate the root mean squared error
rmse = np.sqrt(np.mean(((y_pred_inv - y_test_inv) ** 2)))
print('RMSE:', rmse)


# In[43]:


# Load the data
data = df

# Set the target variable and predictors
target_var = 'модуль упругости, ГПа'
predictors = list(set(data.columns) - set([target_var]))

# Split the data into train and test sets
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Define the function to create the input/output sequences
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, :])
        y.append(data[i, 4])
    X, y = np.array(X), np.array(y)
    return X, y

# Define the sequence length
seq_length = 5

# Create the train and test sequences
X_train, y_train = create_sequences(train_data_scaled, seq_length)
X_test, y_test = create_sequences(test_data_scaled, seq_length)

# Build the RNN model
model = Sequential()
model.add(SimpleRNN(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(SimpleRNN(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Visualize the model
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Make predictions
y_pred = model.predict(X_test)

# Inverse the scaling
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred)

# Calculate the root mean squared error
rmse = np.sqrt(np.mean(((y_pred_inv - y_test_inv) ** 2)))
print('RMSE:', rmse)


# In[44]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Make predictions on test set
y_pred = model.predict(X_test)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('MSE: {:.3f}'.format(mse))
print('RMSE: {:.3f}'.format(rmse))
print('MAE: {:.3f}'.format(mae))
print('R^2: {:.3f}'.format(r2))


# In[ ]:




