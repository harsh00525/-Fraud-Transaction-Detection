# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# %%
train_df = pd.read_csv("./Preprocessed.csv")
test_df = pd.read_csv("./Test_Preprocessed.csv")

# %%
test_df.drop('Unnamed: 0', inplace=True, axis =1)

# %%
train_df.drop('Unnamed: 0', inplace=True, axis =1)

# %%
Ytrain = train_df['isFraud']

# %%
train_df.drop('isFraud', inplace=True, axis =1)

# %%
tf.random.set_seed(42)

# %%
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=len(train_df.axes[1])))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %%
model.fit(train_df, Ytrain, epochs=100, verbose=0)

# %%
ypred = model.predict(test_df)

# %%
CSV4 = pd.DataFrame(ypred)
file = CSV4.to_csv("PredictionsNN.csv")


