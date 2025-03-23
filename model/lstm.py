#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Bidirectional
from pathlib import Path
#%%
# parameters
dense = 64
dropout = 0.3
ep = 60
seq_length = 3
batch_size = 32
units =128
target_column = 'Close'
predictors = ['Date','polarity','Close']

#%%
# prepare the data
root = Path(__file__).resolve().parent.parent
data_path = root / 'data' / 'df.csv'
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df[predictors]
df.set_index('Date',inplace=True)
numeric_cols = df.select_dtypes(include=[np.number]).columns
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[numeric_cols])
df_scaled = pd.DataFrame(df_scaled, columns=numeric_cols,index=df.index)

def create_sequences(data, target_column, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i + seq_length].values)
        y.append(data.iloc[i + seq_length][target_column])  
    return np.array(X), np.array(y)
X, y = create_sequences(df_scaled, target_column, seq_length)
split = int(len(X) * 0.7)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

#%%
# Construct LSTM model
model = Sequential([
    Bidirectional(LSTM(units, return_sequences=True, input_shape=(seq_length, X.shape[2]))),
    Dropout(dropout),
    LSTM(units, return_sequences=False),
    Dropout(dropout),
    Dense(dense, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=ep, batch_size=batch_size, validation_data=(X_test, y_test))
# Loss curve plotting
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()
plt.show()
# fit the model
model.fit(X_train, y_train, epochs=ep, batch_size=batch_size, validation_data=(X_test, y_test))
# predict
y_pred_lstm = model.predict(X_test)
train_predict = model.predict(X_train)
# inverse normalization
scaler = MinMaxScaler()
scaler.fit(df[[target_column]])
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(y_pred_lstm)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 
r2_lstm = r2_score(original_ytest, test_predict) 
print(f"LSTM R² Score: {r2_lstm:.4f}")

# MSE and MAE
mse_lstm = mean_squared_error(original_ytest, test_predict)
mae_lstm = mean_absolute_error(original_ytest, test_predict)
print(f'LSTM MSE: {mse_lstm}, MAE: {mae_lstm}')

#%%
# Plotting
plt.figure(figsize=(12, 7))
plt.style.use('ggplot')  
plt.plot(df_scaled.index[split + seq_length:], original_ytest, 
         label='Actual Close', color='blue', linewidth=3.5)
plt.plot(df_scaled.index[split + seq_length:], test_predict, 
         label='Predicted Close', linestyle='--', color='orange', linewidth=3.5)
plt.title('LSTM Prediction vs Actual Close', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close Price', fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.legend(fontsize=12, loc='upper left', frameon=True, fancybox=True)
plt.tight_layout()
plt.show()