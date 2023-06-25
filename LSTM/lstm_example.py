# Import necessary libraries
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# Define the sequence
sequence = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# Split the sequence into input/output samples
X = []
y = []
n_time_steps = 3

for i in range(len(sequence)):
    end_ix = i + n_time_steps
    if end_ix > len(sequence)-1:
        break
    seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
    X.append(seq_x)
    y.append(seq_y)

X = np.array(X)
y = np.array(y)

# Reshape input into [samples, time_steps, features] for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Define LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_time_steps, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
# Fit model
model.fit(X, y, epochs=300, verbose=0)


# Predict
x_input = np.array([0.7, 0.8, 0.9]).reshape((1, n_time_steps, 1))
yhat = model.predict(x_input, verbose=0)
