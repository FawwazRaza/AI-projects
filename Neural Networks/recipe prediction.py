
pip install tensorflow keras matplotlib numpy

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('recipes1.csv')

x = data.iloc[:, 2:6].values
y = data.iloc[:, 1].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.1, random_state=42)

actual_recipe_names = label_encoder.inverse_transform(y_test)

# for recipe_name in actual_recipe_names:
#     print(recipe_name)

import numpy as np
from sklearn.metrics import mean_squared_error

def relu(inp):
    return np.maximum(0, inp)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(inp):
    exp_x = np.exp(inp)
    return exp_x / np.sum(exp_x)

def weights():
    w_0 = {
        'node 0': np.array([2, 1, 2.5, 1]),
        'node 1': np.array([2, 1, 2.5, 1]),
        'node 2': np.array([2, 1, 2.5, 1])
    }
    w_1 = {
        'node 0': np.array([2, 1, 2.5])
    }
    return w_0, w_1


def forward(x, weights_0, weights_1):
    if len(x.shape) == 1:
        x = x.reshape(1, -1)

    h1 = np.dot(x, weights_0['node 0'])
    h1_out = relu(h1)
    h2 = np.dot(x, weights_0['node 1'])
    h2_out = relu(h2)
    h3 = np.dot(x, weights_0['node 2'])
    h3_out = relu(h3)

    hidden_layer = np.vstack([h1_out, h2_out, h3_out]).T
    output = np.dot(hidden_layer, weights_1['node 0'])

    return output, hidden_layer

def loss_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def backpropagation(x, y, output, hidden_layer, weights_0, weights_1):
    if len(x.shape) == 1:
        x = x.reshape(1, -1)

    output_error = 2 * (output - y)
    hidden_error = np.outer(output_error, weights_1['node 0'])

    w1_updates = {'node 0': np.dot(hidden_layer.T, output_error) / len(x)}

    w0_updates = {}

    for i in range(3):
        hidden_input = np.dot(x, weights_0[f'node {i}'])
        relu_deriv = relu_derivative(hidden_input)
        hidden_error_for_node = hidden_error[:, i] * relu_deriv
        w0_updates[f'node {i}'] = np.dot(x.T, hidden_error_for_node) / len(x)

    return w0_updates, w1_updates


def learning(learn_rate, x, y, weights_0, weights_1):
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    output, hidden_layer = forward(x, weights_0, weights_1)
    w0_updates, w1_updates = backpropagation(x, y, output, hidden_layer, weights_0, weights_1)

    weights_0_updated = {
        'node 0': weights_0['node 0'] - learn_rate * w0_updates['node 0'],
        'node 1': weights_0['node 1'] - learn_rate * w0_updates['node 1'],
        'node 2': weights_0['node 2'] - learn_rate * w0_updates['node 2']
    }

    weights_1_updated = {
        'node 0': weights_1['node 0'] - learn_rate * w1_updates['node 0']
    }

    return weights_0_updated, weights_1_updated

weight_0, weight_1 = weights()

out_1 = []
for i in x_train:
    output, _ = forward(i, weight_0, weight_1)
    out_1.append(output)

train_error = loss_error(y_train, out_1)


learn_rate = 0.01
weight_0, weight_1 = learning(learn_rate, x_train, y_train, weight_0, weight_1)

out_1 = []
for i in x_train:
    output, _ = forward(i, weight_0, weight_1)
    out_1.append(output)

out_1 = np.array(out_1).reshape(-1)

train_error = loss_error(y_train, out_1)

pip install tensorflow
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error

n_cols = x_train.shape[1]


model = Sequential()
model.add(Dense(101, activation='relu', input_shape=(n_cols,)))
model.add(Dense(54, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1)

ts = model.predict(x_test)

r_values = ts.round()
int_values = r_values.astype(int)
flat_values = int_values.flatten()
predicted_recipe = label_encoder.inverse_transform(flat_values)
print("predicted recipe names:")
for rn in predicted_recipe:
    print(rn)



