# basic training loop for different models

import tensorflow as tf
import numpy as np


CHAMP_NUM = 170
PLAYER_NUM = 10

# load data
DATAPATH = "data/kr_24h/game_data.npy"
data = np.load(DATAPATH)
print("Loaded data of shape: ", data.shape)

x = data[:,:-1]
y = data[:,-1]
# convert y to float
y = y.astype(np.float32)

y.shape = (y.shape[0], 1)

# convert champion ids to indices and then one-hot encode
from champion_dicts import ChampionConverter

conv = ChampionConverter()
x_idx = np.zeros((x.shape[0], x.shape[1]))
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        x_idx[i,j] = conv.champion_ids_to_indices[x[i,j]]

x = x_idx

# convert to one hot
print("Converting to one hot...")
one_hot = np.zeros((x.shape[0], x.shape[1], CHAMP_NUM))
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        one_hot[i,j,int(x[i,j]-1)] = 1

x_hot = one_hot

def split_data(x, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    l = x.shape[0]
    train_x = x[:int(l*train_ratio)]
    val_x = x[int(l*train_ratio):int(l*(train_ratio+val_ratio))]
    test_x = x[int(l*(train_ratio+val_ratio)):]

    train_y = y[:int(l*train_ratio)]
    val_y = y[int(l*train_ratio):int(l*(train_ratio+val_ratio))]
    test_y = y[int(l*(train_ratio+val_ratio)):]

    return train_x, train_y, val_x, val_y, test_x, test_y

def print_random_sample(x, y):
    import random
    i = random.randint(0, x.shape[0]-1)
    print("Sample: ", i)
    print(x[i])
    print(y[i])



train_x_1h, train_y, val_x_1h, val_y, test_x_1h, test_y = split_data(x_hot, y)

train_x, _, val_x, _, test_x, _ = split_data(x, y)


avg_win_chance = np.average(train_y)
print("Average win chance: ", avg_win_chance)


# baseline model
class BaselineModel(tf.keras.Model):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu', input_shape=(CHAMP_NUM*PLAYER_NUM,))
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense3 = tf.keras.layers.Dense(128, activation='relu')
        self.dense4 = tf.keras.layers.Dense(64, activation='relu')
        self.dense5 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.dense5(x)
    

# train baseline model
baseline_model = BaselineModel()
baseline_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

#baseline_model.fit(train_x_1h, train_y, epochs=10, validation_data=(val_x_1h, val_y))

# evaluate baseline model
#baseline_model.evaluate(test_x_1h, test_y)


class WinChanceV2(tf.keras.Model):
    def __init__(self):
        super(WinChanceV2, self).__init__()
        self.embedding = tf.keras.layers.Embedding(CHAMP_NUM, 32, input_length=PLAYER_NUM)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.dense3 = tf.keras.layers.Dense(128, activation='relu')
        self.dense4 = tf.keras.layers.Dense(64, activation='relu')
        self.dense5 = tf.keras.layers.Dense(1, activation='sigmoid')


    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.dense5(x)

from keras import backend as K

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(K.equal(y_true, 1), K.ones_like(y_true) * alpha, K.ones_like(y_true) * (1 - alpha))
        loss = -K.sum(alpha_t * K.pow(1 - p_t, gamma) * K.log(p_t))
        return loss

    return focal_loss_fixed

# train win chance model
win_chance_model = WinChanceV2()
win_chance_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

win_chance_model.fit(train_x, train_y, epochs=5, validation_data=(val_x, val_y), batch_size=32)

# evaluate win chance model with specific test data
# 5x champ 1 vs 5x champ 2
while True:
    champ1 = input("Champ 1: ")
    champ2 = input("Champ 2: ")
    closest1 = conv.get_closest_champion_name(champ1)
    closest2 = conv.get_closest_champion_name(champ2)
    idx1 = conv.champion_names_to_indices[closest1]
    idx2 = conv.champion_names_to_indices[closest2]
    test_in = [idx1]*5 + [idx2]*5
    test_in = np.array(test_in)

    y_pred = win_chance_model.predict(test_in.reshape(1,10))

    print("Win chance of ", closest1, " vs ", closest2, ": ", y_pred[0][0])


