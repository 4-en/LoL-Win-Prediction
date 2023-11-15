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
        self.dense1 = tf.keras.layers.Dense(256, activation='relu', input_shape=(None,CHAMP_NUM,PLAYER_NUM))
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense4 = tf.keras.layers.Dense(64, activation='relu')
        self.dense5 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = tf.reshape(inputs, (-1, CHAMP_NUM*PLAYER_NUM))
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense4(x)
        return self.dense5(x)
    

# train baseline model
#baseline_model = BaselineModel()
#baseline_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#              loss=tf.keras.losses.MeanSquaredError(),
#              metrics=['accuracy'])

# summarize baseline model
#baseline_model.build(input_shape=(None, CHAMP_NUM, PLAYER_NUM))
#baseline_model.summary()

#baseline_model.fit(train_x_1h, train_y, epochs=10, validation_data=(val_x_1h, val_y))

# evaluate baseline model
#res = baseline_model.evaluate(test_x_1h, test_y)
#print("Baseline model test loss: ", res[0])
#print("Baseline model test accuracy: ", res[1])


class WinChance(tf.keras.Model):
    def __init__(self):
        super(WinChance, self).__init__()
        self.conv0 = tf.keras.layers.Conv1D(32, 1, activation='relu', input_shape=(PLAYER_NUM, CHAMP_NUM))
        self.conv1 = tf.keras.layers.Conv1D(32, 2, activation='relu', input_shape=(PLAYER_NUM, 32))
        self.maxpool = tf.keras.layers.MaxPool1D(2)
        self.flatten = tf.keras.layers.Flatten()
        # 160
        self.dense1 = tf.keras.layers.Dense(220, activation='relu')
        self.dense2 = tf.keras.layers.Dense(220, activation='relu')

        self.conv2 = tf.keras.layers.Conv1D(64, 2, activation='relu', input_shape=(10,22))
        self.maxpool2 = tf.keras.layers.MaxPool1D(2)
        self.flatten2 = tf.keras.layers.Flatten()
        # 60
        self.dense3 = tf.keras.layers.Dense(300, activation='relu')
        self.dense4 = tf.keras.layers.Dense(300, activation='relu')

        self.conv3 = tf.keras.layers.Conv1D(128, 3, activation='relu', input_shape=(10,30))
        self.maxpool3 = tf.keras.layers.MaxPool1D(2)
        self.flatten3 = tf.keras.layers.Flatten()
        # 28
        self.dense5 = tf.keras.layers.Dense(420, activation='relu')
        self.dense6 = tf.keras.layers.Dense(128, activation='relu')
        self.denseOut = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        inputs = tf.reshape(inputs, (-1, PLAYER_NUM, CHAMP_NUM))

        x = self.conv0(inputs)
        #x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.reshape(x, (-1, 10, 22))

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = tf.reshape(x, (-1, 10, 30))

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten3(x)
        x = self.dense5(x)
        x = self.dense6(x)
        return self.denseOut(x)
    
#win_chance_model_1 = WinChance()
#win_chance_model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#              loss=tf.keras.losses.MeanSquaredError(),
#              metrics=['accuracy'])

#win_chance_model_1.fit(train_x_1h, train_y, epochs=10, validation_data=(val_x_1h, val_y))




class WinChanceV2(tf.keras.Model):
    def __init__(self):
        super(WinChanceV2, self).__init__()
        self.embedding = tf.keras.layers.Embedding(CHAMP_NUM, 32, input_length=PLAYER_NUM)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(69, activation='relu')
        self.dense3 = tf.keras.layers.Dense(69, activation='relu')
        self.dense4 = tf.keras.layers.Dense(69, activation='relu')
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

def no_avg_loss(scale=1):
    mse = tf.keras.losses.MeanSquaredError()
    def loss(y_true, y_pred):
        # calculate max difference between all y_trues
        max_diff_true = tf.reduce_max(y_true) - tf.reduce_min(y_true)
        # calculate max difference between all y_preds
        max_diff_pred = tf.reduce_max(y_pred) - tf.reduce_min(y_pred)
        return mse(y_true, y_pred) + scale * (max_diff_true / (max_diff_pred+1e-8))
    return loss

# train win chance model
win_chance_model = WinChanceV2()
win_chance_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

from augmentation import MatchAugmentation

aug = MatchAugmentation(train_x, train_y, aug_chance=0.5, batch_size=32)
val_aug = MatchAugmentation(val_x, val_y, aug_chance=0.5, batch_size=32)
val_aug_x, val_aug_y = [], []
for i in range(len(val_aug)):
    x, y = val_aug[i]
    x = x[0]
    y = y[0]
    assert x.shape == (10,) and y.shape == (1,)
    val_aug_x.append(x)
    val_aug_y.append(y)

val_aug_x = np.array(val_aug_x)
val_aug_y = np.array(val_aug_y)
print(val_aug_x[-1].shape, val_aug_y[-1].shape)

#win_chance_model.fit(aug, epochs=5, validation_data=(val_x, val_y), batch_size=32)
#win_chance_model.fit(aug, epochs=10, validation_data=(val_aug_x, val_aug_y), batch_size=32)

# save model
#win_chance_model.save_weights("models/win_chance_model_v2.h5")

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


