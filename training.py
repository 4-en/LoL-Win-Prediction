# basic training loop for different models

from typing import Any
import tensorflow as tf
import numpy as np


CHAMP_NUM = 170
PLAYER_NUM = 10

# load data
DATAPATH = "data/kr_24h/game_data.npy"
data = np.load(DATAPATH)
print("Loaded data of shape: ", data.shape)

np.random.shuffle(data)

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

class TrivialModel(tf.keras.Model):
    """A trivial model that always predicts the average win chance"""
    def __init__(self):
        super(TrivialModel, self).__init__()
        self.prediction = avg_win_chance

    def call(self, inputs):
        if len(inputs.shape) > 1:
            return np.array([self.prediction]*inputs.shape[0])
        return np.array([self.prediction])


# baseline model, just some dense layers
class BaselineModel(tf.keras.Model):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu', input_shape=(None,CHAMP_NUM))
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.dense4 = tf.keras.layers.Dense(128, activation='relu')
        self.dense5 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = tf.reshape(inputs, (-1, PLAYER_NUM, CHAMP_NUM))
        # same dense for every player
        x = self.dense1(x)
        # shape = (-1, 10, 32  )
        # flatten
        x = tf.reshape(x, (-1, 32*10))
        # 3 dense layers, last one is output of (-1, 1)
        x = self.dense2(x)
        x = self.dense4(x)
        return self.dense5(x)

import matplotlib.pyplot as plt
def plot_hist(hist) -> None:
    
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()

# lr scheduler
scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0001 * 0.96**epoch)
#win_chance_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#              loss=tf.keras.losses.MeanSquaredError(),
#              metrics=['accuracy'])

from augmentation import MatchAugmentation

aug = MatchAugmentation(train_x, train_y, aug_chance=0.95, batch_size=32)
val_aug = MatchAugmentation(val_x, val_y, aug_chance=0.8, batch_size=1, max_replace=0)
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



from models.SynergyModel import SynergyModel
from models.basic_embedding_model import BasicEmbedding

from models.DeepConv_model import DeepConv

from models.prob_sample_model import SamplingModel

model = DeepConv()
# l2 regularization with binary crossentropy loss
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

hist = model.fit(aug, epochs=5, validation_data=(val_aug_x, val_aug_y), batch_size=32, callbacks=[scheduler])

# fit without augmentation
#hist = model.fit(train_x, train_y, epochs=5, validation_data=(val_x, val_y), batch_size=32, callbacks=[scheduler])
#hist = model.fit(train_x_1h, train_y, epochs=5, validation_data=(val_x_1h, val_y), batch_size=32, callbacks=[scheduler])

plot_hist(hist)
# save model weights
#model.save_weights("models/basic_embedding_model_weights.h5")

def find_optimal_champion():
    # get random sample from test data
    l = test_x.shape[0]
    idx = np.random.randint(0, l)
    sample = test_x[idx]
    # shuffle sample
    blue = sample[:5]
    red = sample[5:]
    # shuffle blue
    np.random.shuffle(blue)
    # shuffle red
    np.random.shuffle(red)
    sample = np.concatenate((blue, red))
    pick_order = [0,3,4,7,8,1,2,5,6,9]
    remove_num = np.random.randint(1, 10)
    remove_num = 9
    mask = np.where(np.array(pick_order) < remove_num, 1, 0)
    match = sample * mask

    all_champs = np.arange(conv.champion_count)

    # duplicate matches
    matches = np.tile(match, conv.champion_count)
    matches = matches.reshape((-1, 10))
    matches[:, remove_num] = all_champs[:]

    results = model.predict(matches)

    # sorts all_champs based on results
    results = results.reshape((-1,))
    indicies = np.argsort(results)
    #indicies = indicies[::-1]
    all_champs = all_champs[indicies]
    results = results[indicies]
    blue = [ conv.get_champion_name_from_index(i) for i in match[:5]]
    red = [ conv.get_champion_name_from_index(i) for i in match[5:]]

    print("5 best picks for")
    print(blue)
    print("vs")
    print(red)
    for i in range(5):
        wr = results[i]
        wr_rounded = round(float(wr) * 100, 2)
        champ = all_champs[i]
        print(champ)
        champ_name = conv.get_champion_name_from_index(int(champ))
        print(f"{champ_name}: {wr_rounded}%")

while True:
    find_optimal_champion()
    input()




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

    y_pred = model.predict(test_in.reshape(1,10))

    print("Win chance of ", closest1, " vs ", closest2, ": ", y_pred[0][0])


