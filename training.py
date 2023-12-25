# basic training loop for different models

from typing import Any
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


CHAMP_NUM = 170
PLAYER_NUM = 10

# load data
DATAPATH = "data/kr_24h/game_data_filtered.npy"
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


def split_data(x, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    l = x.shape[0]
    train_x = x[:int(l*train_ratio)]
    val_x = x[int(l*train_ratio):int(l*(train_ratio+val_ratio))]
    test_x = x[int(l*(train_ratio+val_ratio)):]

    train_y = y[:int(l*train_ratio)]
    val_y = y[int(l*train_ratio):int(l*(train_ratio+val_ratio))]
    test_y = y[int(l*(train_ratio+val_ratio)):]

    return train_x, train_y, val_x, val_y, test_x, test_y





train_x, train_y, val_x, val_y, test_x, test_y = split_data(x, y)



avg_win_chance = np.average(train_y)
print("Average Blue side win chance: ", avg_win_chance)


def plot_hist(hist) -> None:
    
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.show()


# lr scheduler
scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0001 * 0.97**epoch)
#win_chance_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#              loss=tf.keras.losses.MeanSquaredError(),
#              metrics=['accuracy'])

from augmentation import MatchAugmentation

# augment training data:
# - shuffle champions, since order in champion select doesn't match specific role
# - replace champions with random ones, more often then not, this will not change the outcome significantly, a lot of combinations possible
# - mask out champions, so that the model can learn to predict the outcome of a match with a missing champion, eg during champion select
aug = MatchAugmentation(train_x, train_y, aug_chance=0.98, batch_size=32, max_replace=4)




# different models
from models.SynergyModel import SynergyModel
from models.basic_embedding_model import BasicEmbedding

from models.DeepConv_model import DeepConv

from models.lol_transformer import LoLTransformer

from models.deep_embedding import DeepEmbedding

from models.prob_sample_model import SamplingModel


import stats


#model = DeepConv(emb_dim=32, conv_layers=3)
model = LoLTransformer(32,32, 8)

model.build((1,10))

model.summary()

#model.load_weights("models/lol_transformer_32_32.h5")

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

aug.batch_size = 4
aug.aug_chance = 0.3
aug.max_replace = 1
hist = model.fit(aug, epochs=2, validation_data=(val_x, val_y), callbacks=[scheduler])
aug.aug_chance = 0.7
aug.batch_size = 16
aug.max_replace = 4
hist = model.fit(aug, epochs=3, validation_data=(val_x, val_y), callbacks=[scheduler])
aug.aug_chance = 0.98
aug.batch_size = 64
hist = model.fit(aug, epochs=20, validation_data=(val_x, val_y), callbacks=[scheduler])

from lol_prediction import LoLPredictor

#pred = LoLPredictor(model)

#chance = pred.win_chance(["Ahri", "Elise", "Singed"], ["DrMundo", "Kassadin"])
#print(chance)

#pred.best_pick(["Ahri", "Elise", "Singed"], ["DrMundo", "Kassadin"])

#ret = pred.best_pick(["Ahri", "Elise", "Singed"], ["DrMundo", "Draven"], available=["Taliyah", "Yone", "Orianna"])



# test accuracy
test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
print("Test accuracy: ", test_acc)
print("Test loss: ", test_loss)

plot_hist(hist)
# save model weights
model.save_weights("models/lol_transformer_32_32_8.h5")

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


