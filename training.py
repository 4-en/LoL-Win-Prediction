# basic training loop for different models

from typing import Any
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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
print(conv.champion_ids)

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


class TrivialModel(tf.keras.Model):
    """A trivial model that always predicts the average win chance"""
    def __init__(self):
        super(TrivialModel, self).__init__()
        self.prediction = avg_win_chance

    def call(self, inputs):
        batch_size = 1
        if len(inputs.shape) > 1:
            if inputs.shape[0] != None:
                batch_size = inputs.shape[0]
        t = tf.constant(self.prediction, shape=(batch_size, 1))
        return t


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
aug = MatchAugmentation(train_x, train_y, aug_chance=0.90, batch_size=64, max_replace=0)

# only shuffle and mask, no replacement
# same augmentations every epoch for validation data
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

from models.lol_transformer import LoLTransformer

from models.deep_embedding import DeepEmbedding

from models.prob_sample_model import SamplingModel

from models.ensemble_model import EnsembleModel

import stats

model0 = TrivialModel()
model0.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])

hist0 = model0.fit(train_x, train_y, epochs=3, validation_data=(val_x, val_y), batch_size=64, callbacks=[scheduler])

#model = DeepConv(emb_dim=32, conv_layers=3)
model1 = BasicEmbedding()
model1.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

hist1 = model1.fit(aug, epochs=3, validation_data=(val_aug_x, val_aug_y), batch_size=64, callbacks=[scheduler])

model2 = DeepEmbedding(n_layers=2) 
model2.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])

hist2 = model2.fit(aug, epochs=1, validation_data=(val_aug_x, val_aug_y), batch_size=64, callbacks=[scheduler])

MODEL = model1
HIST = hist1


# Create an ensemble model
model3 = EnsembleModel(models=[model1, model2])
model3.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=['accuracy'])


hist3 = model3.fit(train_x, train_y, epochs=3, validation_data=(val_x, val_y), batch_size=64, callbacks=[scheduler])




comp = stats.ModelComparator((test_x, test_y))
comp.add_model(model0, hist0, "TrivialModel")
comp.add_model(model1, hist1, "BasicEmbedding")
comp.add_model(model2, hist2, "DeepEmbedding")
comp.add_model(model3, hist3, "EnsembleModel")


from lol_prediction import LoLPredictor



# fit without augmentation
#hist = model.fit(aug, epochs=3, validation_data=(val_x, val_y), batch_size=32, callbacks=[scheduler])
#hist = model.fit(train_x_1h, train_y, epochs=5, validation_data=(val_x_1h, val_y), batch_size=32, callbacks=[scheduler])

# save model weights
#model.save_weights("models/lol_transformer_8_8.h5")

def find_optimal_champion():
    print("finding optimal champions")
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

    results = MODEL.predict(matches)

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


# Choose model
# Champion prediction
# Display blue sides average win rate
# Display comp plots
# Plot history
# Visualize embeddings
# Print test accuracy and loss

def menu():
    print("")
    print("1. Choose Model")
    print("2. Find Optimal Champion")
    print("3. Display Blue Sides Average Win Rate")
    print("4. Display Comp Plots")
    print("5. Plot History")
    print("6. Visualize Embeddings")
    print("7. Print Test Accuracy and Loss")
    print("8. Win Chance of Two Champions Head to Head")
    print("9. LoLPredictor")
    print("0. Exit")

def option1():
    global MODEL
    global HIST
    print("Choose the model used for the predictions")
    print("1. Trivial Model")
    print("2. Basic Embedding")
    print("3. Deep Embedding")
    print("4. Ensemble Model")
    model_choice = input("Enter your choice (1-4): ")
    print("")

    if model_choice == "1":
        MODEL = model0
        HIST = hist0
        print("Model set to Trivial Model")
    elif model_choice == "2":
        MODEL = model1
        HIST = hist1
        print("Model set to Basic Embedding")
    elif model_choice == "3":
        MODEL = model2
        HIST = hist2
        print("Model set to Deep Embedding")
    elif model_choice == "4":
        MODEL = model3
        HIST = hist3
        print("Model set to Ensemble Model")
    else:
        print("Invalid choice. Please enter a number between 1 and 4.")
        print("")

def option2():
    find_optimal_champion()

def option3():
    print("Average Blue side win chance: ", avg_win_chance)
   

def option4():
    comp.plot_histories()
    comp.plot_bar()
    comp.print_summary()
    comp.print_table()

def option5():
    plot_hist(HIST)

def option6():
    stats.print_embedding_norms(MODEL.embedding, CHAMP_NUM)
    stats.visualize_embeddings(MODEL.embedding, CHAMP_NUM)
    

def option7():
    test_loss, test_acc = MODEL.evaluate(test_x, test_y, verbose=2)
    print("Test accuracy: ", test_acc)
    print("Test loss: ", test_loss)

def option8():
    champ1 = input("Champ 1: ")
    champ2 = input("Champ 2: ")
    closest1 = conv.get_closest_champion_name(champ1)
    closest2 = conv.get_closest_champion_name(champ2)
    idx1 = conv.champion_names_to_indices[closest1]
    idx2 = conv.champion_names_to_indices[closest2]
    test_in = [idx1]*5 + [idx2]*5
    test_in = np.array(test_in)

    y_pred = MODEL.predict(test_in.reshape(1,10))

    print("Win chance of ", closest1, " vs ", closest2, ": ", y_pred[0][0])

def option9():
    pred = LoLPredictor(MODEL)

    chance = pred.win_chance(["Ahri", "Elise", "Singed"], ["DrMundo", "Kassadin"])
    print(chance)

    pred.best_pick(["Ahri", "Elise", "Singed"], ["DrMundo", "Kassadin"])

    ret = pred.best_pick(["Ahri", "Elise", "Singed"], ["DrMundo", "Draven"], available=["Taliyah", "Yone", "Orianna"])

    print(ret)

while True:
    menu()
    choice = input("Enter your choice (0-9): ")
    print("")
    if choice == "1":
        option1()
    elif choice == "2":
        option2()
    elif choice == "3":
        option3()
    elif choice == "4":
        option4()
    elif choice == "5":
        option5()
    elif choice == "6":
        option6()
    elif choice == "7":
        option7()
    elif choice == "8":
        option8()
    elif choice == "9":
        option9()
    elif choice == "0":
        print("Exiting the program. Goodbye!")
        break
    else:
        print("Invalid choice. Please enter a number between 0 and 9.")








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

    y_pred = MODEL.predict(test_in.reshape(1,10))

    print("Win chance of ", closest1, " vs ", closest2, ": ", y_pred[0][0])


