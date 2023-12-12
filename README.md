
# League of Legends Win Chance Prediction

### ML model to predict the outcome of a League of Legends match based on champion selection

## Introduction

League of Legends, often abbreviated as LoL, is a popular online
multiplayer video game. It\'s a competitive 5 versus 5 team-based game
in which players control unique champions with special abilities and
work together to defeat the opposing team. The main objective is to
destroy the enemy team\'s Nexus, a structure in their base, while
defending your own. It combines elements of strategy, teamwork, and
individual skill and is known for its strategic depth and fast-paced
action. League of Legends is played by millions of players worldwide and
has a thriving esports scene with professional leagues and tournaments.

In the competitive environment of League of Legends, players are always
looking for ways to improve their chances of winning. Since it\'s a
strategy game, one key element affecting a team\'s success is the mix of
champions they pick. Our aim is to create a model that helps players
make better decisions about champion selection and team composition by
predicting the likelihood of each team winning based on their chosen
champions. This also enables the most dedicated players to dodge an
unfavorable matchup before the game begins in such a case where the
prediction of their chances of winning are looking less than good.

The information about the match is limited to just the champions picked
before the game actually begins, so we are going to be using only this
information for training our model.

## Expected Results

-   since the are many variables we cant account for, we don\'t expect
    to get very high accuracy
-   accuracy at least over avg win rate, but maybe between 55 to 60%
-   use the trained model to predict outcomes of matches during and
    after champ select
-   use the trained model to find optimal champion during champ select
    (best win chance)

## Similar Works

### LoL-Match-Prediction

[LoL-Match-Prediction](https://github.com/minihat/LoL-Match-Prediction)
Similar to our project, this project tries to predict the win chance of
League of Legends matches. One of the key differences is, that the input
has a lot more values than just the 10 champions we are using, including
values that are only known during the game. This makes it a lot easier
to predict the outcome, since there are less unknown factors that cannot
be accounted for.

The architecture used in this project is a simple fully connected
network with multiple layers. We will try something similar for our
simpler models and then compare it with more complex architectures.

In some test, the model achieved accuracies of around 84%, which is much
higher than we expect with our limited input data.

### Let\'s Predict League of Legends Match Score!

[Let\'s Predict League of Legends Match
Score!](https://www.kaggle.com/code/gulsahdemiryurek/let-s-predict-league-of-legends-match-score)
This project uses Decision Trees to predict the outcome of games.
Similar to the first example, this project also uses a lot more data
than we have available. Although we won\'t use an architecture like
this, it is impressive that they achieved an accuracy of \~97%.

Overall, there are a few examples of projects similar to ours, but they
differ in a few aspects. Due to this, we will try our own ideas for
different architectures and measure the performance to compare them.

## Dataset

There are several datasets available online that contain information
about the outcome of the game, champions selected, player stats and much
more. There is also the official Riot Games API available, which could
be used to gather data from the latest version of the game.

For the purpose of this concept, we will be using a dataset from Kaggle.
This gives us easy access to a lot of training data, without being
limited by the API. While this means that the data is not up to date, it
is still a good starting point for our model and useful for evaluating
the concept.

The dataset [League of Legends- 1 day\'s worth of solo queue
KR](https://www.kaggle.com/datasets/junhachoi/all-ranked-solo-games-on-kr-server-24-hours/)
contains information about all ranked matches on the League of Legends
Korean Server during the course of 1 day (GMT 2022/07/02 00:00:00 to
2022/07/03 00:00:00). In total, this amounts to over 250.000 matches.
The advantage this dataset has over other datasets is that it is very
large and one of the most recent ones available. The data is also from a
single day, which means that the game version is the same for all
matches. This is important because the game is constantly being updated
and the balance of champions changes with every patch. This means that
the data from older patches is not as useful for training our model.

### Data cleaning

We will try to clean and improve the dataset by removing some outliers
that could negatively impact training. The data cleaning will only be
applied to the training data, so we can determine if it had a positive
impact. We will also compare the cleaned data with uncleaned data to
decide which method is better.

The dataset contains multiple values that could be helpful to filter the
games. In general, we want to remove games in which the champion
selection had a lower impact than usual. We are doing this by finding
games that were already uneven from the beginning, since this could
indicate that the players\' skill level was further apart than normal or
that something else that had nothing to do with the champions effected
the game, like a player disconnection or not participating.

There is a risk that filtering the games in this way will create some
form of bias, especially against champions that are stronger in the
early stages of the game. We will have to measure how this cleaning
effects the performance of our models and adjust the thresholds for
filtering games if necessary.

The keys we are using to filter games are the following:

-   gameEndedInEarlySurrender
    -   This flag indicates whether one team surrendered early.
-   timePlayed
    -   We are filtering games that are shorter than 20min, since these
        games were probably very uneven.
-   champLevel
    -   We are filtering games were one or more champions are far below
        the average, indicating that they were not participating.

``` python
# if false, preprocessed data will be loaded from disk
# same results, but faster
CONVERT_DATA = False
import numpy as np
train, train_filtered, val, test = None, None, None, None

if CONVERT_DATA:
    # load data used for training
    import data.kr_24h.convert as convert

    # load the data, including stats we dont need
    games_dict = convert.load_raw_csv(file_path="data/kr_24h/kr_soloq_24h/sat df.csv")
    games = list(games_dict.values())

    # shuffle games
    np.random.shuffle(games)

    # split into train, val, test
    train, val, test = convert.split_iterable(games, weights=(90, 5, 5))
    print("train: ", len(train))
    print("val: ", len(val))
    print("test: ", len(test))
    print()

    # convert each match into a list of 10 champions and a 1/0 for win/loss of blue team
    # two copies of train data, one with some matches filtered out
    train, train_filtered = convert.convert_data(train, filter_matches=False), convert.convert_data(train, filter_matches=True)
    val = convert.convert_data(val, filter_matches=False)
    test = convert.convert_data(test, filter_matches=False)
```

``` python
# for some reason, this takes like 20min to run in the notebook, but only 1min directly in python

# to save some time, we saved the data to a file and load it here, but you can also just run the above code if you have time
# the result is the same
import numpy as np
dir = "data/notebook/"


if not CONVERT_DATA:
    # load the data
    train = np.load(dir + "train.npy")
    train_filtered = np.load(dir + "train_filtered.npy")
    val = np.load(dir + "val.npy")
    test = np.load(dir + "test.npy")

    # print lengths
    print("train: ", len(train))
    print("train_filtered: ", len(train_filtered))
    print("val: ", len(val))
    print("test: ", len(test))
```


## Data Analysis

Before we can start training our model, we need to do some data analysis
to get a better understanding of the data. This will help us decide
which features to use and how to process them. It can also help us with
evaluating the performance of our models later on.

### Overall Win Rate

The first thing we want to look at is the overall win rate (of the blue
side). Since the game is not symmetrical, we can\'t assume that the win
rate is 50%. In fact, during most patches, the blue side (bottom left)
has a slightly higher win rate than the red side. This can be explained
by several factors, such as the camera angle, the position of the
minimap, and the position of the HUD. The blue side also has a slight
advantage in champion select, since they get to pick first.

This overall win rate gives us a baseline for our model. If our model is
not able to beat this baseline, then it is not very useful. The overall
win rate is calculated by dividing the number of wins by the total
number of matches.

``` python
# calculate overall win rate here
```

### Champion Win Rate

Next, we want to look at the win rate of each champion. This gives us an
idea of how strong each champion is and how likely they are to win. We
can also see which champions are the most popular and which ones are the
least popular. With this, we can evaluate the performance of our model
and see if it is able to predict the outcome of the game better than
just picking the most popular champions. If we match champions with high
win rates against champions with low win rates, we can also see if our
models are able to predict the outcome of the game correctly.

``` python
# split into x and y
train_x, train_y = train[:, :-1], train[:, -1]
train_filtered_x, train_filtered_y = train_filtered[:, :-1], train_filtered[:, -1]
val_x, val_y = val[:, :-1], val[:, -1]
test_x, test_y = test[:, :-1], test[:, -1]

# convert y to float and to correct shape
def conv_y(y):
    y = y.astype(float)
    y = y.reshape(-1, 1)
    return y

train_y, train_filtered_y = conv_y(train_y), conv_y(train_filtered_y)
val_y, test_y = conv_y(val_y), conv_y(test_y)


# convert champion ids to indices and then one-hot encode
from champion_dicts import ChampionConverter

# see champion_dicts.py for more info
# we have to convert the champion ids from the data into indices, since the ids are not contiguous
# (some ids are 500+, but there are less than 170 champions)
champ_converter = ChampionConverter()

def conv_x(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = champ_converter.get_champion_index_from_id(x[i, j])
    return x

train_x, train_filtered_x = conv_x(train_x), conv_x(train_filtered_x)
val_x, test_x = conv_x(val_x), conv_x(test_x)


import numpy as np

# one-hot encode the champions, used by simple models
CHAMP_NUM = 170 # number of champions, actually a bit less, but this way we could keep same model for more champions
def one_hot_encode(x):
    one_hot = np.zeros((x.shape[0], x.shape[1], CHAMP_NUM))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            one_hot[i,j,int(x[i,j]-1)] = 1
    return one_hot

train_x_1hot = one_hot_encode(train_x)
train_filtered_x_1hot = one_hot_encode(train_filtered_x)
val_x_1hot = one_hot_encode(val_x)
test_x_1hot = one_hot_encode(test_x)


```

``` python
# calculate average win chance, we should at least beat this :)
avg_win_chance = np.average(train_y)
print("average blue side win chance: ", avg_win_chance)
```


    average blue side win chance:  0.5176064194987985

``` python
import tensorflow as tf

import stats # for printing and plotting model performance



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
        x = tf.reshape(inputs, (-1, 10, CHAMP_NUM))
        # same dense for every player
        x = self.dense1(x)
        # shape = (-1, 10, 32  )
        # flatten
        x = tf.reshape(x, (-1, 32*10))
        # 3 dense layers, last one is output of (-1, 1)
        x = self.dense2(x)
        x = self.dense4(x)
        return self.dense5(x)
```

``` python
# trivial model for comparison
trivial_model = TrivialModel()
trivial_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# this model is not trainable, but we can still fit it to get the history for plotting
trivial_hist = trivial_model.fit(train_x_1hot, train_y, epochs=10, batch_size=256, validation_data=(val_x_1hot, val_y))

# train baseline model

base_model = BaselineModel()
base_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

base_hist = base_model.fit(train_x_1hot, train_y, epochs=6, batch_size=32, validation_data=(val_x_1hot, val_y))

# plot the results
stats.plot_history(trivial_hist)
stats.plot_history(base_hist)
```


    Epoch 1/10
    911/911 [==============================] - 2s 2ms/step - loss: 0.6925 - accuracy: 0.5176 - val_loss: 0.6933 - val_accuracy: 0.5072
    Epoch 2/10
    911/911 [==============================] - 1s 2ms/step - loss: 0.6925 - accuracy: 0.5176 - val_loss: 0.6933 - val_accuracy: 0.5072
    Epoch 3/10
    911/911 [==============================] - 1s 2ms/step - loss: 0.6925 - accuracy: 0.5176 - val_loss: 0.6933 - val_accuracy: 0.5072
    Epoch 4/10
    911/911 [==============================] - 1s 2ms/step - loss: 0.6925 - accuracy: 0.5176 - val_loss: 0.6933 - val_accuracy: 0.5072
    Epoch 5/10
    911/911 [==============================] - 1s 2ms/step - loss: 0.6925 - accuracy: 0.5176 - val_loss: 0.6933 - val_accuracy: 0.5072
    Epoch 6/10
    911/911 [==============================] - 1s 2ms/step - loss: 0.6925 - accuracy: 0.5176 - val_loss: 0.6933 - val_accuracy: 0.5072
    Epoch 7/10
    911/911 [==============================] - 1s 2ms/step - loss: 0.6925 - accuracy: 0.5176 - val_loss: 0.6933 - val_accuracy: 0.5072
    Epoch 8/10
    911/911 [==============================] - 1s 2ms/step - loss: 0.6925 - accuracy: 0.5176 - val_loss: 0.6933 - val_accuracy: 0.5072
    Epoch 9/10
    911/911 [==============================] - 1s 2ms/step - loss: 0.6925 - accuracy: 0.5176 - val_loss: 0.6933 - val_accuracy: 0.5072
    Epoch 10/10
    911/911 [==============================] - 1s 2ms/step - loss: 0.6925 - accuracy: 0.5176 - val_loss: 0.6933 - val_accuracy: 0.5072
    Epoch 1/6
    7283/7283 [==============================] - 22s 3ms/step - loss: 0.6904 - accuracy: 0.5299 - val_loss: 0.6913 - val_accuracy: 0.5252
    Epoch 2/6
    7283/7283 [==============================] - 21s 3ms/step - loss: 0.6883 - accuracy: 0.5394 - val_loss: 0.6899 - val_accuracy: 0.5317
    Epoch 3/6
    7283/7283 [==============================] - 21s 3ms/step - loss: 0.6869 - accuracy: 0.5447 - val_loss: 0.6906 - val_accuracy: 0.5313
    Epoch 4/6
    7283/7283 [==============================] - 21s 3ms/step - loss: 0.6837 - accuracy: 0.5547 - val_loss: 0.6951 - val_accuracy: 0.5220
    Epoch 5/6
    7283/7283 [==============================] - 22s 3ms/step - loss: 0.6768 - accuracy: 0.5698 - val_loss: 0.6962 - val_accuracy: 0.5234
    Epoch 6/6
    7283/7283 [==============================] - 21s 3ms/step - loss: 0.6652 - accuracy: 0.5886 - val_loss: 0.7105 - val_accuracy: 0.5187

![](vertopal_9b673558d3ed45ecaffbe0033b33f888/b6fd1fec12d73f16dbee732102ee615add562b65.png)

![](vertopal_9b673558d3ed45ecaffbe0033b33f888/c25666e72385b83ccf5cb6711d1943662c2ba018.png)

![](vertopal_9b673558d3ed45ecaffbe0033b33f888/46b89c2d260461cee7717ad515815bcee6448291.png)

![](vertopal_9b673558d3ed45ecaffbe0033b33f888/2a6369f926a81eaaa5c13dad1dbc14bd16d6c42f.png)

As we can see, the baseline model does manage to get a higher accuracy
than the average win rate. The validation accuracy starts going down
really quickly, while the training accuracy keeps going up. This
indicates that the model is overfitting to the training data. This is
not surprising, since the model is very simple and the training data is
very large. We will have to use a more complex model and try to use some
regularization to prevent overfitting.

## Cleaned Data

Now we are going to check if using the cleaned data improves the
performance of the model. To do that, we are simply going to train the
model with the cleaned data and use the same val data as before. We will
then compare the results to see if there is any improvement.

``` python
# same model, but with some matches filtered out
base_model_filtered = BaselineModel()
base_model_filtered.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

bmf_hist = base_model_filtered.fit(train_filtered_x_1hot, train_filtered_y, epochs=6, batch_size=32, validation_data=(val_x_1hot, val_y))

# plot the results
stats.plot_history(bmf_hist)
```


    Epoch 1/6
    6397/6397 [==============================] - 15s 2ms/step - loss: 0.6906 - accuracy: 0.5298 - val_loss: 0.6906 - val_accuracy: 0.5283
    Epoch 2/6
    6397/6397 [==============================] - 14s 2ms/step - loss: 0.6882 - accuracy: 0.5409 - val_loss: 0.6910 - val_accuracy: 0.5324
    Epoch 3/6
    6397/6397 [==============================] - 14s 2ms/step - loss: 0.6869 - accuracy: 0.5441 - val_loss: 0.6917 - val_accuracy: 0.5318
    Epoch 4/6
    6397/6397 [==============================] - 14s 2ms/step - loss: 0.6837 - accuracy: 0.5551 - val_loss: 0.6921 - val_accuracy: 0.5285
    Epoch 5/6
    6397/6397 [==============================] - 14s 2ms/step - loss: 0.6763 - accuracy: 0.5719 - val_loss: 0.6967 - val_accuracy: 0.5210
    Epoch 6/6
    6397/6397 [==============================] - 14s 2ms/step - loss: 0.6630 - accuracy: 0.5946 - val_loss: 0.7158 - val_accuracy: 0.5062

![](vertopal_9b673558d3ed45ecaffbe0033b33f888/ad43c47b8cb3ace6fbf531132c725f81cbaffd1b.png)



![](vertopal_9b673558d3ed45ecaffbe0033b33f888/8c10c449e911551bfd079f3a3a2d579b71587518.png)

We can see that our validation accuracy is slightly higher at almost
0.54 and that the overfitting is a bit weaker, although it is still
quite a lot. We can also observe, that the training accuracy is
increasing faster than before. This is probably due to the fact that the
filtered data contains less noise, so fewer games that are not as
dependent on the champions as the average, which means that it is easier
to learn the impact of each champion.

## Embeddings

Instead of using one-hot encoded vectors, we want to use an embedding
layer. This is not only more convenient and easier to work with, it
could also help to improve the models\' performance as well as the
training speed.

After the embedding layer, this model should be the same as the Baseline
model, so we expect similar performance.

``` python
# a basic model that uses embeddings at the first layer instead of one hot vectors
class BasicEmbedding(tf.keras.Model):
    def __init__(self, champ_num=170, embed_dim=32):
        super(BasicEmbedding, self).__init__()
        self.champ_num = champ_num
        self.embed_dim = embed_dim
        player_num = 10
        self.embedding = tf.keras.layers.Embedding(champ_num, embed_dim, input_length=player_num)
        self.flatten = tf.keras.layers.Flatten()
        
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.dense4 = tf.keras.layers.Dense(128, activation='relu')
        self.dense5 = tf.keras.layers.Dense(1, activation='sigmoid')


    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.flatten(x)
        x = self.dense2(x)
        x = self.dense4(x)
        return self.dense5(x)
```

``` python
basic_embedding = BasicEmbedding()
basic_embedding.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

be_hist = basic_embedding.fit(train_filtered_x, train_filtered_y, epochs=6, batch_size=32, validation_data=(val_x, val_y))

# plot the results
stats.plot_history(be_hist)
```

::: {.output .stream .stdout}
    Epoch 1/6
:::

::: {.output .stream .stdout}
    6397/6397 [==============================] - 19s 3ms/step - loss: 0.6904 - accuracy: 0.5292 - val_loss: 0.6905 - val_accuracy: 0.5280
    Epoch 2/6
    6397/6397 [==============================] - 17s 3ms/step - loss: 0.6885 - accuracy: 0.5393 - val_loss: 0.6906 - val_accuracy: 0.5281
    Epoch 3/6
    6397/6397 [==============================] - 17s 3ms/step - loss: 0.6871 - accuracy: 0.5434 - val_loss: 0.6910 - val_accuracy: 0.5307
    Epoch 4/6
    6397/6397 [==============================] - 17s 3ms/step - loss: 0.6846 - accuracy: 0.5521 - val_loss: 0.6930 - val_accuracy: 0.5319
    Epoch 5/6
    6397/6397 [==============================] - 17s 3ms/step - loss: 0.6788 - accuracy: 0.5651 - val_loss: 0.6955 - val_accuracy: 0.5311
    Epoch 6/6
    6397/6397 [==============================] - 17s 3ms/step - loss: 0.6681 - accuracy: 0.5840 - val_loss: 0.7040 - val_accuracy: 0.5231
:::

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/a3dc915f3977b1191c3db9770105c4c5debd3b68.png)
:::

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/927d1dda56502cacadddb121c9f67772d30d8d6b.png)
:::
:::

::: {.cell .markdown}
We can see that this model is quite similar to the baseline version. We
still have the overfitting problem and get a validation accuracy of
slightly below 0.54. In the next step, we will try to use some data
augmentation to improve the performance of our model and reduce
overfitting.
:::

::: {.cell .markdown}
## Data Augmentation

To reduce overfitting, we are going to use some data augmentation. We
will do this by randomly swapping the champions of the blue and red
side. This will create new training data that is similar to the original
data, but not exactly the same. This should help to reduce overfitting
and improve the performance of our model.

Since one of our goals is to predict the win chance during champion
select and calculate the optimal pick for a given state, we will also
introduce masking to our input data, which will allow us to hide the
champions that have not been picked yet. Since the order in which
champions are selected is more or less random anyway, masking the
selection and keeping the outcome of the game should still result in one
of all possible combinations during champion select. If we do this often
enough, the model hopefully learns how champions interact with each
other and which champions are good against others.

Lastly, we will randomly replace some champions with a random champion.
This is basically introducing some noise to the data, which should help
to reduce overfitting and improve the performance of our model. In
addition, this will effectively increase the size of our training data,
which should also help to reduce overfitting.
:::

::: {.cell .code execution_count="11"}
``` python

import augmentation # for data augmentation
# see augmentation.py for more info

# aug_chance is the chance that a match will be augmented
# max_replace is the maximum number of champions that can be replaced
# batch_size is the batch size used for training
aug = augmentation.MatchAugmentation(train_filtered_x, train_filtered_y, aug_chance=0.90, batch_size=64, max_replace=3)

aug_emb = BasicEmbedding()
aug_emb.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

aug_hist = aug_emb.fit(aug, epochs=20, validation_data=(val_x, val_y))

# plot the results
stats.plot_history(aug_hist)
```

::: {.output .stream .stdout}
    Epoch 1/20
    3198/3198 [==============================] - 13s 4ms/step - loss: 0.6918 - accuracy: 0.5220 - val_loss: 0.6905 - val_accuracy: 0.5285
    Epoch 2/20
    3198/3198 [==============================] - 11s 4ms/step - loss: 0.6910 - accuracy: 0.5265 - val_loss: 0.6910 - val_accuracy: 0.5303
    Epoch 3/20
    3198/3198 [==============================] - 11s 3ms/step - loss: 0.6909 - accuracy: 0.5273 - val_loss: 0.6913 - val_accuracy: 0.5247
    Epoch 4/20
    3198/3198 [==============================] - 11s 4ms/step - loss: 0.6906 - accuracy: 0.5303 - val_loss: 0.6912 - val_accuracy: 0.5265
    Epoch 5/20
    3198/3198 [==============================] - 11s 4ms/step - loss: 0.6906 - accuracy: 0.5286 - val_loss: 0.6911 - val_accuracy: 0.5271
    Epoch 6/20
    3198/3198 [==============================] - 11s 3ms/step - loss: 0.6906 - accuracy: 0.5287 - val_loss: 0.6910 - val_accuracy: 0.5302
    Epoch 7/20
    3198/3198 [==============================] - 11s 4ms/step - loss: 0.6904 - accuracy: 0.5288 - val_loss: 0.6916 - val_accuracy: 0.5271
    Epoch 8/20
    3198/3198 [==============================] - 11s 4ms/step - loss: 0.6904 - accuracy: 0.5301 - val_loss: 0.6905 - val_accuracy: 0.5299
    Epoch 9/20
    3198/3198 [==============================] - 11s 3ms/step - loss: 0.6903 - accuracy: 0.5307 - val_loss: 0.6909 - val_accuracy: 0.5248
    Epoch 10/20
    3198/3198 [==============================] - 11s 4ms/step - loss: 0.6903 - accuracy: 0.5295 - val_loss: 0.6907 - val_accuracy: 0.5286
    Epoch 11/20
    3198/3198 [==============================] - 11s 4ms/step - loss: 0.6906 - accuracy: 0.5291 - val_loss: 0.6913 - val_accuracy: 0.5275
    Epoch 12/20
    3198/3198 [==============================] - 11s 3ms/step - loss: 0.6903 - accuracy: 0.5294 - val_loss: 0.6912 - val_accuracy: 0.5322
    Epoch 13/20
    3198/3198 [==============================] - 11s 4ms/step - loss: 0.6903 - accuracy: 0.5299 - val_loss: 0.6906 - val_accuracy: 0.5333
    Epoch 14/20
    3198/3198 [==============================] - 11s 4ms/step - loss: 0.6903 - accuracy: 0.5296 - val_loss: 0.6930 - val_accuracy: 0.5308
    Epoch 15/20
    3198/3198 [==============================] - 11s 3ms/step - loss: 0.6901 - accuracy: 0.5292 - val_loss: 0.6903 - val_accuracy: 0.5316
    Epoch 16/20
    3198/3198 [==============================] - 11s 4ms/step - loss: 0.6901 - accuracy: 0.5296 - val_loss: 0.6900 - val_accuracy: 0.5307
    Epoch 17/20
    3198/3198 [==============================] - 11s 4ms/step - loss: 0.6901 - accuracy: 0.5302 - val_loss: 0.6904 - val_accuracy: 0.5301
    Epoch 18/20
    3198/3198 [==============================] - 11s 3ms/step - loss: 0.6902 - accuracy: 0.5296 - val_loss: 0.6903 - val_accuracy: 0.5336
    Epoch 19/20
    3198/3198 [==============================] - 11s 3ms/step - loss: 0.6901 - accuracy: 0.5301 - val_loss: 0.6900 - val_accuracy: 0.5312
    Epoch 20/20
    3198/3198 [==============================] - 12s 4ms/step - loss: 0.6901 - accuracy: 0.5303 - val_loss: 0.6911 - val_accuracy: 0.5280
:::

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/3f2bdd62810f4762a76e37e1a9b0f9774587312f.png)
:::

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/f48920dcd6a32d1d90ee4d08faf352202e0aee01.png)
:::
:::

::: {.cell .markdown}
These first test show that the augmentation prevents the model from
overfitting. The accuracy is not higher than before, but we can see that
it keeps improving even after 20 epochs, when it started going down
before. This means that we could probably train the model for longer and
get a higher accuracy, but the downside is that the training will take
longer.

We will now try some other parameters for the augmentation and see if we
can improve the performance of our model. It is possible that we
introduced too much noise, which could be the reason why the accuracy is
not higher than before.
:::

::: {.cell .code execution_count="12"}
``` python

# lower chance of augmentation
# lower max_replace
aug2 = augmentation.MatchAugmentation(train_filtered_x, train_filtered_y, aug_chance=0.85, batch_size=64, max_replace=2)

aug_emb2 = BasicEmbedding()
aug_emb2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

aug_hist2 = aug_emb2.fit(aug2, epochs=20, validation_data=(val_x, val_y))

# plot the results
stats.plot_history(aug_hist2)
```

::: {.output .stream .stdout}
    Epoch 1/20
    3198/3198 [==============================] - 11s 3ms/step - loss: 0.6918 - accuracy: 0.5212 - val_loss: 0.6911 - val_accuracy: 0.5275
    Epoch 2/20
    3198/3198 [==============================] - 11s 4ms/step - loss: 0.6909 - accuracy: 0.5273 - val_loss: 0.6908 - val_accuracy: 0.5278
    Epoch 3/20
    3198/3198 [==============================] - 11s 3ms/step - loss: 0.6905 - accuracy: 0.5286 - val_loss: 0.6906 - val_accuracy: 0.5265
    Epoch 4/20
    3198/3198 [==============================] - 11s 3ms/step - loss: 0.6904 - accuracy: 0.5295 - val_loss: 0.6916 - val_accuracy: 0.5279
    Epoch 5/20
    3198/3198 [==============================] - 11s 3ms/step - loss: 0.6901 - accuracy: 0.5300 - val_loss: 0.6910 - val_accuracy: 0.5265
    Epoch 6/20
    3198/3198 [==============================] - 11s 3ms/step - loss: 0.6899 - accuracy: 0.5312 - val_loss: 0.6903 - val_accuracy: 0.5312
    Epoch 7/20
    3198/3198 [==============================] - 11s 3ms/step - loss: 0.6901 - accuracy: 0.5319 - val_loss: 0.6905 - val_accuracy: 0.5320
    Epoch 8/20
    3198/3198 [==============================] - 11s 4ms/step - loss: 0.6902 - accuracy: 0.5299 - val_loss: 0.6907 - val_accuracy: 0.5262
    Epoch 9/20
    3198/3198 [==============================] - 11s 3ms/step - loss: 0.6900 - accuracy: 0.5315 - val_loss: 0.6919 - val_accuracy: 0.5278
    Epoch 10/20
    3198/3198 [==============================] - 11s 3ms/step - loss: 0.6903 - accuracy: 0.5304 - val_loss: 0.6914 - val_accuracy: 0.5268
    Epoch 11/20
    3198/3198 [==============================] - 11s 4ms/step - loss: 0.6900 - accuracy: 0.5299 - val_loss: 0.6907 - val_accuracy: 0.5301
    Epoch 12/20
    3198/3198 [==============================] - 11s 4ms/step - loss: 0.6900 - accuracy: 0.5308 - val_loss: 0.6903 - val_accuracy: 0.5254
    Epoch 13/20
    3198/3198 [==============================] - 11s 3ms/step - loss: 0.6900 - accuracy: 0.5306 - val_loss: 0.6906 - val_accuracy: 0.5239
    Epoch 14/20
    3198/3198 [==============================] - 11s 4ms/step - loss: 0.6899 - accuracy: 0.5306 - val_loss: 0.6906 - val_accuracy: 0.5230
    Epoch 15/20
    3198/3198 [==============================] - 13s 4ms/step - loss: 0.6900 - accuracy: 0.5318 - val_loss: 0.6904 - val_accuracy: 0.5311
    Epoch 16/20
    3198/3198 [==============================] - 13s 4ms/step - loss: 0.6897 - accuracy: 0.5315 - val_loss: 0.6913 - val_accuracy: 0.5297
    Epoch 17/20
    3198/3198 [==============================] - 13s 4ms/step - loss: 0.6897 - accuracy: 0.5317 - val_loss: 0.6907 - val_accuracy: 0.5300
    Epoch 18/20
    3198/3198 [==============================] - 12s 4ms/step - loss: 0.6897 - accuracy: 0.5307 - val_loss: 0.6905 - val_accuracy: 0.5308
    Epoch 19/20
    3198/3198 [==============================] - 12s 4ms/step - loss: 0.6897 - accuracy: 0.5308 - val_loss: 0.6906 - val_accuracy: 0.5282
    Epoch 20/20
    3198/3198 [==============================] - 12s 4ms/step - loss: 0.6898 - accuracy: 0.5311 - val_loss: 0.6902 - val_accuracy: 0.5288
:::

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/8f1c063e7fb94118b5a77a70296e581aca71614f.png)
:::

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/9eb3a3d486addbb54c0359ffd31f6bc258fc1a2d.png)
:::
:::

::: {.cell .markdown}
We tested a few different parameters, but it seemed like the best
results were achieved with the following parameters:

-   aug_chance: 0.85
-   max_replace: 2
-   batch_size: 64

This does not necessarily mean that these are also the best parameters
for different models, but we will use them as default values for now and
see if we can improve then later.
:::

::: {.cell .code execution_count="13"}
``` python
import stats
# this module has a few functions and classes that make plotting and comparing performance easier
# see stats.py for more details

comparator = stats.ModelComparator((test_x, test_y))

comparator.add_model(trivial_model, trivial_hist, "Avg chance")
comparator.add_model(basic_embedding, be_hist, "Basic Emb.")
comparator.add_model(aug_emb2, aug_hist2, "Augm. Emb.")

comparator.plot_histories()
```

::: {.output .stream .stdout}
    Evaluating  Avg chance
      1/405 [..............................] - ETA: 25s - loss: 0.6938 - accuracy: 0.5000
:::

::: {.output .stream .stdout}
    405/405 [==============================] - 1s 1ms/step - loss: 0.6926 - accuracy: 0.5173
    Test accuracy:  0.6925509572029114
    Test loss:  0.5172626972198486
    Evaluating  Basic Emb.
    405/405 [==============================] - 1s 2ms/step - loss: 0.7082 - accuracy: 0.5174
    Test accuracy:  0.7081529498100281
    Test loss:  0.5174171328544617
    Evaluating  Augm. Emb.
    405/405 [==============================] - 1s 2ms/step - loss: 0.6898 - accuracy: 0.5313
    Test accuracy:  0.6898033618927002
    Test loss:  0.5313199758529663
:::

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/3b860f72474ab6888d06fb870dd24766c6848c8b.png)
:::

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/ece89860c28fb4a3cfc5d047ced27b05a63fca67.png)
:::
:::

::: {.cell .markdown}
In the graphs above, we can see how the embedding model with
augmentation performs clearly the best and does not suffer from
overfitting. In the next steps we will try to create more complex models
and hopefully improve the performance further.
:::

::: {.cell .markdown}
## Different Architectures

In this section, we will try to use different architectures and measure
their performance.

The first one is still similar to the BasicEmbedding model, since it
starts with an embedding layers followed by a few dense layers. The
difference hear is the number of layers, which, depending on the
parameters, will be a lot higher, and the residual connections to make
training this deeper model easier.

We will also adjust the learning parameters and try out which of them
work the best.
:::

::: {.cell .code execution_count="14"}
``` python
# training parameters
aug = augmentation.MatchAugmentation(train_filtered_x, train_filtered_y, aug_chance=0.90, batch_size=64, max_replace=2)

# adapts learning rate and batch size based on epoch
def create_scheduler(lr_f, bs_f, min_lr=1e-10, max_lr=1e-3, min_bs=4, max_bs=256, aug=aug):
    def schedul_f(epoch):
      bs = bs_f(epoch)
      bs = max(bs, min_bs)
      bs = min(bs, max_bs)

      # adjust batch size in augmentor
      # doesnt work yet :(
      #aug.batch_size = bs

      lr = lr_f(epoch)
      lr = max(lr, min_lr)
      lr = min(lr, max_lr)

      return lr

    scheduler = tf.keras.callbacks.LearningRateScheduler(schedul_f)
    return scheduler
```
:::

::: {.cell .code execution_count="15"}
``` python
# Deep Embedding

# a deep embedding model with multiple layers consisting of dense layers and residual connections and layer norm
class DeepEmbedding(tf.keras.Model):
    def __init__(self, champ_num=170, embed_dim=32, n_layers=4, layer_size=3):
        super(DeepEmbedding, self).__init__()
        self.champ_num = champ_num
        self.embed_dim = embed_dim
        self.n_layers=n_layers
        self.layer_size=layer_size
        player_num = 10
        self.embedding = tf.keras.layers.Embedding(champ_num, embed_dim)
        self.flatten = tf.keras.layers.Flatten()
        self.layer_norm = tf.keras.layers.LayerNormalization()


        self.deep_layers = []
        for _ in range(n_layers):
            l = []
            for _ in range(layer_size):
                l.append(
                    tf.keras.layers.Dense(player_num*embed_dim, activation='gelu')
                )
            self.deep_layers.append(l)

            

        self.dense_output1 = tf.keras.layers.Dense(embed_dim*5, activation='gelu')
        self.dense_output2 = tf.keras.layers.Dense(embed_dim*2, activation='gelu')
        self.dense_output3 = tf.keras.layers.Dense(1, activation='sigmoid')


    def call(self, inputs):
        x0 = self.embedding(inputs)
        x0 = self.flatten(x0)

        for i in range(self.n_layers):
            x = self.deep_layers[i][0](x0)
            for j in range(1, self.layer_size):
                x = self.deep_layers[i][j](x)
                
            x += x0
            x0 = self.layer_norm(x)
        
        x = self.dense_output1(x0)
        x = self.dense_output2(x)
        return self.dense_output3(x)
```
:::

::: {.cell .code execution_count="16"}
``` python

deep_emb_model = DeepEmbedding(n_layers=4, layer_size=3)
deep_emb_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

scheduler = create_scheduler(lambda epoch: 0.0001 * 0.97**epoch, lambda epoch: 8*(2**(epoch//5)))
deep_emb_hist = deep_emb_model.fit(aug, epochs=10, validation_data=(val_x, val_y), callbacks=[scheduler])
```

::: {.output .stream .stdout}
    Epoch 1/10
    3198/3198 [==============================] - 32s 9ms/step - loss: 0.6958 - accuracy: 0.5067 - val_loss: 0.6935 - val_accuracy: 0.5110 - lr: 1.0000e-04
    Epoch 2/10
    3198/3198 [==============================] - 28s 9ms/step - loss: 0.6927 - accuracy: 0.5131 - val_loss: 0.6918 - val_accuracy: 0.5227 - lr: 9.7000e-05
    Epoch 3/10
    3198/3198 [==============================] - 28s 9ms/step - loss: 0.6914 - accuracy: 0.5230 - val_loss: 0.6909 - val_accuracy: 0.5302 - lr: 9.4090e-05
    Epoch 4/10
    3198/3198 [==============================] - 28s 9ms/step - loss: 0.6908 - accuracy: 0.5260 - val_loss: 0.6920 - val_accuracy: 0.5247 - lr: 9.1267e-05
    Epoch 5/10
    3198/3198 [==============================] - 28s 9ms/step - loss: 0.6906 - accuracy: 0.5287 - val_loss: 0.6908 - val_accuracy: 0.5335 - lr: 8.8529e-05
    Epoch 6/10
    3198/3198 [==============================] - 28s 9ms/step - loss: 0.6906 - accuracy: 0.5299 - val_loss: 0.6910 - val_accuracy: 0.5271 - lr: 8.5873e-05
    Epoch 7/10
    3198/3198 [==============================] - 28s 9ms/step - loss: 0.6904 - accuracy: 0.5290 - val_loss: 0.6908 - val_accuracy: 0.5291 - lr: 8.3297e-05
    Epoch 8/10
    3198/3198 [==============================] - 28s 9ms/step - loss: 0.6901 - accuracy: 0.5316 - val_loss: 0.6905 - val_accuracy: 0.5333 - lr: 8.0798e-05
    Epoch 9/10
    3198/3198 [==============================] - 28s 9ms/step - loss: 0.6902 - accuracy: 0.5300 - val_loss: 0.6903 - val_accuracy: 0.5316 - lr: 7.8374e-05
    Epoch 10/10
    3198/3198 [==============================] - 28s 9ms/step - loss: 0.6901 - accuracy: 0.5315 - val_loss: 0.6906 - val_accuracy: 0.5342 - lr: 7.6023e-05
:::
:::

::: {.cell .code execution_count="17"}
``` python

stats.plot_history(deep_emb_hist)
```

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/d5e371137a0add941e59973a4c674dea46228a1d.png)
:::

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/179d0eb06414c5235abfc6e64ccbb098eddbbb30.png)
:::
:::

::: {.cell .markdown}
The smaller version of the DeepEmbedding model performed relatively well
and kept steadily increasing during training, so we will try to train a
version with a few more layers to see if this can improve the
performance further.
:::

::: {.cell .code execution_count="28"}
``` python
aug = augmentation.MatchAugmentation(train_filtered_x, train_filtered_y, aug_chance=0.90, batch_size=64, max_replace=2)

deeper_emb_model = DeepEmbedding(n_layers=10, layer_size=5)
deeper_emb_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

scheduler = create_scheduler(lambda epoch: 0.0001 * 0.90**epoch, lambda epoch: 8*(2**(epoch//5)))
deeper_emb_hist = deeper_emb_model.fit(aug, epochs=30, validation_data=(val_x, val_y), callbacks=[scheduler])
```

::: {.output .stream .stdout}
    Epoch 1/30
    3198/3198 [==============================] - 85s 25ms/step - loss: 0.6946 - accuracy: 0.5080 - val_loss: 0.6934 - val_accuracy: 0.5066 - lr: 1.0000e-04
    Epoch 2/30
    3198/3198 [==============================] - 80s 25ms/step - loss: 0.6925 - accuracy: 0.5163 - val_loss: 0.6920 - val_accuracy: 0.5255 - lr: 9.0000e-05
    Epoch 3/30
    3198/3198 [==============================] - 76s 24ms/step - loss: 0.6913 - accuracy: 0.5250 - val_loss: 0.6912 - val_accuracy: 0.5251 - lr: 8.1000e-05
    Epoch 4/30
    3198/3198 [==============================] - 76s 24ms/step - loss: 0.6908 - accuracy: 0.5275 - val_loss: 0.6906 - val_accuracy: 0.5314 - lr: 7.2900e-05
    Epoch 5/30
    3198/3198 [==============================] - 76s 24ms/step - loss: 0.6906 - accuracy: 0.5283 - val_loss: 0.6906 - val_accuracy: 0.5313 - lr: 6.5610e-05
    Epoch 6/30
    3198/3198 [==============================] - 77s 24ms/step - loss: 0.6903 - accuracy: 0.5291 - val_loss: 0.6909 - val_accuracy: 0.5284 - lr: 5.9049e-05
    Epoch 7/30
    3198/3198 [==============================] - 77s 24ms/step - loss: 0.6906 - accuracy: 0.5282 - val_loss: 0.6908 - val_accuracy: 0.5314 - lr: 5.3144e-05
    Epoch 8/30
    3198/3198 [==============================] - 76s 24ms/step - loss: 0.6902 - accuracy: 0.5303 - val_loss: 0.6908 - val_accuracy: 0.5283 - lr: 4.7830e-05
    Epoch 9/30
    3198/3198 [==============================] - 78s 24ms/step - loss: 0.6901 - accuracy: 0.5307 - val_loss: 0.6909 - val_accuracy: 0.5325 - lr: 4.3047e-05
    Epoch 10/30
    3198/3198 [==============================] - 82s 25ms/step - loss: 0.6900 - accuracy: 0.5306 - val_loss: 0.6906 - val_accuracy: 0.5314 - lr: 3.8742e-05
    Epoch 11/30
    3198/3198 [==============================] - 93s 29ms/step - loss: 0.6901 - accuracy: 0.5304 - val_loss: 0.6920 - val_accuracy: 0.5231 - lr: 3.4868e-05
    Epoch 12/30
    3198/3198 [==============================] - 112s 35ms/step - loss: 0.6899 - accuracy: 0.5319 - val_loss: 0.6907 - val_accuracy: 0.5258 - lr: 3.1381e-05
    Epoch 13/30
    3198/3198 [==============================] - 105s 33ms/step - loss: 0.6900 - accuracy: 0.5318 - val_loss: 0.6910 - val_accuracy: 0.5260 - lr: 2.8243e-05
    Epoch 14/30
    3198/3198 [==============================] - 93s 29ms/step - loss: 0.6898 - accuracy: 0.5323 - val_loss: 0.6907 - val_accuracy: 0.5278 - lr: 2.5419e-05
    Epoch 15/30
    3198/3198 [==============================] - 80s 25ms/step - loss: 0.6896 - accuracy: 0.5329 - val_loss: 0.6907 - val_accuracy: 0.5316 - lr: 2.2877e-05
    Epoch 16/30
    3198/3198 [==============================] - 80s 25ms/step - loss: 0.6897 - accuracy: 0.5324 - val_loss: 0.6908 - val_accuracy: 0.5299 - lr: 2.0589e-05
    Epoch 17/30
    3198/3198 [==============================] - 80s 25ms/step - loss: 0.6897 - accuracy: 0.5324 - val_loss: 0.6907 - val_accuracy: 0.5280 - lr: 1.8530e-05
    Epoch 18/30
    3198/3198 [==============================] - 81s 25ms/step - loss: 0.6898 - accuracy: 0.5333 - val_loss: 0.6910 - val_accuracy: 0.5258 - lr: 1.6677e-05
    Epoch 19/30
    3198/3198 [==============================] - 80s 25ms/step - loss: 0.6896 - accuracy: 0.5351 - val_loss: 0.6907 - val_accuracy: 0.5276 - lr: 1.5009e-05
    Epoch 20/30
    3198/3198 [==============================] - 80s 25ms/step - loss: 0.6897 - accuracy: 0.5318 - val_loss: 0.6908 - val_accuracy: 0.5292 - lr: 1.3509e-05
    Epoch 21/30
    3198/3198 [==============================] - 81s 25ms/step - loss: 0.6896 - accuracy: 0.5331 - val_loss: 0.6908 - val_accuracy: 0.5277 - lr: 1.2158e-05
    Epoch 22/30
    3198/3198 [==============================] - 81s 25ms/step - loss: 0.6897 - accuracy: 0.5329 - val_loss: 0.6909 - val_accuracy: 0.5299 - lr: 1.0942e-05
    Epoch 23/30
    3198/3198 [==============================] - 81s 25ms/step - loss: 0.6893 - accuracy: 0.5338 - val_loss: 0.6911 - val_accuracy: 0.5277 - lr: 9.8477e-06
    Epoch 24/30
    3198/3198 [==============================] - 81s 25ms/step - loss: 0.6894 - accuracy: 0.5342 - val_loss: 0.6910 - val_accuracy: 0.5290 - lr: 8.8629e-06
    Epoch 25/30
    3198/3198 [==============================] - 82s 26ms/step - loss: 0.6893 - accuracy: 0.5338 - val_loss: 0.6909 - val_accuracy: 0.5316 - lr: 7.9766e-06
    Epoch 26/30
    3198/3198 [==============================] - 80s 25ms/step - loss: 0.6894 - accuracy: 0.5329 - val_loss: 0.6909 - val_accuracy: 0.5297 - lr: 7.1790e-06
    Epoch 27/30
    3198/3198 [==============================] - 81s 25ms/step - loss: 0.6897 - accuracy: 0.5330 - val_loss: 0.6909 - val_accuracy: 0.5313 - lr: 6.4611e-06
    Epoch 28/30
    3198/3198 [==============================] - 81s 25ms/step - loss: 0.6893 - accuracy: 0.5335 - val_loss: 0.6910 - val_accuracy: 0.5309 - lr: 5.8150e-06
    Epoch 29/30
    3198/3198 [==============================] - 81s 25ms/step - loss: 0.6897 - accuracy: 0.5324 - val_loss: 0.6907 - val_accuracy: 0.5288 - lr: 5.2335e-06
    Epoch 30/30
    3198/3198 [==============================] - 81s 25ms/step - loss: 0.6895 - accuracy: 0.5333 - val_loss: 0.6907 - val_accuracy: 0.5294 - lr: 4.7101e-06
:::
:::

::: {.cell .code execution_count="29"}
``` python
stats.plot_history(deeper_emb_hist)
```

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/198baba17e2c65a0a4169c7e2d0a332d5bf5416d.png)
:::

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/e8f1b5b161c5982e251fe143b1c2d52f6ff7f873.png)
:::
:::

::: {.cell .markdown}
We\'ve tested a few different hyperparameters like learning rates, batch
sizes, epochs, number of layers and layer sizes, but overall the model
does not perform better compared to the smaller one and just starts
overfitting after more training. We will move on for now and test
different architectures that are more specialized to the specific task,
instead of just using a bunch of dense layers.

## Convolution Model

This model consists of a set amount of layers that are stacked after
another. In contrast to the previous model, this model tries to compare
the individual inputs, the 10 different champions, to each other instead
of just processing everything at once with dense layers. To do this, we
are using a conv1d layer after a simple linear layer to hopefully learn
more general relationships.

In addition, this model uses two embedding layers instead of one. As
before, the first one is for the champions, but the second one is for
the team, which is 0 for when the champion index is 0, 1 for the blue
side and 2 for the red side. These embeddings get added to one tensor
and then passed through the layer. The idea behind this is that the
relationship between champions depends on if they are on the same or on
different teams, and with this technique the model could learn to treat
them differently, while still being based on the same champion. The
dense models before didn\'t have this requirement, since every layer was
densely connected with the previous one, which means that the order of
the embeddings also included the team.
:::

::: {.cell .code execution_count="32"}
``` python
# a model with multiple layers that consist of conv1d, max pool, dense layers and residual connections
class DeepConv(tf.keras.Model):
    def __init__(self, champ_num=170, emb_dim=32, n_layers=3):
        super().__init__()
        player_num = 10
        self.n_layers = n_layers
        self.player_num = player_num
        self.champ_num = champ_num
        self.emb_dim = emb_dim
        self.embedding = tf.keras.layers.Embedding(champ_num, emb_dim, input_length=player_num)
        #self.expand = tf.keras.layers.Dense(emb_dim*player_num, activation=None)
        self.team_embedding = tf.keras.layers.Embedding(3, emb_dim, input_length=player_num)
        self.deep_layers = []
        self.layer_norm = tf.keras.layers.LayerNormalization()
        for i in range(n_layers):
            layer = []
            lname = "_layer_" + str(i)
            tf.keras.layers.Reshape((-1, player_num*emb_dim))
            layer.append(tf.keras.layers.Dense(emb_dim*player_num*player_num, activation=None, name="expand"+lname))
            tf.keras.layers.Reshape((-1, player_num*emb_dim, player_num))
            layer.append(tf.keras.layers.Conv1D(emb_dim, 5, strides=5, padding="same", activation='gelu', name="conv1d"+lname))
            layer.append(tf.keras.layers.MaxPool1D(2, name="maxpool"+lname))
            
            layer.append(tf.keras.layers.Dense(emb_dim, activation='gelu', name="dense"+lname))

            self.deep_layers.append(layer)
            

        #self.maxpool = tf.keras.layers.MaxPool1D(10)
        self.flatten = tf.keras.layers.Flatten()


        self.dense1 = tf.keras.layers.Dense(emb_dim*6, activation='gelu')
        self.dense2 = tf.keras.layers.Dense(emb_dim*3, activation='gelu')
        self.out = tf.keras.layers.Dense(1, activation='sigmoid')





    def call(self, inputs):
        # mask for empty slots
        mask = tf.where(inputs == 0, 0, 1)
        x = self.embedding(inputs)

        # team embedding
        team_vals = [1,1,1,1,1,2,2,2,2,2]
        team_vals = tf.convert_to_tensor(team_vals)
        team_vals = tf.reshape(team_vals, (-1, 10))
        team_vals = team_vals * mask
        team_vals = tf.cast(team_vals, tf.int32)
        team_vals = self.team_embedding(team_vals)

        # add team embedding to champion embedding
        x = x + team_vals

        for layer in self.deep_layers:
            xl = layer[0](x)
            for i in range(1, len(layer)):
                xl = layer[i](xl)

            # residual connection
            x+=xl
            x = self.layer_norm(x)


        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        return x
```
:::

::: {.cell .code execution_count="33"}
``` python
aug = augmentation.MatchAugmentation(train_filtered_x, train_filtered_y, aug_chance=0.90, batch_size=64, max_replace=2)

deep_conv_model = DeepConv()
deep_conv_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

scheduler = create_scheduler(lambda epoch: 0.0001 * 0.90**epoch, lambda epoch: 8*(2**(epoch//5)))
deeper_conv_hist = deep_conv_model.fit(aug, epochs=30, validation_data=(val_x, val_y), callbacks=[scheduler])
```

::: {.output .stream .stdout}
    Epoch 1/30
    3198/3198 [==============================] - 60s 18ms/step - loss: 0.6938 - accuracy: 0.5073 - val_loss: 0.6931 - val_accuracy: 0.5072 - lr: 1.0000e-04
    Epoch 2/30
    3198/3198 [==============================] - 58s 18ms/step - loss: 0.6930 - accuracy: 0.5117 - val_loss: 0.6930 - val_accuracy: 0.5074 - lr: 9.0000e-05
    Epoch 3/30
    3198/3198 [==============================] - 58s 18ms/step - loss: 0.6920 - accuracy: 0.5200 - val_loss: 0.6915 - val_accuracy: 0.5252 - lr: 8.1000e-05
    Epoch 4/30
    3198/3198 [==============================] - 58s 18ms/step - loss: 0.6914 - accuracy: 0.5251 - val_loss: 0.6919 - val_accuracy: 0.5261 - lr: 7.2900e-05
    Epoch 5/30
    3198/3198 [==============================] - 58s 18ms/step - loss: 0.6908 - accuracy: 0.5275 - val_loss: 0.6924 - val_accuracy: 0.5214 - lr: 6.5610e-05
    Epoch 6/30
    3198/3198 [==============================] - 57s 18ms/step - loss: 0.6906 - accuracy: 0.5289 - val_loss: 0.6910 - val_accuracy: 0.5281 - lr: 5.9049e-05
    Epoch 7/30
    3198/3198 [==============================] - 57s 18ms/step - loss: 0.6903 - accuracy: 0.5302 - val_loss: 0.6913 - val_accuracy: 0.5294 - lr: 5.3144e-05
    Epoch 8/30
    3198/3198 [==============================] - 58s 18ms/step - loss: 0.6903 - accuracy: 0.5310 - val_loss: 0.6906 - val_accuracy: 0.5289 - lr: 4.7830e-05
    Epoch 9/30
    3198/3198 [==============================] - 57s 18ms/step - loss: 0.6902 - accuracy: 0.5312 - val_loss: 0.6917 - val_accuracy: 0.5353 - lr: 4.3047e-05
    Epoch 10/30
    3198/3198 [==============================] - 58s 18ms/step - loss: 0.6904 - accuracy: 0.5293 - val_loss: 0.6907 - val_accuracy: 0.5329 - lr: 3.8742e-05
    Epoch 11/30
    3198/3198 [==============================] - 57s 18ms/step - loss: 0.6901 - accuracy: 0.5317 - val_loss: 0.6904 - val_accuracy: 0.5336 - lr: 3.4868e-05
    Epoch 12/30
    3198/3198 [==============================] - 57s 18ms/step - loss: 0.6901 - accuracy: 0.5309 - val_loss: 0.6903 - val_accuracy: 0.5329 - lr: 3.1381e-05
    Epoch 13/30
    3198/3198 [==============================] - 58s 18ms/step - loss: 0.6900 - accuracy: 0.5322 - val_loss: 0.6906 - val_accuracy: 0.5316 - lr: 2.8243e-05
    Epoch 14/30
    3198/3198 [==============================] - 66s 21ms/step - loss: 0.6902 - accuracy: 0.5308 - val_loss: 0.6905 - val_accuracy: 0.5328 - lr: 2.5419e-05
    Epoch 15/30
    3198/3198 [==============================] - 71s 22ms/step - loss: 0.6901 - accuracy: 0.5310 - val_loss: 0.6909 - val_accuracy: 0.5326 - lr: 2.2877e-05
    Epoch 16/30
    3198/3198 [==============================] - 57s 18ms/step - loss: 0.6900 - accuracy: 0.5311 - val_loss: 0.6909 - val_accuracy: 0.5325 - lr: 2.0589e-05
    Epoch 17/30
    3198/3198 [==============================] - 57s 18ms/step - loss: 0.6899 - accuracy: 0.5313 - val_loss: 0.6906 - val_accuracy: 0.5314 - lr: 1.8530e-05
    Epoch 18/30
    3198/3198 [==============================] - 58s 18ms/step - loss: 0.6899 - accuracy: 0.5320 - val_loss: 0.6907 - val_accuracy: 0.5320 - lr: 1.6677e-05
    Epoch 19/30
    3198/3198 [==============================] - 57s 18ms/step - loss: 0.6897 - accuracy: 0.5329 - val_loss: 0.6906 - val_accuracy: 0.5299 - lr: 1.5009e-05
    Epoch 20/30
    3198/3198 [==============================] - 55s 17ms/step - loss: 0.6898 - accuracy: 0.5317 - val_loss: 0.6907 - val_accuracy: 0.5295 - lr: 1.3509e-05
    Epoch 21/30
    3198/3198 [==============================] - 58s 18ms/step - loss: 0.6897 - accuracy: 0.5320 - val_loss: 0.6906 - val_accuracy: 0.5294 - lr: 1.2158e-05
    Epoch 22/30
    3198/3198 [==============================] - 57s 18ms/step - loss: 0.6896 - accuracy: 0.5341 - val_loss: 0.6909 - val_accuracy: 0.5298 - lr: 1.0942e-05
    Epoch 23/30
    3198/3198 [==============================] - 56s 18ms/step - loss: 0.6896 - accuracy: 0.5329 - val_loss: 0.6907 - val_accuracy: 0.5303 - lr: 9.8477e-06
    Epoch 24/30
    3198/3198 [==============================] - 56s 18ms/step - loss: 0.6897 - accuracy: 0.5325 - val_loss: 0.6906 - val_accuracy: 0.5299 - lr: 8.8629e-06
    Epoch 25/30
    3198/3198 [==============================] - 55s 17ms/step - loss: 0.6896 - accuracy: 0.5334 - val_loss: 0.6911 - val_accuracy: 0.5292 - lr: 7.9766e-06
    Epoch 26/30
    3198/3198 [==============================] - 57s 18ms/step - loss: 0.6897 - accuracy: 0.5319 - val_loss: 0.6906 - val_accuracy: 0.5310 - lr: 7.1790e-06
    Epoch 27/30
    3198/3198 [==============================] - 57s 18ms/step - loss: 0.6898 - accuracy: 0.5320 - val_loss: 0.6908 - val_accuracy: 0.5280 - lr: 6.4611e-06
    Epoch 28/30
    3198/3198 [==============================] - 58s 18ms/step - loss: 0.6897 - accuracy: 0.5332 - val_loss: 0.6906 - val_accuracy: 0.5283 - lr: 5.8150e-06
    Epoch 29/30
    3198/3198 [==============================] - 55s 17ms/step - loss: 0.6897 - accuracy: 0.5315 - val_loss: 0.6908 - val_accuracy: 0.5278 - lr: 5.2335e-06
    Epoch 30/30
    3198/3198 [==============================] - 57s 18ms/step - loss: 0.6897 - accuracy: 0.5334 - val_loss: 0.6907 - val_accuracy: 0.5278 - lr: 4.7101e-06
:::
:::

::: {.cell .code execution_count="35"}
``` python
stats.plot_history(deeper_conv_hist)
```

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/e38c99b62600758892b5efff84dedacd611f770a.png)
:::

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/dfe6507d77876a6bb645236e033de6c0fceac29d.png)
:::
:::

::: {.cell .markdown}
## Attention Mechanism

Since the goal is to learn the relationship between the 10 champions,
similar to how a language model tries to learn the relationship between
tokens in a sequence, we will try to create a model that uses multi head
attention to predict the outcome of games.

The model will start with champions and team embeddings like the
previous model. In a series of attention blocks, we will use masked
multi head attention to transform the embedding sequence. The mask is
used where the input index is 0, which means that the champion hasn\'t
been selected yet. Inside the attention block we will also apply a
residual connection with layer norm to improve the learning performance
of deeper models.
:::

::: {.cell .code execution_count="41"}
``` python
# a single transformer block that can have multiple heads
class LoLTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, embedding_dim):
        super(LoLTransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        self.flatten = tf.keras.layers.Flatten()

        # multi-head attention
        self.query = tf.keras.layers.Dense(self.embedding_dim*num_heads, input_shape=(self.embedding_dim,), activation=None)
        self.key = tf.keras.layers.Dense(self.embedding_dim*num_heads, input_shape=(self.embedding_dim,), activation=None)
        self.value = tf.keras.layers.Dense(self.embedding_dim*num_heads, input_shape=(self.embedding_dim,), activation=None)

        self.sqrt_d = tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))


        self.layernorm1 = tf.keras.layers.LayerNormalization()

        self.combineHeads = tf.keras.layers.Dense(self.embedding_dim, activation=None)

        self.nonLin = tf.keras.layers.Dense(self.embedding_dim, input_shape=(self.embedding_dim,), activation='gelu')

    def split_heads(self, x):
        # x.shape = (batch_size, 10, embedding_dim)
        # split into qvk for each head
        q = tf.reshape(self.query(x), (-1, self.num_heads, 10, self.embedding_dim))
        k = tf.reshape(self.key(x), (-1, self.num_heads, 10, self.embedding_dim))
        v = tf.reshape(self.value(x), (-1, self.num_heads, 10, self.embedding_dim))

        return q, k, v
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # perform self attention for each head
        # q, k, v shape = (batch_size, num_heads, 10, embedding_dim)

        # calculate attention weights
        # q shape = (batch_size, num_heads, 10, 1, embedding_dim)
        # k shape = (batch_size, num_heads, 1, 10, embedding_dim)
        attention_weights = tf.matmul(q, k, transpose_b=True) / self.sqrt_d
        # shape = (batch_size, num_heads, 10, 10)

        # apply mask if not None
        if mask != None:
            #mask = mask.astype('float32')
            mask = tf.reshape(mask, (-1, 1, 1, 10))
            attention_weights+= mask  * -1e9

        # softmax
        attention_weights = tf.nn.softmax(attention_weights)
        # shape = (batch_size, num_heads, 10, 10)

        # apply attention weights to values
        # v shape = (batch_size, num_heads, 10, embedding_dim)
        output = tf.matmul(attention_weights, v)
        # shape = (batch_size, num_heads, 10, embedding_dim)
        return output

    def combine_heads(self, x):
        # x.shape = (batch_size, num_heads, 10, embedding_dim)
        # flatten before combination layer
        x = tf.transpose(x, perm=[0,2,1,3])
        # x.shape = (batch_size, 10, num_heads, embedding_dim)
        x = tf.reshape(x, ((-1, 10, self.num_heads*self.embedding_dim)))
        x = self.combineHeads(x)
        # x.shape = (batch_size, 10,embedding_dim)
        return x

    def call(self, x, mask=None):
        # x.shape = (batch_size, 10, embedding_dim)
        x_original = x
        
        # multi-head attention
        q, k, v = self.split_heads(x)
        # q, k, v shape = (batch_size, num_heads, 10, embedding_dim)
        x = self.scaled_dot_product_attention(q, k, v, mask=mask)
        # x shape = (batch_size, num_heads, 10, embedding_dim)
        x = self.combine_heads(x)
        # x.shape = (batch_size, 10, embedding_dim)

        # residual connection
        x = x + x_original
        x = self.layernorm1(x)
        # x.shape = (batch_size, 10, embedding_dim)

        # feed forward
        x = self.nonLin(x)
        # x.shape = (batch_size, 10, embedding_dim)

        # residual connection
        x = x + x_original
        x = self.layernorm1(x)

        return x

# base model consisting of the embeddings, multiple transformer blocks and some dense layers at the end
class LoLTransformer(tf.keras.Model):
    def __init__(self, num_layers=4, num_heads=4, embedding_dim=32, champ_vocab_size=170):
        super(LoLTransformer, self).__init__()

        self.embedding_dim = embedding_dim
        self.champ_vocab_size = champ_vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        # embedding layers
        self.champ_embedding = tf.keras.layers.Embedding(self.champ_vocab_size, self.embedding_dim)
        self.team_embedding = tf.keras.layers.Embedding(2, self.embedding_dim)

        # transformer layers
        self.transformer_layers = [LoLTransformerBlock(self.num_heads, self.embedding_dim) for _ in range(self.num_layers)]

        # output layers
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(10*self.embedding_dim,))



    def call(self, x):
        # x.shape = (batch_size, 10), where each element is a champion index

        mask = tf.where(x == 0, 1.0, 0.0)

        # get embeddings
        x = self.champ_embedding(x)
        # x.shape = (batch_size, 10, embedding_dim)
        # add team embeddings, 0-4 are team 1, 5-9 are team 2
        t = tf.concat([self.team_embedding(tf.zeros((1,5))), self.team_embedding(tf.ones((1,5)))], axis=1)
        x = x + t
        # x.shape = (batch_size, 10, embedding_dim)
        
        # transformer layers
        for transformer in self.transformer_layers:
            x = transformer(x, mask=mask)
        # x.shape = (batch_size, 10, embedding_dim)

        # output layers
        x = tf.reshape(x, (-1, 10*self.embedding_dim))
        x = self.dense(x)
        # x.shape = (batch_size, 1)
        return x
```
:::

::: {.cell .code execution_count="42"}
``` python
aug = augmentation.MatchAugmentation(train_filtered_x, train_filtered_y, aug_chance=0.90, batch_size=64, max_replace=2)

lol_transformer = LoLTransformer()
lol_transformer.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

scheduler = create_scheduler(lambda epoch: 0.0001 * 0.90**epoch, lambda epoch: 8*(2**(epoch//5)))
lol_transformer_hist = lol_transformer.fit(aug, epochs=20, validation_data=(val_x, val_y), callbacks=[scheduler])
```

::: {.output .stream .stdout}
    Epoch 1/30
    3198/3198 [==============================] - 56s 17ms/step - loss: 0.6991 - accuracy: 0.5039 - val_loss: 0.6933 - val_accuracy: 0.5101 - lr: 1.0000e-04
    Epoch 2/30
    3198/3198 [==============================] - 53s 16ms/step - loss: 0.6927 - accuracy: 0.5137 - val_loss: 0.6917 - val_accuracy: 0.5272 - lr: 9.0000e-05
    Epoch 3/30
    3198/3198 [==============================] - 53s 17ms/step - loss: 0.6913 - accuracy: 0.5234 - val_loss: 0.6908 - val_accuracy: 0.5281 - lr: 8.1000e-05
    Epoch 4/30
    3198/3198 [==============================] - 53s 17ms/step - loss: 0.6908 - accuracy: 0.5257 - val_loss: 0.6911 - val_accuracy: 0.5241 - lr: 7.2900e-05
    Epoch 5/30
    3198/3198 [==============================] - 54s 17ms/step - loss: 0.6906 - accuracy: 0.5280 - val_loss: 0.6907 - val_accuracy: 0.5326 - lr: 6.5610e-05
    Epoch 6/30
    3198/3198 [==============================] - 53s 17ms/step - loss: 0.6904 - accuracy: 0.5295 - val_loss: 0.6906 - val_accuracy: 0.5346 - lr: 5.9049e-05
    Epoch 7/30
    3198/3198 [==============================] - 53s 17ms/step - loss: 0.6903 - accuracy: 0.5279 - val_loss: 0.6910 - val_accuracy: 0.5315 - lr: 5.3144e-05
    Epoch 8/30
    3198/3198 [==============================] - 53s 17ms/step - loss: 0.6904 - accuracy: 0.5295 - val_loss: 0.6906 - val_accuracy: 0.5312 - lr: 4.7830e-05
    Epoch 9/30
    3198/3198 [==============================] - 54s 17ms/step - loss: 0.6902 - accuracy: 0.5298 - val_loss: 0.6912 - val_accuracy: 0.5290 - lr: 4.3047e-05
    Epoch 10/30
    3198/3198 [==============================] - 54s 17ms/step - loss: 0.6902 - accuracy: 0.5289 - val_loss: 0.6907 - val_accuracy: 0.5333 - lr: 3.8742e-05
    Epoch 11/30
    3198/3198 [==============================] - 54s 17ms/step - loss: 0.6901 - accuracy: 0.5304 - val_loss: 0.6904 - val_accuracy: 0.5311 - lr: 3.4868e-05
    Epoch 12/30
    3198/3198 [==============================] - 54s 17ms/step - loss: 0.6901 - accuracy: 0.5302 - val_loss: 0.6905 - val_accuracy: 0.5312 - lr: 3.1381e-05
    Epoch 13/30
    3198/3198 [==============================] - 53s 17ms/step - loss: 0.6900 - accuracy: 0.5306 - val_loss: 0.6905 - val_accuracy: 0.5343 - lr: 2.8243e-05
    Epoch 14/30
    3198/3198 [==============================] - 51s 16ms/step - loss: 0.6897 - accuracy: 0.5321 - val_loss: 0.6907 - val_accuracy: 0.5328 - lr: 2.5419e-05
    Epoch 15/30
    3198/3198 [==============================] - 52s 16ms/step - loss: 0.6899 - accuracy: 0.5309 - val_loss: 0.6903 - val_accuracy: 0.5315 - lr: 2.2877e-05
    Epoch 16/30
    3198/3198 [==============================] - 52s 16ms/step - loss: 0.6899 - accuracy: 0.5303 - val_loss: 0.6908 - val_accuracy: 0.5305 - lr: 2.0589e-05
    Epoch 17/30
    3198/3198 [==============================] - 54s 17ms/step - loss: 0.6898 - accuracy: 0.5315 - val_loss: 0.6904 - val_accuracy: 0.5330 - lr: 1.8530e-05
    Epoch 18/30
    3198/3198 [==============================] - 54s 17ms/step - loss: 0.6899 - accuracy: 0.5317 - val_loss: 0.6905 - val_accuracy: 0.5303 - lr: 1.6677e-05
    Epoch 19/30
    3198/3198 [==============================] - 52s 16ms/step - loss: 0.6899 - accuracy: 0.5315 - val_loss: 0.6904 - val_accuracy: 0.5317 - lr: 1.5009e-05
    Epoch 20/30
    3198/3198 [==============================] - 52s 16ms/step - loss: 0.6900 - accuracy: 0.5321 - val_loss: 0.6905 - val_accuracy: 0.5319 - lr: 1.3509e-05
    Epoch 21/30
    3198/3198 [==============================] - 55s 17ms/step - loss: 0.6898 - accuracy: 0.5323 - val_loss: 0.6904 - val_accuracy: 0.5320 - lr: 1.2158e-05
    Epoch 22/30
    3198/3198 [==============================] - 54s 17ms/step - loss: 0.6898 - accuracy: 0.5315 - val_loss: 0.6906 - val_accuracy: 0.5322 - lr: 1.0942e-05
    Epoch 23/30
    3198/3198 [==============================] - 54s 17ms/step - loss: 0.6897 - accuracy: 0.5327 - val_loss: 0.6906 - val_accuracy: 0.5332 - lr: 9.8477e-06
    Epoch 24/30
    3198/3198 [==============================] - 54s 17ms/step - loss: 0.6898 - accuracy: 0.5324 - val_loss: 0.6904 - val_accuracy: 0.5309 - lr: 8.8629e-06
    Epoch 25/30
    3198/3198 [==============================] - 54s 17ms/step - loss: 0.6898 - accuracy: 0.5323 - val_loss: 0.6905 - val_accuracy: 0.5316 - lr: 7.9766e-06
    Epoch 26/30
    3198/3198 [==============================] - 54s 17ms/step - loss: 0.6897 - accuracy: 0.5320 - val_loss: 0.6905 - val_accuracy: 0.5311 - lr: 7.1790e-06
    Epoch 27/30
    3198/3198 [==============================] - 54s 17ms/step - loss: 0.6897 - accuracy: 0.5321 - val_loss: 0.6905 - val_accuracy: 0.5320 - lr: 6.4611e-06
    Epoch 28/30
    3198/3198 [==============================] - 54s 17ms/step - loss: 0.6899 - accuracy: 0.5319 - val_loss: 0.6905 - val_accuracy: 0.5323 - lr: 5.8150e-06
    Epoch 29/30
    3198/3198 [==============================] - 54s 17ms/step - loss: 0.6897 - accuracy: 0.5328 - val_loss: 0.6906 - val_accuracy: 0.5297 - lr: 5.2335e-06
    Epoch 30/30
    3198/3198 [==============================] - 54s 17ms/step - loss: 0.6895 - accuracy: 0.5334 - val_loss: 0.6906 - val_accuracy: 0.5305 - lr: 4.7101e-06
:::
:::

::: {.cell .code execution_count="43"}
``` python
stats.plot_history(lol_transformer_hist)
```

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/f53f40dfdfea7efc185a25f12e8e0c63081eddd9.png)
:::

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/24689e1209604d6813387c0d32ac62a18bea4408.png)
:::
:::

::: {.cell .code execution_count="44"}
``` python
aug = augmentation.MatchAugmentation(train_filtered_x, train_filtered_y, aug_chance=0.95, batch_size=32, max_replace=2)

lol_transformer_8_8 = LoLTransformer(num_layers=8, num_heads=8)
lol_transformer_8_8.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

scheduler = create_scheduler(lambda epoch: 0.0001 * 0.96**epoch, lambda epoch: 8*(2**(epoch//5)))
lol_transformer_8_8_hist = lol_transformer_8_8.fit(aug, epochs=20, validation_data=(val_x, val_y), callbacks=[scheduler])
```

::: {.output .stream .stdout}
    Epoch 1/40
    6397/6397 [==============================] - 183s 28ms/step - loss: 0.6976 - accuracy: 0.5083 - val_loss: 0.6929 - val_accuracy: 0.5135 - lr: 1.0000e-04
    Epoch 2/40
    6397/6397 [==============================] - 182s 28ms/step - loss: 0.6923 - accuracy: 0.5197 - val_loss: 0.6921 - val_accuracy: 0.5164 - lr: 9.6000e-05
    Epoch 3/40
    6397/6397 [==============================] - 179s 28ms/step - loss: 0.6915 - accuracy: 0.5231 - val_loss: 0.6907 - val_accuracy: 0.5262 - lr: 9.2160e-05
    Epoch 4/40
    6397/6397 [==============================] - 179s 28ms/step - loss: 0.6911 - accuracy: 0.5269 - val_loss: 0.6919 - val_accuracy: 0.5242 - lr: 8.8474e-05
    Epoch 5/40
    6397/6397 [==============================] - 180s 28ms/step - loss: 0.6907 - accuracy: 0.5281 - val_loss: 0.6911 - val_accuracy: 0.5277 - lr: 8.4935e-05
    Epoch 6/40
    6397/6397 [==============================] - 181s 28ms/step - loss: 0.6909 - accuracy: 0.5266 - val_loss: 0.6915 - val_accuracy: 0.5250 - lr: 8.1537e-05
    Epoch 7/40
    6397/6397 [==============================] - 181s 28ms/step - loss: 0.6908 - accuracy: 0.5276 - val_loss: 0.6906 - val_accuracy: 0.5286 - lr: 7.8276e-05
    Epoch 8/40
    6397/6397 [==============================] - 180s 28ms/step - loss: 0.6907 - accuracy: 0.5278 - val_loss: 0.6908 - val_accuracy: 0.5282 - lr: 7.5145e-05
    Epoch 9/40
    6397/6397 [==============================] - 179s 28ms/step - loss: 0.6905 - accuracy: 0.5287 - val_loss: 0.6908 - val_accuracy: 0.5319 - lr: 7.2139e-05
    Epoch 10/40
    6397/6397 [==============================] - 180s 28ms/step - loss: 0.6904 - accuracy: 0.5284 - val_loss: 0.6904 - val_accuracy: 0.5325 - lr: 6.9253e-05
    Epoch 11/40
    6397/6397 [==============================] - 180s 28ms/step - loss: 0.6904 - accuracy: 0.5283 - val_loss: 0.6903 - val_accuracy: 0.5295 - lr: 6.6483e-05
    Epoch 12/40
    6397/6397 [==============================] - 180s 28ms/step - loss: 0.6904 - accuracy: 0.5291 - val_loss: 0.6904 - val_accuracy: 0.5311 - lr: 6.3824e-05
    Epoch 13/40
    6397/6397 [==============================] - 181s 28ms/step - loss: 0.6904 - accuracy: 0.5273 - val_loss: 0.6909 - val_accuracy: 0.5236 - lr: 6.1271e-05
    Epoch 14/40
    6397/6397 [==============================] - 181s 28ms/step - loss: 0.6902 - accuracy: 0.5309 - val_loss: 0.6904 - val_accuracy: 0.5282 - lr: 5.8820e-05
    Epoch 15/40
    6397/6397 [==============================] - 181s 28ms/step - loss: 0.6902 - accuracy: 0.5304 - val_loss: 0.6903 - val_accuracy: 0.5286 - lr: 5.6467e-05
    Epoch 16/40
    6397/6397 [==============================] - 182s 28ms/step - loss: 0.6902 - accuracy: 0.5305 - val_loss: 0.6904 - val_accuracy: 0.5276 - lr: 5.4209e-05
    Epoch 17/40
    6397/6397 [==============================] - 182s 28ms/step - loss: 0.6900 - accuracy: 0.5310 - val_loss: 0.6909 - val_accuracy: 0.5290 - lr: 5.2040e-05
    Epoch 18/40
    6397/6397 [==============================] - 182s 28ms/step - loss: 0.6899 - accuracy: 0.5302 - val_loss: 0.6905 - val_accuracy: 0.5300 - lr: 4.9959e-05
    Epoch 19/40
    6397/6397 [==============================] - 181s 28ms/step - loss: 0.6901 - accuracy: 0.5305 - val_loss: 0.6905 - val_accuracy: 0.5282 - lr: 4.7960e-05
    Epoch 20/40
    6397/6397 [==============================] - 182s 28ms/step - loss: 0.6899 - accuracy: 0.5306 - val_loss: 0.6906 - val_accuracy: 0.5311 - lr: 4.6042e-05
    Epoch 21/40
    6397/6397 [==============================] - 181s 28ms/step - loss: 0.6899 - accuracy: 0.5310 - val_loss: 0.6906 - val_accuracy: 0.5312 - lr: 4.4200e-05
    Epoch 22/40
    6397/6397 [==============================] - 181s 28ms/step - loss: 0.6899 - accuracy: 0.5306 - val_loss: 0.6904 - val_accuracy: 0.5267 - lr: 4.2432e-05
    Epoch 23/40
    6397/6397 [==============================] - 182s 28ms/step - loss: 0.6900 - accuracy: 0.5311 - val_loss: 0.6908 - val_accuracy: 0.5266 - lr: 4.0735e-05
    Epoch 24/40
    6397/6397 [==============================] - 184s 29ms/step - loss: 0.6897 - accuracy: 0.5318 - val_loss: 0.6903 - val_accuracy: 0.5309 - lr: 3.9106e-05
    Epoch 25/40
    6397/6397 [==============================] - 184s 29ms/step - loss: 0.6897 - accuracy: 0.5320 - val_loss: 0.6905 - val_accuracy: 0.5326 - lr: 3.7541e-05
    Epoch 26/40
    6397/6397 [==============================] - 184s 29ms/step - loss: 0.6895 - accuracy: 0.5320 - val_loss: 0.6913 - val_accuracy: 0.5254 - lr: 3.6040e-05
    Epoch 27/40
    6397/6397 [==============================] - 184s 29ms/step - loss: 0.6899 - accuracy: 0.5302 - val_loss: 0.6906 - val_accuracy: 0.5266 - lr: 3.4598e-05
    Epoch 28/40
    6397/6397 [==============================] - 185s 29ms/step - loss: 0.6898 - accuracy: 0.5315 - val_loss: 0.6905 - val_accuracy: 0.5292 - lr: 3.3214e-05
    Epoch 29/40
    6397/6397 [==============================] - 186s 29ms/step - loss: 0.6895 - accuracy: 0.5336 - val_loss: 0.6914 - val_accuracy: 0.5258 - lr: 3.1886e-05
    Epoch 30/40
    6397/6397 [==============================] - 185s 29ms/step - loss: 0.6895 - accuracy: 0.5326 - val_loss: 0.6908 - val_accuracy: 0.5287 - lr: 3.0610e-05
    Epoch 31/40
    6397/6397 [==============================] - 186s 29ms/step - loss: 0.6896 - accuracy: 0.5313 - val_loss: 0.6907 - val_accuracy: 0.5283 - lr: 2.9386e-05
    Epoch 32/40
    6397/6397 [==============================] - 185s 29ms/step - loss: 0.6894 - accuracy: 0.5324 - val_loss: 0.6905 - val_accuracy: 0.5264 - lr: 2.8210e-05
    Epoch 33/40
    6397/6397 [==============================] - 178s 28ms/step - loss: 0.6893 - accuracy: 0.5331 - val_loss: 0.6913 - val_accuracy: 0.5273 - lr: 2.7082e-05
    Epoch 34/40
    6397/6397 [==============================] - 171s 27ms/step - loss: 0.6894 - accuracy: 0.5321 - val_loss: 0.6907 - val_accuracy: 0.5281 - lr: 2.5999e-05
    Epoch 35/40
    6397/6397 [==============================] - 172s 27ms/step - loss: 0.6891 - accuracy: 0.5337 - val_loss: 0.6908 - val_accuracy: 0.5295 - lr: 2.4959e-05
    Epoch 36/40
    6397/6397 [==============================] - 172s 27ms/step - loss: 0.6894 - accuracy: 0.5332 - val_loss: 0.6907 - val_accuracy: 0.5279 - lr: 2.3960e-05
    Epoch 37/40
    6397/6397 [==============================] - 172s 27ms/step - loss: 0.6893 - accuracy: 0.5343 - val_loss: 0.6910 - val_accuracy: 0.5284 - lr: 2.3002e-05
    Epoch 38/40
    6397/6397 [==============================] - 171s 27ms/step - loss: 0.6892 - accuracy: 0.5340 - val_loss: 0.6908 - val_accuracy: 0.5252 - lr: 2.2082e-05
    Epoch 39/40
    6397/6397 [==============================] - 170s 27ms/step - loss: 0.6894 - accuracy: 0.5323 - val_loss: 0.6909 - val_accuracy: 0.5265 - lr: 2.1199e-05
    Epoch 40/40
    6397/6397 [==============================] - 172s 27ms/step - loss: 0.6893 - accuracy: 0.5330 - val_loss: 0.6907 - val_accuracy: 0.5248 - lr: 2.0351e-05
:::
:::

::: {.cell .code execution_count="45"}
``` python
stats.plot_history(lol_transformer_8_8_hist)
```

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/6468db48398d57c79f025aab6a225f6853c72b65.png)
:::

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/a1f0be439f2e004369e5301bf2f49741139bebf2.png)
:::
:::

::: {.cell .markdown}
## Analysis

After training a few different models, we will try to compare them with
each other. To do that, we will not only consider the accuracy, but also
a few other measurements that could give us a better indication of the
performance and usability.

### Confidence

We will calculate the confidence and average confidence over the test
data. To do this, we have to define how we are calculating this
confidence first. The lowest possible confidence should be at
predictions of 0.5 and the highest at 0.0 or 1.0. The confidence values
should range from 0 to 1.

There are different ways to create a function for this, but here is a
simple example:

    conf(y_hat) = (2*y_hat - 1)^2

### Confident Accuracy

Using the concept of the previously introduced confidence, we can
calculate an accuracy value that is reflecting the accuracy for
confident values stronger than for less confident values. Even if our
overall accuracy is low, the model could still be useful. For example,
if the model predict around 0.5 9 out of 10 times with an accuracy of
50%, but 1 out of 10 times either close to 0 or close to 1 with a high
accuracy, those highly confident predictions would be very useful in the
games where they apply. This could be that case if only specific
champion combinations influence the win chance heavily, while most
combinations matter much less.

We can calculate the confident accuracy with the following function:

    cacc(y_hat, y) = 0.5 + |(y_hat - 0.5)| * 1 if |y_hat - y| < 0.5 else -1

With this formula, predictions around 0.5 will always have a confident
accuracy of 0.5, no matter the true accuracy. Only high or low
predictions will impact the overall confident accuracy.

### Selected Accuracy

We will also calculate the accuracy above some selected confidence
thresholds. For example, we will calculate the accuracy for all
confidence values above 0.5. This gives us another indication of how
good the model performs for specific ranges.
:::

::: {.cell .code execution_count="86"}
``` python
import stats

def confidence(y_hat):
    return tf.pow((2*y_hat -1), 2)


def conf_acc(y_hat, y):
    #y = tf.cast(y, dtype=tf.float32)
    acc_mult = tf.where(tf.abs(y_hat-y)<0.5, 1.0, -1.0)
    return 0.5 + tf.abs(y_hat-0.5) * acc_mult


# calc avg for models
def avg_confidence(model):
    y_hat = model.predict(test_x, batch_size=64, verbose=False)

    c = confidence(y_hat)
    c = tf.reshape(c, (-1,))
    return float(tf.reduce_mean(c))

def avg_conf_acc(model):
    y_hat = model.predict(test_x, batch_size=64, verbose=False)
    y = test_y

    ca = conf_acc(y_hat, y)
    ca = tf.reshape(ca, (-1,))
    return float(tf.reduce_mean(ca))

def create_selected_accuracy(min_conf=0.2):
    def sel_acc(model):
        y_hat = model.predict(test_x, batch_size=64, verbose=False)
        y = test_y

        acc = tf.where(tf.abs(y_hat-y)<0.5, 1, 0).numpy()
        conf = confidence(y_hat).numpy()

        acc = np.reshape(acc, (-1,))
        conf = np.reshape(conf, (-1,))

        acc = acc.tolist()
        conf = conf.tolist()
        conf_acc = []
        for a, c in zip(acc, conf):
            if c >= min_conf:
                conf_acc.append(a)

        if len(conf_acc) == 0:
            return 0.5
        conf_acc = np.array(conf_acc)
        return conf_acc.mean()
    return sel_acc

# comparator from stats
# see stats.py for details
comparator = stats.ModelComparator((test_x, test_y))

# add our measurements
comparator.add_measurement("Avg Conf", avg_confidence)
comparator.add_measurement("Conf Acc", avg_conf_acc)
comparator.add_measurement("Conf>0.01 Acc", create_selected_accuracy(0.01))
comparator.add_measurement("Conf>0.1 Acc", create_selected_accuracy(0.1))
comparator.add_measurement("Conf>0.3 Acc", create_selected_accuracy(0.3))

# TODO: fix bug with trivial model, output batch size is smaller than input?
#comparator.add_model(trivial_model, trivial_hist, "Trivial")
comparator.add_model(basic_embedding, be_hist, "BasicEmb")
comparator.add_model(aug_emb2, aug_hist2, "AugEmb")
comparator.add_model(deep_emb_model, deep_emb_hist, "DeepEmb")
comparator.add_model(deeper_emb_model, deeper_emb_hist, "DeeperEmb")
comparator.add_model(deep_conv_model, deeper_conv_hist, "DeepConv")
comparator.add_model(lol_transformer, lol_transformer_hist, "Tranformer")
comparator.add_model(lol_transformer_8_8, lol_transformer_8_8_hist, "Transformer8_8")

comparator.plot_histories()

_ = comparator.print_table()



```

::: {.output .stream .stdout}
    Evaluating  BasicEmb
      1/405 [..............................] - ETA: 6s - loss: 0.7348 - accuracy: 0.4688
:::

::: {.output .stream .stdout}
    405/405 [==============================] - 1s 2ms/step - loss: 0.7082 - accuracy: 0.5174
    Test accuracy:  0.7081529498100281
    Test loss:  0.5174171328544617
    Evaluating  AugEmb
    405/405 [==============================] - 1s 2ms/step - loss: 0.6898 - accuracy: 0.5313
    Test accuracy:  0.6898033618927002
    Test loss:  0.5313199758529663
    Evaluating  DeepEmb
    405/405 [==============================] - 2s 4ms/step - loss: 0.6900 - accuracy: 0.5345
    Test accuracy:  0.6900472044944763
    Test loss:  0.5344867706298828
    Evaluating  DeeperEmb
    405/405 [==============================] - 3s 8ms/step - loss: 0.6894 - accuracy: 0.5344
    Test accuracy:  0.6893748044967651
    Test loss:  0.5344095230102539
    Evaluating  DeepConv
    405/405 [==============================] - 5s 12ms/step - loss: 0.6898 - accuracy: 0.5289
    Test accuracy:  0.6897768378257751
    Test loss:  0.5289255976676941
    Evaluating  Tranformer
    405/405 [==============================] - 3s 8ms/step - loss: 0.6893 - accuracy: 0.5333
    Test accuracy:  0.6893306374549866
    Test loss:  0.5332509279251099
    Evaluating  Transformer8_8
    405/405 [==============================] - 7s 16ms/step - loss: 0.6897 - accuracy: 0.5358
    Test accuracy:  0.6896950006484985
    Test loss:  0.5357998013496399
:::

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/17a747ce7c15bb8dcbe23d328e9ddfede176d798.png)
:::

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/d6a006932f01012894e14fa7d67295f5102f6937.png)
:::

::: {.output .stream .stdout}
    | Model | Test acc | Test loss | Avg Conf | Conf Acc | Conf>0.01 Acc | Conf>0.1 Acc | Conf>0.3 Acc |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | BasicEmb | 0.7082 | 0.5174 | 0.0478 | 0.5055 | 0.5322 | 0.5526 | 0.6058 |
    | DeepEmb | 0.6900 | 0.5345 | 0.0129 | 0.5048 | 0.5590 | 0.7069 | 0.5000 |
    | AugEmb | 0.6898 | 0.5313 | 0.0063 | 0.5032 | 0.5669 | 1.0000 | 0.5000 |
    | DeepConv | 0.6898 | 0.5289 | 0.0057 | 0.5031 | 0.5753 | 0.5000 | 0.5000 |
    | Transformer8_8 | 0.6897 | 0.5358 | 0.0096 | 0.5041 | 0.5643 | 0.7436 | 0.5000 |
    | DeeperEmb | 0.6894 | 0.5344 | 0.0091 | 0.5042 | 0.5709 | 0.5000 | 0.5000 |
    | Tranformer | 0.6893 | 0.5333 | 0.0089 | 0.5041 | 0.5739 | 0.6667 | 0.5000 |
:::
:::

::: {.cell .markdown}
Table output in markdown

Note:

conf \> 0.1 == \~0.65+

conf \> 0.01 == 0.55+

  ---------------------------------------------------------------------------------------------
  Model            Test loss  Test acc  Avg Conf  Conf Acc   Conf\>0.01   Conf\>0.1   Conf\>0.3
                                                                    Acc         Acc         Acc
  ---------------- --------- --------- --------- --------- ------------ ----------- -----------
  BasicEmb            0.7082    0.5174    0.0478    0.5055       0.5322      0.5526      0.6058

  DeepEmb             0.6900    0.5345    0.0129    0.5048       0.5590      0.7069      0.5000

  AugEmb              0.6898    0.5313    0.0063    0.5032       0.5669      1.0000      0.5000

  DeepConv            0.6898    0.5289    0.0057    0.5031       0.5753      0.5000      0.5000

  Transformer8_8      0.6897    0.5358    0.0096    0.5041       0.5643      0.7436      0.5000

  DeeperEmb           0.6894    0.5344    0.0091    0.5042       0.5709      0.5000      0.5000

  Tranformer          0.6893    0.5333    0.0089    0.5041       0.5739      0.6667      0.5000
  ---------------------------------------------------------------------------------------------
:::

::: {.cell .code execution_count="79"}
``` python
stats.visualize_embeddings(lol_transformer_8_8.champ_embedding, size=170)
```

::: {.output .display_data}
![](vertopal_9b673558d3ed45ecaffbe0033b33f888/feef20b93e55bceb2d724e923b077bbe8780791d.png)
:::
:::

::: {.cell .markdown}
## Results

While testing different architectures and hyperparameters, we conclude
that it is to some extent possible to predict the outcomes of games
based on the selected champions, but not with the highest accuracy due
to many other variables that can not be accounted for. The positive part
of this is that we know that games are not decided by the selected
champions, but by the events that happen during the game.
:::

::: {.cell .markdown}
## Application

In this section we will briefly demonstrate how the trained models could
be used. We have created a simple class that has operations. Ideally,
this would have a user-friendly UI or take the inputs directly from the
client, but for testing purposes this is just a python class with some
methods you can call.

### Win Chance Prediction

Just like we did in our tests, this method returns the predicted win
chance for a given list of champions. To make it a bit easier to use, we
can just input two lists of champions, one for each team, and get the
predicted win chance for the blue side. We can also use names directly
instead of converting to indices first, which makes this a bit simpler
to use.
:::

::: {.cell .code execution_count="80"}
``` python
from lol_prediction import LoLPredictor

predictor = LoLPredictor(lol_transformer_8_8)
```
:::

::: {.cell .markdown}
#### Example Champion Select

![champ select](data/notebook/img/champ_sel1.png)
:::

::: {.cell .code execution_count="81"}
``` python
blue = ["VelKoz", "Jhin", "Taliyah"]
red = ["Xayah", "Graves", "Malzahar", "LeBlanc"]

# predict blue side win chance
predictor.win_chance(blue, red)
```

::: {.output .stream .stdout}
    1/1 [==============================] - 0s 32ms/step
:::

::: {.output .execute_result execution_count="81"}
    0.5423562526702881
:::
:::

::: {.cell .markdown}
### Optimal Champion

By calculating the win chance for all available champions, we can find
the champion that gives us the highest win chance.
:::

::: {.cell .code execution_count="82"}
``` python
# predict next best champion (for blue side)
_=predictor.best_pick(blue, red)
```

::: {.output .stream .stdout}
    5/5 [==============================] - 0s 22ms/step
    5 best picks for
    ['Velkoz', 'Jhin', 'Taliyah', 'null', 'null']
    vs
    ['Xayah', 'Graves', 'Malzahar', 'Leblanc', 'null']
    86
    Shyvana: 57.66%
    19
    Warwick: 57.53%
    161
    Renata: 57.5%
    95
    Sejuani: 57.24%
    32
    Amumu: 57.19%
:::
:::

::: {.cell .markdown}
We can also specify a list of champions to check, in case some of them
are banned, or the player can only play some of them.
:::

::: {.cell .code execution_count="83"}
``` python
my_champs = ["KhaZix", "LeeSin", "Gragas", "Ivern"]
_=predictor.best_pick(blue, red, my_champs)
```

::: {.output .stream .stdout}
    1/1 [==============================] - 0s 29ms/step
    5 best picks for
    ['Velkoz', 'Jhin', 'Taliyah', 'null', 'null']
    vs
    ['Xayah', 'Graves', 'Malzahar', 'Leblanc', 'null']
    70
    Gragas: 54.78%
    101
    Khazix: 54.73%
    145
    Ivern: 53.37%
    60
    LeeSin: 50.88%
:::
:::

