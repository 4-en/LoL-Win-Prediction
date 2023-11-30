import tensorflow as tf


# a basic model that uses embeddings at the first layer instead of one hot vectors
class BasicEmbedding(tf.keras.Model):
    def __init__(self, champ_num=170, player_num=10):
        super(BasicEmbedding, self).__init__()
        self.embedding = tf.keras.layers.Embedding(champ_num, 32, input_length=player_num)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='gelu')
        self.dense2 = tf.keras.layers.Dense(178, activation='gelu')
        self.dense3 = tf.keras.layers.Dense(104, activation='gelu')
        self.dense4 = tf.keras.layers.Dense(64, activation='gelu')
        self.dense5 = tf.keras.layers.Dense(1, activation='sigmoid')


    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.dense5(x)