import tensorflow as tf


# a basic model that uses embeddings at the first layer instead of one hot vectors
class DeepConv(tf.keras.Model):
    def __init__(self, champ_num=170, emb_dim=32, conv_layers=5):
        super().__init__()
        player_num = 10
        self.embedding = tf.keras.layers.Embedding(champ_num, emb_dim, input_length=player_num)
        self.expand = tf.keras.layers.Dense(emb_dim*player_num, activation=None, input_shape=(None, emb_dim))
        
        self.layers = []
        for _ in range(conv_layers):
            self.layers.append(tf.keras.layers.Conv1D(emb_dim, 2, activation='gelu', input_shape=(player_num, emb_dim)))
            self.layers.append(tf.keras.layers.MaxPool1D(2))
            self.layers.append(tf.keras.layers.Dense(emb_dim, activation='gelu'))


    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.expand(x)
        # x = (-1, 10, 320)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.dense5(x)