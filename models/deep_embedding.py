import tensorflow as tf


# a deep embedding model with multiple layers consisting of dense layers and residual connections and layer norm
class DeepEmbedding(tf.keras.Model):
    def __init__(self, champ_num=170, embed_dim=32, n_layers=4):
        super(DeepEmbedding, self).__init__()
        self.champ_num = champ_num
        self.embed_dim = embed_dim
        self.n_layers=n_layers
        player_num = 10
        self.embedding = tf.keras.layers.Embedding(champ_num, embed_dim)
        self.flatten = tf.keras.layers.Flatten()
        self.layer_norm = tf.keras.layers.LayerNormalization()

        self.start_dense = tf.keras.layers.Dense(player_num*embed_dim, activation='gelu')
        
        self.fcs = []
        for _ in range(n_layers):
            self.fcs.append(
                tf.keras.layers.Dense(player_num*embed_dim, activation='gelu')
            )

        self.dense_output = tf.keras.layers.Dense(1, activation='sigmoid')


    def call(self, inputs):
        x0 = self.embedding(inputs)
        x0 = self.flatten(x0)

        x = self.start_dense(x0)
        for i in range(self.n_layers):
            x += x0
            x = self.layer_norm(x)
            x = self.fcs[i](x)
            x += x0
            x = self.layer_norm(x)
        
        return self.dense_output(x)