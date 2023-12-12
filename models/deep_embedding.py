import tensorflow as tf


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