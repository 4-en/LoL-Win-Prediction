import tensorflow as tf



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