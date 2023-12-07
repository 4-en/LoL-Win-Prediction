import tensorflow as tf


# a basic model that uses embeddings at the first layer instead of one hot vectors
class DeepConv(tf.keras.Model):
    def __init__(self, champ_num=170, emb_dim=32, conv_layers=3):
        super().__init__()
        player_num = 10
        self.player_num = player_num
        self.champ_num = champ_num
        self.emb_dim = emb_dim
        self.embedding = tf.keras.layers.Embedding(champ_num, emb_dim, input_length=player_num)
        #self.expand = tf.keras.layers.Dense(emb_dim*player_num, activation=None)
        self.team_embedding = tf.keras.layers.Embedding(3, emb_dim, input_length=player_num)
        self.clayers = []
        for i in range(conv_layers):
            lname = "_layer_" + str(i)
            # add expand layer
            self.clayers.append(self.save_x)
            self.clayers.append(tf.keras.layers.Dense(emb_dim*player_num, activation=None, name="expand"+lname))
            self.clayers.append(self.order_expanded)
            self.clayers.append(tf.keras.layers.Conv1D(emb_dim, 10, strides=10, padding="same", activation='gelu', name="conv1d"+lname))
            #self.clayers.append(tf.keras.layers.MaxPool1D(2, name="maxpool"+lname))
            
            self.clayers.append(tf.keras.layers.Dense(emb_dim, activation='gelu', name="dense"+lname))
            
            self.clayers.append(self.add_saved)
            # layer norm
            self.clayers.append(tf.keras.layers.LayerNormalization(name="layer_norm"+lname))

        #self.maxpool = tf.keras.layers.MaxPool1D(10)
        self.flatten = tf.keras.layers.Flatten()


        self.dense1 = tf.keras.layers.Dense(emb_dim, activation='gelu')
        self.dense2 = tf.keras.layers.Dense(emb_dim, activation='gelu')
        self.out = tf.keras.layers.Dense(1, activation='sigmoid')


    def save_x(self, x):
        # store for residual connection
        # maybe dont store as a class variable, rework loop later
        self.x = x
        return x

    def add_saved(self, x):
        return self.x + x

    def order_expanded(self, x):
        # x is (-1, 10, 320)
        # swap the player and version axes
        #print("before swap: ", x.shape)
        x = tf.reshape(x, (-1, 10, 10, 32))
        # swap the player and version axes
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (-1, 100, 32))
        #print("after swap: ", x.shape)
        return x


    def call(self, inputs):
        # mask for empty slots
        mask = tf.where(inputs == 0, 0, 1)
        x = self.embedding(inputs)
        #x = self.expand(x)
        # x = (-1, 10, 320)
        #x = tf.reshape(x, (-1, 10, 10, 32))
        # batch, player, 10versions, 32
        # swap the player and version axes
        #x = tf.transpose(x, perm=[0, 2, 1, 3])
        #x = tf.reshape(x, (-1, 100, 32))
        
        team_vals = [1,1,1,1,1,2,2,2,2,2]
        team_vals = tf.convert_to_tensor(team_vals)
        team_vals = tf.reshape(team_vals, (-1, 10))
        team_vals = team_vals * mask
        team_vals = tf.cast(team_vals, tf.int32)
        team_vals = self.team_embedding(team_vals)
        x = x + team_vals

        for layer in self.clayers:
            x = layer(x)
            #print(x.shape)

        #x = tf.reshape(x, (-1, 10, 10*32))
        #x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        return x