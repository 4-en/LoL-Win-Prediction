import tensorflow as tf

# goal: create a model that predicts win chance by specifically comparing team synergy and enemy counters

# synergy head that compares 1 champion with 4 champions (synergy) or 5 champions (counter)
class SynergyHead(tf.keras.layers.Layer):
    def __init__(self, in_count, in_dim=32, ff_count=4):
        super(SynergyHead, self).__init__()
        self.in_count = in_count
        self.in_dim = in_dim
        self.ff_count = ff_count


        # goal, compare the first champion with the rest
        # first, multiply the first champion with the rest, then continue with dense layers
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)
        ff_layers = []
        for _ in range(ff_count):
            ff_layers.append(tf.keras.layers.Dense(in_dim*(in_count-1), activation=self.leaky_relu, input_shape=(None, in_count-1, in_dim)))
        self.ff_layers = ff_layers

        self.final_dense = tf.keras.layers.Dense(in_dim, activation=self.leaky_relu, input_shape=(None, in_count-1, in_dim))



    def call(self, inputs):

        # split the input into the first champion and the rest
        x1 = inputs[:, 0, :]
        xRest = inputs[:, 1:, :]


        # multiply the first champion with the rest
        x1 = tf.expand_dims(x1, axis=1)
        xMult = tf.multiply(x1, xRest)
        # keep xMult to add to the result later

        # feed the result into the dense layers
        x = xMult
        # flatten
        #print(self.in_count, self.in_dim)
        x = tf.reshape(x, (-1, (self.in_count-1)*self.in_dim))
        for layer in self.ff_layers:
            x = layer(x)

        # add the result of the multiplication
        xMult = tf.reshape(xMult, (-1, (self.in_count-1)*self.in_dim))
        x = tf.add(x, xMult)

        # normalize the result
        x = tf.math.l2_normalize(x, axis=1)

        # final dense layer
        x = self.final_dense(x)

        # normalize the result
        x = tf.math.l2_normalize(x, axis=1)


        return x






# input: 10 dimensional vector of champion indices
# main model: turn champion indices into single scalar value representing win chance
class SynergyModel(tf.keras.Model):
    def __init__(self, champion_count=170, heads=4, embed_dim=32, combination_layers=4):
        super(SynergyModel, self).__init__()
        self.champion_count = champion_count # amount of champions in the game
        self.head_count = heads # amount of heads in the multihead attention layer for both synergy and counter
        self.embed_dim = embed_dim # dimension of the embedding layer

        SYNERGIES = 5 # amount of synergies to consider, 4 other teammates and 1 target
        COUNTERS = 6 # amount of counters to consider, 5 enemies and 1 target

        self.embedding = tf.keras.layers.Embedding(champion_count, embed_dim)
        # state: (batch_size, 10, 32)

        # weights to expand the input to the amount of heads
        # state -> (batch_size, 10*4*2, 32)
        # 1 row, 4*2*10 columns
        self.h_weights = tf.keras.layers.Dense(heads*2*embed_dim, activation=None, input_shape=(None, None, embed_dim))

        # synergy heads
        self.synergy_heads = []
        for _ in range(heads):
            self.synergy_heads.append(SynergyHead(SYNERGIES, embed_dim))

        # counter heads
        self.counter_heads = []
        for _ in range(heads):
            self.counter_heads.append(SynergyHead(COUNTERS, embed_dim))

        self.reduceHeads = tf.keras.layers.Dense(embed_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.1), input_shape=(None, heads*2*embed_dim))

        # final dense layers
        self.combination_dense = []
        for _ in range(combination_layers):
            self.combination_dense.append(tf.keras.layers.Dense(embed_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))

        self.final_dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        # inputs: (batch_size, 10)
        # output: (batch_size, 1)

        # embed the inputs
        x = self.embedding(inputs)

        x = self.h_weights(x)
        
        x = tf.reshape(x, (-1, self.head_count*2, 10, self.embed_dim))

        # feed the input into the synergy heads
        synergy_out = []
        for i, head in enumerate(self.synergy_heads):
            head_out = []
            for j in range(10):
                myChamp = x[:, i, j, :]
                myChamp = tf.expand_dims(myChamp, axis=1)
                offset = 0 if j < 5 else 5
                team = x[:, i, offset:offset+5, :]
                # remove the champion from the team
                team = tf.concat([team[:, :j-offset], team[:, j+1-offset:]], axis=1)
                # concatenate the champion with the team
                head_out.append(head(tf.concat([myChamp, team], axis=1)))

            # add together the results of the synergy head
            # since the output is supposed to represent the synergy of the team, we sum the results of the synergy head
            # to get the synergy of the team (in theory :D)
            synergy_out.append(tf.reduce_sum(head_out, axis=0))



        counter_out = []
        for i, head in enumerate(self.counter_heads):
            head_out = []
            for j in range(10):
                myChamp = x[:, i, j, :]
                myChamp = tf.expand_dims(myChamp, axis=1)
                offset = 5 if j < 5 else 0
                team = x[:, i, offset:offset+5, :]
                # concatenate the champion with the team
                head_out.append(head(tf.concat([myChamp, team], axis=1)))

            # add together the results of the synergy head
            counter_out.append(tf.reduce_sum(head_out, axis=0))

        # concatenate the synergy and counter outputs
        x = tf.concat(synergy_out + counter_out, axis=1)
        # shape: (batch_size, 4*2*32)

        # combine the heads
        x = self.reduceHeads(x)

        # feed the result into the combination dense layers
        for layer in self.combination_dense:
            x = layer(x)

        # final dense layer
        x = self.final_dense(x)

        return x


if __name__ == "__main__":
    # create some test values
    x = tf.random.uniform((32, 10), minval=0, maxval=170, dtype=tf.int32)
    y = tf.random.uniform((32, 1), minval=0, maxval=1, dtype=tf.float32)

    # create the model
    model = SynergyModel()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    
    # train the model
    model.fit(x, y, epochs=10, batch_size=16)

    # save weights
    model.save_weights("models/synergy_model.h5")


