import tensorflow as tf


# a model that generates samples based on predicted mean and variance

class ProbabilisticModel(tf.keras.Model):
    def __init__(self, champion_count=170, embed_dim=32, combination_layers=4):
        super(ProbabilisticModel, self).__init__()
        self.champion_count = champion_count # amount of champions in the game
        
        self.embed_dim = embed_dim # dimension of the embedding layer

        self.flatten = tf.keras.layers.Flatten()

        
        self.embedding = tf.keras.layers.Embedding(champion_count, embed_dim, input_length=10)
        # (-1, 10, 32)
        #self.qkv = tf.keras.layers.Dense(embed_dim*3, activation='gelu')
        # (-1, 10, 32*3)
        # linear layer to reorder embeddings
        self.lin1 = tf.keras.layers.Dense(10*embed_dim, activation=None)
        # (-1, 320)
        self.conv1 = tf.keras.layers.Conv1D(32, 3, activation='gelu')
        self.maxpool1 = tf.keras.layers.MaxPool1D(2)
        self.nlin1 = tf.keras.layers.Dense(32, activation='gelu')

        self.conv2 = tf.keras.layers.Conv1D(32, 3, activation='gelu')
        self.maxpool2 = tf.keras.layers.MaxPool1D(2)
        self.nlin2 = tf.keras.layers.Dense(32, activation='gelu')

        self.meanVar = tf.keras.layers.Dense(2, activation=None)

    def call(self, x):

        x = self.embedding(x)
        x = self.flatten(x)
        x = self.lin1(x)
        
        x = tf.reshape(x, (-1, 10, self.embed_dim))

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.nlin1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.nlin2(x)

        x = self.meanVar(x)

        x = tf.reshape(x, (-1, 2))

        return x
    

class SamplingModel(tf.keras.Model):
    def __init__(self, n_samples=10, champion_count=170, embed_dim=32):
        super().__init__()
        self.n_samples = n_samples
        self.champion_count = champion_count
        self.embed_dim = embed_dim

        self.prob = ProbabilisticModel(champion_count, embed_dim)

    def call(self, x):
        x = self.prob(x)
        mean = x[:, 0]
        mean = tf.expand_dims(mean, -1)
        var = x[:, 1]
        var = tf.expand_dims(var, -1)

        # get gaussian distribution with mean 1 and variance 0 for each sample
        # use parameterization trick to get samples
        # z = mu + sigma * epsilon
        # epsilon ~ N(0, 1)
        epsilon = tf.random.normal((self.n_samples,))
        #print(mean.shape, var.shape, epsilon.shape)
        var = tf.math.exp(0.5 * var) * epsilon
        #print(var.shape)
        samples = mean + var
        return samples
    
    def avg(self, x):
        x = self(x)
        x = tf.math.reduce_mean(x, -1)
        return x
    
    def get_p_model(self):
        return self.prob
    

if __name__ == '__main__':
    model = SamplingModel()
    # 10 random integers between 0 and 169
    x = tf.random.uniform((32, 10), maxval=170, dtype=tf.int32)
    # target = 1
    y_target = tf.ones((32, 1))
    y = model(x)

    print("y_pred: ", y)

    loss = tf.keras.losses.binary_crossentropy(y_target, y)


    print("Loss: ", loss)

    print("Model summary: ")
    model.summary()

    # test loss for all ones
    all_ones = tf.ones((32,10))
    loss = tf.keras.losses.binary_crossentropy(y_target, all_ones)
    print("Loss: ", loss)





