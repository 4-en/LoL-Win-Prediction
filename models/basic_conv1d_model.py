import tensorflow as tf

# a basic model that uses conv1d on the one hot vectors

class WinChance(tf.keras.Model):
    def __init__(self, player_num=10, champ_num=170):
        super(WinChance, self).__init__()

        self.player_num = player_num
        self.champ_num = champ_num

        # this conv1d layer is used to convert the one hot vectors into a more useful representation
        self.conv0 = tf.keras.layers.Conv1D(32, 1, activation='relu', input_shape=(player_num, champ_num))

        # this conv1d layer is used to compare two champions
        self.conv1 = tf.keras.layers.Conv1D(32, 2, activation='relu', input_shape=(player_num, 32))
        self.maxpool = tf.keras.layers.MaxPool1D(2)
        self.flatten = tf.keras.layers.Flatten()
        # 160
        self.dense1 = tf.keras.layers.Dense(220, activation='relu')
        self.dense2 = tf.keras.layers.Dense(220, activation='relu')

        self.conv2 = tf.keras.layers.Conv1D(64, 2, activation='relu', input_shape=(10,22))
        self.maxpool2 = tf.keras.layers.MaxPool1D(2)
        self.flatten2 = tf.keras.layers.Flatten()
        # 60
        self.dense3 = tf.keras.layers.Dense(300, activation='relu')
        self.dense4 = tf.keras.layers.Dense(300, activation='relu')

        self.conv3 = tf.keras.layers.Conv1D(128, 3, activation='relu', input_shape=(10,30))
        self.maxpool3 = tf.keras.layers.MaxPool1D(2)
        self.flatten3 = tf.keras.layers.Flatten()
        # 28
        self.dense5 = tf.keras.layers.Dense(420, activation='relu')
        self.dense6 = tf.keras.layers.Dense(128, activation='relu')
        self.denseOut = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        inputs = tf.reshape(inputs, (-1, self.player_num, self.champ_num))

        x = self.conv0(inputs)
        #x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.reshape(x, (-1, 10, 22))

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = tf.reshape(x, (-1, 10, 30))

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten3(x)
        x = self.dense5(x)
        x = self.dense6(x)
        return self.denseOut(x)