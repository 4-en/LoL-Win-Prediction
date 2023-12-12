import tensorflow as tf

# league of legends match prediction that is (more or less) based on transformer architecture

# input: 10 champion ids (5 for each team) -> win chance between 0 and 1 (based on team 1 winning)




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
    

if __name__ == "__main__":
    # create fake data
    batch_size = 2

    x = tf.random.uniform((batch_size, 10), minval=0, maxval=170, dtype=tf.int32)
    y = tf.random.uniform((batch_size, 1), minval=0, maxval=2, dtype=tf.int32)

    # create model
    model = LoLTransformer()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # test model
    y_pred = model(x)
    print(y_pred.shape)