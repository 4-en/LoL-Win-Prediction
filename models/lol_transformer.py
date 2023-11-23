import tensorflow as tf

# league of legends match prediction that is (more or less) based on transformer architecture

# input: 10 champion ids (5 for each team) -> win chance between 0 and 1 (based on team 1 winning)




class LoLTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, embedding_dim):
        super(LoLTransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        # multi-head attention
        self.lin_qkv = tf.keras.layers.Dense(3*self.embedding_dim*self.num_heads, input_shape=(self.embedding_dim,), activation=None)

        self.layernorm1 = tf.keras.layers.LayerNormalization()

        #self.head_ffs = [tf.keras.layers.Dense()]

    def split_heads(self, x):
        # x.shape = (batch_size, 10, embedding_dim)
        # split into num_heads
        x = self.lin_qkv(x)
        # x.shape = (batch_size, 10, 3*embedding_dim*num_heads)
        x.reshape(-1, 10, self.num_heads, 3, self.embedding_dim)

        # split into q, k, v
        q = x[:,:,:,0,:]
        k = x[:,:,:,1,:]
        v = x[:,:,:,2,:]

        return q, k, v
    
    def scaled_dot_product_attention(self, q, k, v):
        # perform self attention for each head


        

    def call(self, x):
        x_original = x
        # do something with x...

        # combine with original input
        x = x + x_original
        x = self.layernorm1(x)
        pass


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
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(10,self.embedding_dim))


    def call(self, x):
        # x.shape = (batch_size, 10), where each element is a champion index

        # get embeddings
        x = self.champ_embedding(x)
        # x.shape = (batch_size, 10, embedding_dim)
        # add team embeddings, 0-4 are team 1, 5-9 are team 2
        t = tf.concat([self.team_embedding(tf.zeros((x.shape[0],5))), self.team_embedding(tf.ones((x.shape[0],5)))], axis=1)
        x = x + t
        # x.shape = (batch_size, 10, embedding_dim)
        
        # transformer layers
        for transformer in self.transformer_layers:
            x = transformer(x)
        # x.shape = (batch_size, 10, embedding_dim)

        # output layers
        x = self.dense(x)
        # x.shape = (batch_size, 1)
        return x