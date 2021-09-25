import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tfomics.layers import MultiHeadAttention

class MultiHeadAttention2(layers.Layer):
    def __init__(self, d_model, num_heads, name=None):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model, use_bias=False)
        self.wk = layers.Dense(d_model, use_bias=False)
        self.wv = layers.Dense(d_model, use_bias=False)

        self.dense = layers.Dense(d_model)
        
    def split_heads(self, x, batch_size, seq_len):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        seq_len = tf.constant(q.shape[1])

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size, seq_len)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size, seq_len)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size, seq_len)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, seq_len, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights



@tf.function
def scaled_dot_product_attention(q, k, v):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    s = tf.shape(scaled_attention_logits)
    flat_last_dim  = tf.reshape(scaled_attention_logits, (s[0], s[1], -1))
    attention_weights = tf.nn.softmax(flat_last_dim, axis=2)  # (..., seq_len_q, seq_len_k)
    attention_weights = tf.reshape(attention_weights, s)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights





class Activation(layers.Layer):
    
    def __init__(self, func, name='conv_activation'):
        super(Activation, self).__init__(name=name)
        
        funcs = {
            'relu' : self.relu,
            'exp' : self.exp,
            'softplus' : self.softplus,
            'gelu' : self.gelu,
            'sigmoid' : self.modified_sigmoid
        }
        self.func = funcs[func]
        
    def call(self, inputs):
        return self.func(inputs)
        
    def relu(self, inputs):
        return tf.math.maximum(0., inputs)
    
    def exp(self, inputs):
        return tf.math.exp(inputs)
        
    def softplus(self, inputs):
        return tf.math.log(1 + tf.math.exp(inputs))
    
    def gelu(self, inputs):
        return 0.5 * inputs * (1.0 + tf.math.erf(inputs / tf.cast(1.4142135623730951, inputs.dtype)))
    
    def modified_sigmoid(self, inputs):
        return 10 * tf.nn.sigmoid(inputs - 8)





def CNN(in_shape=(200, 4), num_filters=32, batch_norm=True, activation='relu', pool_size=4, dense_units=512, num_out=12):

    inputs = Input(shape=in_shape)
    nn = layers.Conv1D(filters=num_filters, kernel_size=19, use_bias=False, padding='same')(inputs)
    if batch_norm:
        nn = layers.BatchNormalization()(nn)
    nn = layers.Activation(activation, name='conv_activation')(nn)
    nn = layers.MaxPool1D(pool_size=pool_size)(nn)
    nn = layers.Dropout(0.1)(nn)

    nn = layers.Flatten()(nn)

    nn = layers.Dense(dense_units, use_bias=False)(nn)
    nn = layers.BatchNormalization()(nn)
    nn = layers.Activation('relu')(nn)
    nn = layers.Dropout(0.5)(nn)

    outputs = layers.Dense(num_out, activation='sigmoid')(nn)

    return Model(inputs=inputs, outputs=outputs)

def CNN_ATT(in_shape=(200, 4), num_filters=32, kernel_size=19, batch_norm=True, activation='relu', pool_size=5, layer_norm=False, heads=4, vector_size=32, layer_norm_after=False, dense_units=512, num_out=12):

    inputs = Input(shape=in_shape)
    nn = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, use_bias=False, padding='same')(inputs)
    if batch_norm:
        nn = layers.BatchNormalization()(nn)
    nn = Activation(activation, name='conv_activation')(nn)
    nn = layers.MaxPool1D(pool_size=pool_size)(nn)
    nn = layers.Dropout(0.1)(nn)

    if layer_norm:
        nn = layers.LayerNormalization()(nn)
    nn, w = MultiHeadAttention(num_heads=heads, d_model=heads*vector_size)(nn, nn, nn)
    if layer_norm_after:
    	nn = layers.LayerNormalization()(nn)
    nn = layers.Dropout(0.1)(nn)

    nn = layers.Flatten()(nn)

    nn = layers.Dense(dense_units, use_bias=False)(nn)
    nn = layers.BatchNormalization()(nn)
    nn = layers.Activation('relu')(nn)
    nn = layers.Dropout(0.5)(nn)

    outputs = layers.Dense(num_out, activation='sigmoid')(nn)

    return Model(inputs=inputs, outputs=outputs)

def CNN_LSTM(in_shape=(200, 4), num_filters=32, batch_norm=True, activation='relu', pool_size=4, lstm_units=128, dense_units=512, num_out=12):
    
    inputs = Input(shape=in_shape)
    nn = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, use_bias=False, padding='same')(inputs)
    if batch_norm:
        nn = layers.BatchNormalization()(nn)
    nn = Activation(activation, name='conv_activation')(nn)
    nn = layers.MaxPool1D(pool_size=pool_size)(nn)
    nn = layers.Dropout(0.1)(nn)

    forward = layers.LSTM(lstm_units//2, return_sequences=True, use_bias=False)
    backward = layers.LSTM(lstm_units//2, activation='relu', return_sequences=True, go_backwards=True, use_bias=False)
    nn = layers.Bidirectional(forward, backward_layer=backward)(nn)
    nn = layers.Dropout(0.1)(nn)

    nn = layers.Flatten()(nn)

    nn = layers.Dense(dense_units, use_bias=False)(nn)
    nn = layers.BatchNormalization()(nn)
    nn = layers.Activation('relu')(nn)
    nn = layers.Dropout(0.5)(nn)

    outputs = layers.Dense(num_out, activation='sigmoid')(nn)

    return Model(inputs=inputs, outputs=outputs)

def CNN_LSTM_ATT(in_shape=(200, 4), num_filters=32, batch_norm=True, activation='relu', pool_size=5, lstm_units=128, layer_norm=False, heads=1, vector_size=32, dense_units=512, num_out=12):

    inputs = Input(shape=in_shape)
    nn = layers.Conv1D(filters=num_filters, kernel_size=19, use_bias=False, padding='same')(inputs)
    if batch_norm:
        nn = layers.BatchNormalization()(nn)
    nn = layers.Activation(activation, name='conv_activation')(nn)
    nn = layers.MaxPool1D(pool_size=pool_size)(nn)
    nn = layers.Dropout(0.1)(nn)

    forward = layers.LSTM(lstm_units//2, return_sequences=True, use_bias=False)
    backward = layers.LSTM(lstm_units//2, activation='relu', return_sequences=True, go_backwards=True, use_bias=False)
    nn = layers.Bidirectional(forward, backward_layer=backward)(nn)
    nn = layers.Dropout(0.1)(nn)
    
    if layer_norm:
        nn = layers.LayerNormalization()(nn)
    nn, w = MultiHeadAttention(num_heads=heads, d_model=heads*vector_size)(nn, nn, nn)
    nn = layers.Dropout(0.1)(nn)

    nn = layers.Flatten()(nn)

    nn = layers.Dense(dense_units, use_bias=False)(nn)
    nn = layers.BatchNormalization()(nn)
    nn = layers.Activation('relu')(nn)
    nn = layers.Dropout(0.5)(nn)

    outputs = layers.Dense(num_out, activation='sigmoid')(nn)

    return Model(inputs=inputs, outputs=outputs)

def CNN_TRANS(in_shape=(200, 4), num_filters=32, batch_norm=True, activation='relu', pool_size=4, num_layers=1, heads=8, d_model=64, dense_units=512, num_out=12):
    
    inputs = Input(shape=in_shape)
    nn = layers.Conv1D(filters=num_filters, kernel_size=19, use_bias=False, padding='same')(inputs)
    if batch_norm:
        nn = layers.BatchNormalization()(nn)
    nn = layers.Activation(activation, name='conv_activation')(nn)
    nn = layers.MaxPool1D(pool_size=pool_size)(nn)
    nn = layers.Dropout(0.1)(nn)
    
    nn = layers.Dense(units=key_size, use_bias=False)(nn)
    
    nn = layers.LayerNormalization(epsilon=1e-6)(nn)
    for i in range(num_layers):
        nn2,_ = MultiHeadAttention(d_model=d_model, num_heads=heads)(nn, nn, nn)
        nn2 = layers.Dropout(0.1)(nn2)
        nn = layers.Add()([nn, nn2])
        nn = layers.LayerNormalization(epsilon=1e-6)(nn)
        nn2 = layers.Dense(32, activation='relu')(nn)
        nn2 = layers.Dropout(0.2)(nn2)
        nn2 = layers.Dense(key_size)(nn2)
        nn2 = layers.Dropout(0.1)(nn2)
        nn = layers.Add()([nn, nn2])
        nn = layers.LayerNormalization(epsilon=1e-6)(nn)
    
    nn = layers.Flatten()(nn)

    nn = layers.Dense(dense_units, use_bias=False)(nn)
    nn = layers.BatchNormalization()(nn)
    nn = layers.Activation('relu')(nn)
    nn = layers.Dropout(0.5)(nn)

    outputs = layers.Dense(num_out, activation='sigmoid')(nn)

    return Model(inputs=inputs, outputs=outputs)

def CNN_LSTM_TRANS(in_shape=(200, 4), num_filters=32, batch_norm=True, activation='relu', pool_size=4, num_layers=1, heads=8, d_model=64, dense_units=512, num_out=12):
    
    inputs = Input(shape=in_shape)
    nn = layers.Conv1D(filters=num_filters, kernel_size=19, use_bias=False, padding='same')(inputs)
    if batch_norm:
        nn = layers.BatchNormalization()(nn)
    nn = layers.Activation(activation, name='conv_activation')(nn)
    nn = layers.MaxPool1D(pool_size=pool_size)(nn)
    nn = layers.Dropout(0.1)(nn)
    
    forward = layers.LSTM(key_size // 2, return_sequences=True)
    backward = layers.LSTM(key_size // 2, activation='relu', return_sequences=True, go_backwards=True)
    nn = layers.Bidirectional(forward, backward_layer=backward)(nn)
    nn = layers.Dropout(0.1)(nn)
    
    nn = layers.LayerNormalization(epsilon=1e-6)(nn)
    for i in range(num_layers):
        nn2,_ = MultiHeadAttention(d_model=d_model, num_heads=heads)(nn, nn, nn)
        nn2 = layers.Dropout(0.1)(nn2)
        nn = layers.Add()([nn, nn2])
        nn = layers.LayerNormalization(epsilon=1e-6)(nn)
        nn2 = layers.Dense(32, activation='relu')(nn)
        nn2 = layers.Dropout(0.2)(nn2)
        nn2 = layers.Dense(key_size)(nn2)
        nn2 = layers.Dropout(0.1)(nn2)
        nn = layers.Add()([nn, nn2])
        nn = layers.LayerNormalization(epsilon=1e-6)(nn)
    
    nn = layers.Flatten()(nn)

    nn = layers.Dense(dense_units, use_bias=False)(nn)
    nn = layers.BatchNormalization()(nn)
    nn = layers.Activation('relu')(nn)
    nn = layers.Dropout(0.5)(nn)

    outputs = layers.Dense(num_out, activation='sigmoid')(nn)

    return Model(inputs=inputs, outputs=outputs)

