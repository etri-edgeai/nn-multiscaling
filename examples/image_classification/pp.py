from __future__ import print_function

import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

def get_model_tf(vocab_size, embed_dim=32, num_heads=2, ff_dim=32, maxlen=20):

    class TransformerBlock(layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
            super(TransformerBlock, self).__init__()
            self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn = keras.Sequential(
                [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
            )
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = layers.Dropout(rate)
            self.dropout2 = layers.Dropout(rate)

        def call(self, inputs, training):
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

    class TokenAndPositionEmbedding(layers.Layer):
        def __init__(self, maxlen, vocab_size, embed_dim):
            super(TokenAndPositionEmbedding, self).__init__()
            self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
            self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

        def call(self, x):
            maxlen = tf.shape(x)[-1]
            positions = tf.range(start=0, limit=maxlen, delta=1)
            positions = self.pos_emb(positions)
            x = self.token_emb(x)
            return x + positions

    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="relu")(x)

    return keras.Model(inputs=inputs, outputs=outputs)

def get_model_brnn(vocab_size):

    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(vocab_size, 8)(inputs)
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.GRU(16, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(16))(x)
    x = layers.Dropout(0.1)(x)
    # Add a classifier
    outputs = layers.Dense(1, activation="relu")(x)
    return keras.Model(inputs, outputs)

def to_integer(data, voca):
    
    x = []
    y = []
    for d in data:
        targets, acc = d
        targets_ = [
            voca[t] if t in voca else len(voca)+1
            for t in targets
        ]
        y.append(acc)
        #targets_ = sorted(targets_) # sort by id
        x.append(targets_)

    y_numpy = np.array(y, np.float32)
    return (np.array(x), y_numpy)


def train(data):
    trainsize = int(len(data) * 0.8)
    raw_train_data = data[:trainsize]
    raw_test_data = data[trainsize:]

    print(len(raw_train_data), len(raw_test_data))
    maxlen = 20

    # build voca
    id_ = 1
    voca = {}
    for d in raw_train_data:
        targets, acc = d
        for t in targets:
            if t not in voca:
                voca[t] = id_
                id_ += 1
    
    x_train, y_train = to_integer(raw_train_data, voca)
    x_test, y_test = to_integer(raw_test_data, voca)

    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

    #model = get_model_tf(len(voca)+2, maxlen=maxlen) # including unknown, padding
    model = get_model_brnn(len(voca)+2) # including unknown, padding

    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss="mean_squared_error")
    history = model.fit(
        x_train, y_train, batch_size=3, epochs=100, validation_data=(x_test, y_test)
        )


if __name__ == "__main__":
    data = "dataset/pp.json"
    with open(data, "r") as f:
        data = json.load(f)
    train(data)
