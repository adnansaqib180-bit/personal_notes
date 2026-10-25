# LeNet CNN model
import numpy as np
# import keras_hub as kb
from keras.models import Sequential
from keras.layers import *

# model = Sequential()
# model.add(Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=(32, 32, 1)))
# model.add(AveragePooling2D(pool_size=(2, 2), strides=2,padding='valid'))
# model.add(Conv2D(16, kernel_size=(5, 5), activation='tanh',padding='valid'))
# model.add(AveragePooling2D(pool_size=(2, 2), strides=2,padding='valid'))
# model.add(Flatten())
# model.add(Dense(120, activation='tanh'))
# model.add(Dense(84, activation='tanh'))
# model.add(Dense(10, activation='softmax'))

# print(model.summary())

# simple RNN layer

# model.add(SimpleRNN(150, activation='tanh', return_sequences=True))
# then simply add a TimeDistributed layer to apply the Dense layer to each time step
# model.add(TimeDistributed(Dense(10, activation='softmax')))

# LSTM layer

# model.add(LSTM(150, activation='tanh', return_sequences=True))
# then simply add a TimeDistributed layer to apply the Dense layer to each time step
# model.add(TimeDistributed(Dense(10, activation='softmax')))

# Gru layer 
# model.add(GRU(150,activation='tanh',return_sequences=True))
# then simply add a TimeDistributed layer to apply the Dense layer to each time step
# model.add(TimeDistributed(Dense(10, activation='softmax')))    

# deep rnn

# model.add(SimpleRNN(150, activation='tanh', return_sequences=True))
# model.add(SimpleRNN(150, activation='tanh', return_sequences=True))

# this is a deep rnn with 2 layers 
# you can also use LSTM or GRU instead of SimpleRNN

#bidirectional RNN

# model.add(Bidirectional(SimpleRNN(150, activation='tanh', return_sequences=True)))

# again you can use LSTM or GRU instead of SimpleRNN 

# transformer encoder 

# class Encoder(Layer):
#     def __init__(self,embed_dim,num_heads,ff_dim,dropout=0.1):
#         super().__init__()
#         self.attention = MultiHeadAttention(
#             num_heads=num_heads,
#             key_dim= embed_dim // num_heads
            
#         )
#         self.norm1 = LayerNormalization(epsilon=1e-6)
#         self.norm2 = LayerNormalization(epsilon=1e-6)
#         self.fnn = Sequential(
#             layers=[Dense(ff_dim,activation='relu'),
#             Dense(embed_dim)],
#             trainable=True
#         )
#         self.dropout1 = Dropout(dropout)
#         self.dropout2 = Dropout(dropout)

#     def call(self,inputs,training=False):
#         attention_output = self.attention(
#             key = inputs,
#             value = inputs,
#             query = inputs
#         )
#         attention_output = self.dropout1(attention_output,training=training)
#         x = self.norm1(attention_output + inputs)
#         fnn_output  = self.fnn(x)
#         fnn_output = self.dropout2(fnn_output,training=training)

#         return self.norm2(x+fnn_output)

# class Decoder (Layer):
#     def __init__(self,embed_dim,ff_dim,num_heads,dropout=0.1):
#         super().__init__()
#         self.embeddings = Embedding(input_dim=4,output_dim=embed_dim)
#         self.masked_attention = MultiHeadAttention(
#             num_heads=num_heads,
#             key_dim=embed_dim//num_heads,
#         )   
#         self.attention = MultiHeadAttention(
#             num_heads=num_heads,
#             key_dim= embed_dim // num_heads
#         )
#         self.norm1 = LayerNormalization(epsilon=1e-6)
#         self.norm2 = LayerNormalization(epsilon=1e-6)
#         self.norm3 = LayerNormalization(epsilon=1e-6)
#         self.fnn = Sequential(
#             Dense(ff_dim,activation='relu'),
#             Dense(embed_dim)
#         )
#         self.dropout1 = Dropout(dropout)
#         self.dropout2 = Dropout(dropout)
#         self.dropout3 = Dropout(dropout)
#     def softmax(num):
#         n = np.exp(num - np.max(num, axis=-1, keepdims=True))
#         result = n / np.sum(n, axis=-1, keepdims=True)
#         return result
#     def call(self,inputs,training=False):
#         inputs = self.embeddings(inputs)
#         m_attention_output = self.masked_attention(
#             key = inputs,
#             query = inputs,
#             value = inputs
#         )
#         m_attention_output = self.dropout1(m_attention_output,training=training)
#         x1 = self.norm1(m_attention_output + inputs)
#         attention_output = self.attention(
#             key = x1,
#             query = x1,
#             value = x1
#         )
#         attention_output = self.dropout2(attention_output,training=training)
#         x2 = self.norm2(attention_output + x1)
#         fnn_output = self.fnn(x2)
#         fnn_output = self.dropout3(fnn_output,training=training)

#         x3 = self.norm3(fnn_output,training=training)
#         return self.softmax(x3)

import tiktoken
import numpy as np
import tensorflow as tf
tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab # 
raw_code_data = """
def add_numbers(a, b):
    result = a + b
    return result

def multiply_numbers(x, y):
    res = x * y
    return res
"""

token_ids = tokenizer.encode(raw_code_data)
visual_tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
print("|".join(visual_tokens))
print("\nTotal Tokens:", len(token_ids))
print(f"Total Tokens Generated: {len(token_ids)}")
print("First 10 Token IDs:", token_ids[:10])

seq_length = 8 
X_data = []
y_data = []

for i in range(0, len(token_ids) - seq_length):
    X_window = token_ids[i : i + seq_length]
    y_window = token_ids[i + 1 : i + seq_length + 1]
    
    X_data.append(X_window)
    y_data.append(y_window)

X_train = np.array(X_data)
y_train = np.array(y_data)

print("X_train Shape:", X_train.shape) 
print("y_train Shape:", y_train.shape) 
