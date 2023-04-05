import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from transformers import TFConvBertModel


LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
def bert_bilstm_model(bert_encoder, max_length):
    #bert_encoder = TFConvBertModel.from_pretrained(model_name)
    input_word_ids = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    last_hidden_states = bert_encoder.convbert(input_word_ids)[0]
    x = tf.keras.layers.SpatialDropout1D(0.2)(last_hidden_states)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(LSTM_UNITS, activation = 'tanh', recurrent_activation = 'sigmoid', recurrent_dropout = 0, unroll = False, use_bias = True, reset_after = True, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS, activation = 'tanh', recurrent_activation = 'sigmoid', recurrent_dropout = 0, unroll = False, use_bias = True, return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(5, activation='softmax')(hidden)


    model = Model(inputs=input_word_ids, outputs=result)
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=adam_optimizer,metrics=['accuracy'])
    return model


def bert_bigru_cnn_model(bert_encoder, max_length):
    #bert_encoder = TFConvBertModel.from_pretrained(model_name)
    input_word_ids = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    last_hidden_states = bert_encoder(input_word_ids)[0]
    x1 = SpatialDropout1D(0.1)(last_hidden_states)
    x = Bidirectional(tf.keras.layers.GRU(LSTM_UNITS, activation = 'tanh', recurrent_activation = 'sigmoid', recurrent_dropout = 0, unroll = False, use_bias = True, reset_after = True, return_sequences=True))(x1)
    x = tf.keras.layers.Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    y = Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS, activation = 'tanh', recurrent_activation = 'sigmoid', recurrent_dropout = 0, unroll = False, use_bias = True, return_sequences=True))(x1)
    y = tf.keras.layers.Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(y)
    avg_pool1 = GlobalAveragePooling1D()(x)
    max_pool1 = GlobalMaxPooling1D()(x)
    avg_pool2 = GlobalAveragePooling1D()(y)
    max_pool2 = GlobalMaxPooling1D()(y)
    x = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])
    x = Dense(5, activation = "softmax")(x)

    model = Model(inputs = input_word_ids, outputs = x)
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=adam_optimizer,metrics=['accuracy'])
    return model

def bert_bilstm_attention_model(bert_encoder, max_length):
	#bert_encoder = TFConvBertModel.from_pretrained(model_name)
	input_word_ids = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
	last_hidden_states = bert_encoder(input_word_ids)[0]
	x = Bidirectional(LSTM(384,  return_sequences=True, unit_forget_bias=True))(last_hidden_states)
	attention = TimeDistributed(Dense(1, activation='tanh'))(x)
	attention = Flatten()(attention)
	attention = Activation('softmax')(attention)
	attention = RepeatVector(2 * 384)(attention)
	attention = Permute([2, 1])(attention)
	sent_representation = Multiply()([x, attention])
	sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
	probabilities = Dense(5, activation='softmax')(sent_representation)
	model = Model(inputs = input_word_ids, outputs = probabilities)
	adam_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
	model.compile(loss='sparse_categorical_crossentropy',optimizer=adam_optimizer,metrics=['accuracy'])
	return model