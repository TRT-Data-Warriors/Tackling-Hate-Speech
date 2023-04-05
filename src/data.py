import pandas as pd
from transformers import ConvBertTokenizer, ElectraTokenizer, BertTokenizer, AutoTokenizer
import tensorflow as tf

def bert_encode(data, tokenizer, max_length=32):
	tokens = tokenizer.batch_encode_plus(data, max_length=max_length, padding='max_length', truncation=True)
	return tf.constant(tokens['input_ids'])

def create_tensor_dataset(train_df, validation_df, tokenizer,
						text_column, label_column,
						max_length=32, batch_size=256):
		
	train_encoded = bert_encode(train_df[text_column].tolist(), tokenizer, max_length=max_length)
	dev_encoded = bert_encode(validation_df[text_column].tolist(), tokenizer, max_length=max_length)

	train_dataset = (
					tf.data.Dataset
					.from_tensor_slices((train_encoded, train_df[label_column]))
					.shuffle(100)
					.batch(batch_size)
					)

	dev_dataset = (
					tf.data.Dataset
					.from_tensor_slices((dev_encoded, validation_df[label_column]))
					.batch(batch_size)
				)

	return train_dataset, dev_dataset

def create_test_dataset(test_sentence_list, tokenizer, max_length, batch_size=256):
	test_encoded = bert_encode(test_sentence_list, tokenizer, max_length=max_length)
	test_dataset = (
					tf.data.Dataset
					.from_tensor_slices(test_encoded)
					.batch(batch_size)
					)
	
	return test_dataset, test_encoded