import sys
sys.path.append('./src')
import sys
import pandas as pd
import tensorflow as tf
from transformers import ConvBertTokenizer, TFConvBertModel

from model import bert_bigru_cnn_model, bert_bilstm_attention_model, bert_bilstm_model
from train_model import train_model, fetch_results
from test import *
from data import create_tensor_dataset, create_test_dataset
from text_cleaning import clean_text
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *
import gc

TEXT_COLUMN = 'number_to_text'
LABEL_COLUMN = 'label'
MODEL_NAME = 'dbmdz/convbert-base-turkish-mc4-uncased'

def run(train_data_path, valid_data_path, max_length, epochs, batch_size):
	
	train_df = pd.read_csv(train_data_path)
	valid_df = pd.read_csv(valid_data_path)
 
	# text preprocessing -- default: 'text' column
	if ('text' in train_df.columns) & ('text' in valid_df.columns):
		train_df = clean_text(train_df)
		valid_df = clean_text(valid_df)
 
	le = LabelEncoder()
	train_df[LABEL_COLUMN] = le.fit_transform(train_df.target)
	valid_df[LABEL_COLUMN] = le.transform(valid_df.target)
	label_dict = dict(zip(le.classes_, le.transform(le.classes_)))

	# get tokenizer
	tokenizer = ConvBertTokenizer.from_pretrained(MODEL_NAME)

	X = df[TEXT_COLUMN]
	Y = df[LABEL_COLUMN]

	test_dataset, test_encoded = create_test_dataset(test_df[TEXT_COLUMN].values, tokenizer, max_length=max_length)
	
	skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	skf.get_n_splits(X, Y)
	fold_num = 0
	models = []
	oof = 0
	for train_index, val_index in skf.split(X, Y):
		fold_num+=1
		print("Results for fold",fold_num)
		x_train, x_val = X.iloc[train_index], X.iloc[val_index]
		y_train, y_val = Y.iloc[train_index], Y.iloc[val_index]

		temp_train = pd.concat([x_train, y_train], axis=1)
		temp_valid = pd.concat([x_val, y_val], axis=1)
		train_dataset, dev_dataset = create_tensor_dataset(temp_train, temp_valid, tokenizer, TEXT_COLUMN, LABEL_COLUMN, max_length=max_length)

		model = bert_bilstm_model(bert_encoder, max_length=max_length)
		train_model(model, train_dataset, dev_dataset, batch_size=batch_size, epochs=epochs)
		models.append(model)
		valid_pred = model.predict(dev_dataset)
		f1 = f1_score(y_val, np.argmax(valid_pred, 1), average='macro')
		print(f'{fold_num}. FOLD TEST F1 SCORE: {f1}')

		test_pred = model.predict(test_dataset)
		oof += test_pred
		
		del temp_train, temp_valid, model
		gc.collect()
	
	bert_bilstm_test_predictions = np.argmax(oof/5, 1)
	return bert_bilstm_test_predictions

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--train_data_path", required=True)
arg_parser.add_argument("--valid_data_path", required=True)
arg_parser.add_argument("--max_length", required=True, default=32, type=int)
arg_parser.add_argument("--epochs", required=True, default=20, type=int)
arg_parser.add_argument("--batch_size", required=True, default=256, type=int)

if __name__ == '__main__':
    args = arg_parser.parse_args()
    run(**vars(args))