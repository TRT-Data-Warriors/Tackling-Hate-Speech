from transformers import ConvBertTokenizer, TFConvBertModel
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from transformers import *
import os
from text_cleaning import clean_text

gpu_number = 2 #### GPU number 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[gpu_number], 'GPU') 
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

MAX_LENGTH = 32
BATCH_SIZE = 256

model_name = 'dbmdz/convbert-base-turkish-mc4-uncased'
tokenizer = ConvBertTokenizer.from_pretrained(model_name)

label_to_name = {0:"INSULT", 
                 1:"OTHER", 
                 2:"PROFANITY", 
                 3:"RACIST", 
                 4:"SEXIST"}


custom_object = {"TFConvBertModel": TFConvBertModel, "K":K}

second_model_1 = tf.keras.models.load_model('./2inci_model_mc4_emir_aug_data_dropout01_0.h5', custom_objects=custom_object, compile=False)
second_model_2 = tf.keras.models.load_model('./2inci_model_mc4_emir_aug_data_dropout01_1.h5', custom_objects=custom_object, compile=False)
second_model_3 = tf.keras.models.load_model('./2inci_model_mc4_emir_aug_data_dropout01_2.h5', custom_objects=custom_object, compile=False)
second_model_4 = tf.keras.models.load_model('./2inci_model_mc4_emir_aug_data_dropout01_3.h5', custom_objects=custom_object, compile=False)
second_model_5 = tf.keras.models.load_model('./2inci_model_mc4_emir_aug_data_dropout01_4.h5', custom_objects=custom_object, compile=False)

third_model_1 = tf.keras.models.load_model('fifth_results_9664/3uncu_model_mc4_emir_aug_data_0.h5', custom_objects=custom_object, compile=False)
third_model_2 = tf.keras.models.load_model('fifth_results_9664/3uncu_model_mc4_emir_aug_data_1.h5', custom_objects=custom_object, compile=False)
third_model_3 = tf.keras.models.load_model('fifth_results_9664/3uncu_model_mc4_emir_aug_data_2.h5', custom_objects=custom_object, compile=False)
third_model_4 = tf.keras.models.load_model('fifth_results_9664/3uncu_model_mc4_emir_aug_data_3.h5', custom_objects=custom_object, compile=False)
third_model_5 = tf.keras.models.load_model('fifth_results_9664/3uncu_model_mc4_emir_aug_data_4.h5', custom_objects=custom_object, compile=False)

first_model_1 = tf.keras.models.load_model('./1nci_model_h5-format-models/model0.h5', custom_objects=custom_object, compile=False)
first_model_2 = tf.keras.models.load_model('./1nci_model_h5-format-models/model1.h5', custom_objects=custom_object, compile=False)
first_model_3 = tf.keras.models.load_model('./1nci_model_h5-format-models/model2.h5', custom_objects=custom_object, compile=False)
first_model_4 = tf.keras.models.load_model('./1nci_model_h5-format-models/model3.h5', custom_objects=custom_object, compile=False)
first_model_5 = tf.keras.models.load_model('./1nci_model_h5-format-models/model4.h5', custom_objects=custom_object, compile=False)

def bert_encode(data):
    tokens = tokenizer.batch_encode_plus(data, max_length=MAX_LENGTH, padding='max_length', truncation=True)
    
    return tf.constant(tokens['input_ids'])


def test_predict(test_df):
    
    test_df  = clean_text(test_df)
    TEXT_COLUMN = 'number_to_text'  # bunu verdikleri şekilde uygularsın
    test_encoded = bert_encode(test_df[TEXT_COLUMN].tolist())
    test_dataset = (
        tf.data.Dataset
        .from_tensor_slices((test_encoded))
        .batch(BATCH_SIZE))

    y_kfold_second = 0
    y_kfold_third = 0
    y_kfold_first = 0

    for model in [second_model_1, second_model_2, second_model_3, second_model_4, second_model_5]:
        y_kfold_second += model.predict(test_dataset)

    for model in [third_model_1, third_model_2, third_model_3, third_model_4, third_model_5]:
        y_kfold_third += model.predict(test_dataset)

    for model in [first_model_1, first_model_2, first_model_3, first_model_4, first_model_5]:
        y_kfold_first += model.predict(test_dataset)

    y_pred_all = 0.39 * y_kfold_first / 5 + 0.38 * y_kfold_second / 5 + 0.23 * y_kfold_third / 5
    preds = np.argmax(y_pred_all, 1)

    preds_names = [label_to_name[pred] for pred in preds]

    return preds_names

