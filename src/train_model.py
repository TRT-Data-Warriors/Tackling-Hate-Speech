import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from transformers import TFConvBertModel
from sklearn.metrics import *
from model import *
import logging
logging.disable(logging.INFO)
logging.disable(logging.WARNING)

def train_model(model, train_dataset, dev_dataset, batch_size, epochs):
    model.fit(train_dataset,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=dev_dataset,
              verbose=2) 
    
def fetch_results(model, dev_dataset, true_labels=None, target_names=['INSULT', 'OTHER', 'PROFANITY', 'RACIST', 'SEXIST']):
    probs = model.predict(dev_dataset)
    y_pred = np.argmax(probs, 1)
    if true_labels is not None:
        acc = accuracy_score(true_labels, y_pred)
        f1 = f1_score(true_labels, y_pred, average='macro')
        recall = recall_score(true_labels, y_pred, average='macro')
        precision = precision_score(true_labels, y_pred, average='macro')
        print('Accuracy score: ', acc)
        print('F1 Macro Score: ', f1)
        print('Recall Macro Score: ', recall)
        print('Precision Macro Score: ', precision, '\n')
        print(classification_report(true_labels, y_pred, target_names=target_names))    
        return y_pred, probs
    
    else:
        return y_pred, probs
    
    
    