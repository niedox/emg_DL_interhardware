import pickle
import random
import pandas as pd
from tensorflow.keras import models
from emg1 import pre_processing, reshape_for_amerilike

recordings = ['semgcp002_recording_0']
model_6ch = models.load_model('Amerilike_6ch.h5')

with open('tables.pkl', 'rb') as f:
    Tables = pickle.load(f)

with open('FSs.pkl', 'rb') as f:
    FSs = pickle.load(f)

with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)

with open('labels_name.pkl', 'rb') as f:
    labels_name = pickle.load(f)


Train_Labels, Val_Labels, Test_Labels, Train, Val, Test, Tables = pre_processing(labels, Tables, recordings, FSs)
Train, Val, Test = reshape_for_amerilike(Train, Val, Test)

losses = model_6ch.evaluate(Test, Test_Labels, batch_size=600)
print(losses)

#return Train_Labels, Val_Labels, Test_Labels, Train, Val, Test, Table