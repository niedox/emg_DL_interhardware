import pickle
import random
import pandas as pd
from tensorflow.keras import models
from emg1 import pre_processing, reshape_for_amerilike
print("feedforwardS")
recordings = ['thibault_recording_0']
model_6ch = models.load_model('Amerilike_6ch.h5')

with open('tables_8ch.pkl', 'rb') as f:
    Tables = pickle.load(f)

with open('FSs_8ch.pkl', 'rb') as f:
    FSs = pickle.load(f)

with open('labels_8ch.pkl', 'rb') as f:
    labels = pickle.load(f)

with open('labels_name_8ch.pkl', 'rb') as f:
    labels_name = pickle.load(f)


rem = random.sample([1, 2, 3, 4, 5, 6, 7, 9], 2)

Tables[0] = Tables[0].drop(["Labjack_channel"+str(rem[0]), "Labjack_channel"+str(rem[1])],
                        axis='columns')
Tables[0].columns = ["Labjack_channel1","Labjack_channel2", "Labjack_channel3", "Labjack_channel4", "Labjack_channel5",
                     "Labjack_channel6", "subject_id", "condition"]

Train_Labels, Val_Labels, Test_Labels, Train, Val, Test, Tables = pre_processing(labels, Tables, recordings, FSs)
Train, Val, Test = reshape_for_amerilike(Train, Val, Test)

losses = model_6ch.evaluate(Test[0], Test_Labels, batch_size=600)
print(losses)

#return Train_Labels, Val_Labels, Test_Labels, Train, Val, Test, Table