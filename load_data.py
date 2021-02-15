import pickle
import numpy as np
import sys
import pandas as pd
from emg_data import Raw_Data, Split_Data
from bioml import DB_connection as DBC
from bioml import Data_Loader as DL
from machine_learning import pre_processing
SAVING_PATH = "Data/Raw_Data_6ch_2users.pkl"

###### CONSTANTS #######
RECORDINGS_6ch = ['semgtb005_recording_1', 'semgcp002_recording_0'] #, 'semgtb005_recording_1']]
RECORDINGS_8ch = ['vincent_recording_0'] #[['vincent_recording_0', 'thibault_recording_0']]
CHANNELS_6ch = [['Labjack_channel1','Labjack_channel2','Labjack_channel3','Labjack_channel4','Labjack_channel5',
             'Labjack_channel6']]
CHANNELS_8ch = [['Labjack_channel1','Labjack_channel2','Labjack_channel3','Labjack_channel4','Labjack_channel5',
             'Labjack_channel6', 'Labjack_channel7', 'Labjack_channel9']]
HOST = ['128.178.51.53', 'localhost']
EXP = ['leonardo', 'emg_8ch']


##SET PARAMS
params = {
    'dbname': EXP[0],
    'user': 'postgres',
    'host': HOST[0],
    'port': 5432
    }

refs = [[]]
recordings = RECORDINGS_6ch
channels = CHANNELS_6ch

# Connect
print('connect to DB')
engine, conn, cur, exp_info, subj_info = DBC.connect_DB(params)
print(exp_info)

# Extract

print('Extract Data and Labels')
#for i, rec in enumerate(recordings[0]):

Tables, FSs = DL.H_extract_data(engine, recordings, channels, refs, exp_info)
Labels, label_table_name = DL.H_load_labels(engine, cur, recordings)  # TODO deal with FSs and names
    #if(Tables is None):
     #   Tables = Tables_cur
      #  Labels = Labels_cur
    #else:
#    Tables = pd.concat([Tables, Tables_cur])
#    Labels = pd.concat([Labels, Labels_cur])

print("Data and Labels extracted")


Extracted_Raw_Data = Raw_Data(Tables, FSs, Labels, label_table_name, recordings)

with open(SAVING_PATH, "wb") as f:
    pickle.dump(Extracted_Raw_Data, f)