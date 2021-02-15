import pickle
from tensorflow.keras import models
from emg_data import Split_Data
from machine_learning import reshape_for_amerilike, train_model_on_data, pre_processing

# PATHS
MODEL_PATH = "./saved_models/Amerilike_ch.h5"
DATA_PATH = "./Data/Raw_Data_8ch.pkl"
SAVING_PATH = "./saved_models/Amerilike_8ch_on_6chAugm.h5"

#Pre-processing bool
AUGMENT = 0
DROP = 0

#Load Data
model = models.load_model()
with open(DATA_PATH) as f:
    Raw_Data = pickle.load(f)

#pre_process data
Train_Labels, Val_Labels, Test_Labels, Train, Val, Test, Tables = pre_processing(Raw_Data, AUGMENT)
#create data-set fro training
Split_Data = Split_Data(Tables, Raw_Data.FSs, Raw_Data.Labels, Raw_Data.labels_name, Train, Val, Test,
                        Train_Labels, Val_Labels, Test_Labels)
Data_reshaped = reshape_for_amerilike(Split_Data)

#train
hist, new_model = train_model_on_data(Data_reshaped)

#save model
with open(SAVING_PATH) as f:
    pickle.dump(new_model, f)