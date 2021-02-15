import pandas as pd
from tensorflow.keras import models
import pickle
from machine_learning import reshape_for_amerilike, pre_processing
import matplotlib.pyplot as plt
import numpy as np
from emg_data import Split_Data

MODEL = "./saved_models/Amerilike_pt8_t6A_U1.h5"
DATA_PATH = "./Data/Raw_Data_6ch_User2.pkl"
SAVEFIG = "results/Amerilike_6chAugm_on_8chAugm.png"
AUGMENT = 1
DROP = 0
#recordings = ['semgcp002_recording_0']
#recordings = ['thibault_recording_0']
VAL_RATIO = 1/6
TEST_RATIO = 1/6 #98/100


with open(DATA_PATH, 'rb') as f:
    Data = pickle.load(f)
model = models.load_model(MODEL)


if DROP:
    rem = [2, 6]
    Data.Tables[0] = Data.Tables[0].drop(["Labjack_channel" + str(rem[0]), "Labjack_channel" + str(rem[1])],
                                   axis='columns')
    Data.Tables[0].columns = ["Labjack_channel1", "Labjack_channel2", "Labjack_channel3", "Labjack_channel4",
                         "Labjack_channel5",
                             "Labjack_channel6", "subject_id", "condition"]

Train_Labels, Val_Labels, Test_Labels, Train, Val, Test, Tables = pre_processing(Data, AUGMENT, VAL_RATIO, TEST_RATIO)
Split_Data = Split_Data(Tables, Data.FSs, Data.Labels, Data.labels_name, Train, Val, Test,
                        Train_Labels, Val_Labels, Test_Labels)
Split_Data = reshape_for_amerilike(Split_Data)
test_losses = model.evaluate(Split_Data.Test[0], Split_Data.Test_Labels, batch_size=600)

Test_predicted_labels = pd.DataFrame().reindex_like(Split_Data.Test_Labels)
Test_predicted_labels.iloc[:,:] = model.predict(np.array(Split_Data.Test[0]), batch_size=600)

fig, ax = plt.subplots(6, 1, figsize=(16, 4), sharex=True, sharey=True)

#RAW PREDICTIONS
for i in range(1,7):
    #plt.figure(figsize = (16,4))
    ax_i = ax[i-1]
    ax_i.plot(Test_predicted_labels['Right_Hand_channel{}'.format(i)].values)
    ax_i.plot(Test_Labels['Right_Hand_channel{}'.format(i)].values)

plt.show()

print("Test loss =", test_losses)
#fig.savefig(SAVEFIG)