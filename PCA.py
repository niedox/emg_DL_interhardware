import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from machine_learning import pre_processing

DATAPATH = "./Data/Raw_Data_8ch_User1.pkl"
CHANNELS = ['Labjack_channel1','Labjack_channel2','Labjack_channel3','Labjack_channel4','Labjack_channel5',
             'Labjack_channel6', 'Labjack_channel7', 'Labjack_channel9']
with open(DATAPATH, 'rb') as f:
    Data = pickle.load(f)


Train_Labels, Val_Labels, Test_Labels, Train, Val, Test, Tables = pre_processing(Data, 0, 1/6, 1/6);

X = Train[0][CHANNELS]
print(np.transpose(X))

pca = PCA(n_components=6)
pca.fit(np.transpose(X))

print(pca.components_)


fig, axs = plt.subplots(8, 1, sharex=True, figsize=(16,12))
fig.subplots_adjust(hspace=0)

for i,chans in enumerate(CHANNELS):
    axs[i].plot(Data.Tables[0][chans].values)
plt.show()

fig, ax = plt.subplots(6, 1, figsize=(16, 4), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0)

for i in range(0,6):
    #plt.figure(figsize = (16,4))
    ax_i = ax[i]
    ax_i.plot(pca.components_[i])

plt.show()