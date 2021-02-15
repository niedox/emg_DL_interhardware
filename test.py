import pickle
from emg_data import Raw_Data

with open('./Data/Raw_Data_6ch_User1.pkl', 'rb') as f:
    Data = pickle.load(f)

print(Data.recs)
