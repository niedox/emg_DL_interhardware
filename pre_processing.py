
import pickle
from bioml import Align_Streams as AS
from bioml import Data_Cutter as DC
from bioml import Data_PreProcessing as DPP
from emg_data import Split_Data


AUGMENT = 1
DROP = 0
win_len = 192
step = 10

def pre_processing(rd, augment):


    Labels = rd.Labels
    Tables = rd.Tables
    recs = rd.recs
    FSs = rd.FSs

    print('Correct Labels')
    Labels = DPP.L_correct_Labels_6channels(Labels, 'Right')
    # Align targets with 1st stream only, it should be close to the second stream since it will be the same number of windows
    print('Align streams and targets')
    Tables, Labels = AS.HH_Align(Tables, Labels)

    ##Normalize w/ ref electrode
    # print('Normalize with reference electrode')
    # Tables = DPP.H_NormRef_Data(Tables, channels, refs)

    # Filter notch 50 + 15-500 Hz
    # print('Filter signal (Bandpass + Notch)')
    # Tables = DPP.H_filter_data(Tables, FSs, recs)

    # Cut data
    print('Create three sets: train, validation and test')
    Tables, Labels = DC.H_Cut_Data_on_percentage(Tables, Labels, val_percentage=1 / 6, test_percentage=1 / 6)

    if augment:
        # Augment data by adding channel 6 in first column and channel 1 at last column
        # (To take into account the ring-like organisation of electrodes)
        Tables[0].insert(6, 'Labjack_channel1bis', Tables[0]['Labjack_channel1'], allow_duplicates=False)
        Tables[0].insert(0, 'Labjack_channel6bis', Tables[0]['Labjack_channel6'], allow_duplicates=False)

    # Standardize (mean=0 and std=1)
    # print('Standardize channels on train dataset')
    Tables, stdscale = DPP.H_standardize_training_data(Tables, recs[0])

    # Standardize based on training set to avoid overfitting
    # print('Standardize channels on validation and test datasets')
    Tables = DPP.H_standardize_val_test_data(Tables, recs[0], stdscale)

    # Windowing
    print('Create overlapping time windows')
    Train, Val, Test = DPP.H_cut_time_windows(Tables, FSs, win_len / 1000,
                                              step / 1000)  # if epoching it will arrive here

    # Align targets on windows
    print('Align targets on windows')
    Train_Labels = AS.H_align_with_windows(Train, Labels)
    Val_Labels = AS.H_align_with_windows(Val, Labels)
    Test_Labels = AS.H_align_with_windows(Test, Labels)

    # Rescale labels to be between 0 and 10, here to avoid having recording, set, and subject
    print('Rescale targets')
    Train_Labels = DPP.L_scale_labels(Train_Labels, new_max_value=10)
    Test_Labels = DPP.L_scale_labels(Test_Labels, new_max_value=10)
    if not Val_Labels.empty:
        Val_Labels = DPP.L_scale_labels(Val_Labels, new_max_value=10)

    return Train_Labels, Val_Labels, Test_Labels, Train, Val, Test, Tables


with open('./Data/Raw_Data_6ch.pkl', 'rb') as f:
    Raw_Data = pickle.load(f)

if DROP:
    rem = [2,6]
    Raw_Data.Tables[0] = Raw_Data.Tables[0].drop(["Labjack_channel" + str(rem[0]), "Labjack_channel" + str(rem[1])],
                               axis='columns')
    Raw_Data.Tables[0].columns = ["Labjack_channel1", "Labjack_channel2", "Labjack_channel3", "Labjack_channel4",
                         "Labjack_channel5",
                         "Labjack_channel6", "subject_id", "condition"]

Train_Labels, Val_Labels, Test_Labels, Train, Val, Test, Tables = pre_processing(Raw_Data, AUGMENT)


Emg_Data_8ch = Split_Data(Tables, Raw_Data.FSs, Raw_Data.Labels, Raw_Data.labels_name, Train, Val, Test,
                        Train_Labels, Val_Labels, Test_Labels)


with open('./Data/Emg_Data_6ch_augm.pkl', 'wb') as f:
    pickle.dump(Emg_Data_8ch, f)

