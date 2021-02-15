
from bioml import Align_Streams as AS
from bioml import Data_Cutter as DC
from bioml import Data_PreProcessing as DPP
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
import bioml.models.CNN_Ameri_like_model as MODELS
import numpy as np
from CNN_Ameri_like_model import step_decay


win_len = 192
step = 10


def pre_processing(rd, augment, val_ratio, test_ratio):
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
    Tables, Labels = DC.H_Cut_Data_on_percentage(Tables, Labels, val_percentage=val_ratio, test_percentage=test_ratio)

    if augment:
        # Augment data by adding channel 6 in first column and channel 1 at last column
        # (To take into account the ring-like organisation of electrodes)
        print("Data augmentation")
        Tables[0].insert(6, 'Labjack_channel1bis', Tables[0]['Labjack_channel1'], allow_duplicates=False)
        Tables[0].insert(0, 'Labjack_channel6bis', Tables[0]['Labjack_channel6'], allow_duplicates=False)

    # Standardize (mean=0 and std=1)
    print('Standardize channels on train dataset')
    Tables, stdscale = DPP.H_standardize_training_data(Tables, recs)

    # Standardize based on training set to avoid overfitting
    print('Standardize channels on validation and test datasets')
    Tables = DPP.H_standardize_val_test_data(Tables, recs, stdscale)

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


def reshape_for_amerilike(Data):
    Train = Data.Train
    Val = Data.Validate
    Test = Data.Test

    # reshape input for ameri-like
    # For CNN input needs to be 3d
    for i in range(len(Train[0])):
        Train[0][i] = Train[0][i].values.reshape((48, -1, 1))

    for i in range(len(Val[0])):
        Val[0][i] = Val[0][i].values.reshape((48, -1, 1))

    for i in range(len(Test[0])):
        Test[0][i] = Test[0][i].values.reshape((48, -1, 1))

    Train[0] = np.array(Train[0])
    Val[0] = np.array(Val[0])
    Test[0] = np.array(Test[0])

    Data.Train = Train
    Data.Val = Data.Validate
    Data.Test = Data.Test

    return Data


def train_model_on_data(Data, estimator):
    Train = Data.Train
    Train_Labels = Data.Train_Labels
    Val = Data.Validate
    Val_Labels = Data.Val_Labels


    dim1, dim2, _ = Train[0][0].shape

    # Use model

    print('Train model on data')
    ################################################## TRAINING ##################################################
    # Create model
    lrate_schedule = LearningRateScheduler(step_decay)
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=13, verbose=1, mode='auto', baseline=None,
                                  restore_best_weights=True)
    callbacks_list = [lrate_schedule, early_stopper]

    # Train
    print('INFO: Start Training')

    hist = estimator.fit(Train[0], Train_Labels, batch_size=128, epochs=50, shuffle=True, validation_data=(Val[0], Val_Labels),
                 callbacks=callbacks_list)
    # estimator.fit(Train[0],Train_Labels, batch_size = 128, epochs = 50, shuffle=True, callbacks = callbacks_list)

    # estimator.fit(Train,Train_Labels, batch_size = 640, epochs = 15, shuffle=True)
    return estimator, hist


