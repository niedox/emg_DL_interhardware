
# The architecture of the CNN model largely inspired from Ameri et al.,2019
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

def create_CNN_Ameri_like_model(dim1,dim2,dim_labels,  lambda_ = 1.675e-5, drop_rate = 0.1, nbr_hid1 = 80, nbr_hid2 = 24, kernel_initializer='glorot_uniform'):

    input_dim = (dim1,dim2,1)

    visible = Input(shape = input_dim)

    ######################################## CONVOLUTIONAL LAYERS ##########################################
    # First Block
    conv1 = Conv2D(16, kernel_size = (3,3), padding = 'same', kernel_regularizer= l2(lambda_), kernel_initializer=kernel_initializer)(visible)
    norm1 = BatchNormalization()(conv1)
    act1 = ReLU()(norm1)
    pool1 = AveragePooling2D(pool_size = (2,2))(act1)

    # Second Block
    conv2 = Conv2D(16, kernel_size = (3,3), padding = 'same', kernel_regularizer= l2(lambda_), kernel_initializer=kernel_initializer)(pool1)
    norm2 = BatchNormalization()(conv2)
    act2 = ReLU()(norm2)
    pool2 = AveragePooling2D(pool_size = (2,2))(act2)

    # Third Block
    conv3 = Conv2D(64, kernel_size = (3,3), padding = 'same', kernel_regularizer= l2(lambda_), kernel_initializer=kernel_initializer)(pool2)
    norm3 = BatchNormalization()(conv3)
    act3 = ReLU()(norm3)
    pool3 = AveragePooling2D(pool_size = (2,2))(act3)

    # Fourth Block
    conv4 = Conv2D(64, kernel_size = (3,3), padding = 'same', kernel_regularizer= l2(lambda_), kernel_initializer=kernel_initializer)(pool3)
    norm4 = BatchNormalization()(conv4)
    act4 = ReLU()(norm4)
    pool4 = AveragePooling2D(pool_size = (2,2))(act4)

    # Fifth Block
    conv5 = Conv2D(64, kernel_size = (3,3), padding = 'same', kernel_regularizer= l2(lambda_), kernel_initializer=kernel_initializer)(pool4)
    norm5 = BatchNormalization()(conv5)
    act5 = ReLU()(norm5)
    pool5 = AveragePooling2D(pool_size = (2,2))(act5)

    # Sixth Block
    conv6 = Conv2D(64, kernel_size = (3,3), padding = 'same', kernel_regularizer= l2(lambda_), kernel_initializer=kernel_initializer)(pool5)
    norm6 = BatchNormalization()(conv6)
    act6 = ReLU()(norm6)

    # Seventh Block
    conv7 = Conv2D(16, kernel_size = (3,3), padding = 'same', kernel_regularizer= l2(lambda_), kernel_initializer=kernel_initializer)(act6)
    norm7 = BatchNormalization()(conv7)
    act7 = ReLU()(norm7)

    # Eigthth Block
    conv8 = Conv2D(16, kernel_size = (3,3), padding = 'same', kernel_regularizer= l2(lambda_), kernel_initializer=kernel_initializer)(act7)
    norm8 = BatchNormalization()(conv8)
    act8 = ReLU()(norm8)

    ######################################## FULLY CONNECTED LAYERS #########################################

    # Fully connected layers 
    flat = Flatten()(act8)
    hid1 = Dense(nbr_hid1, activation = 'relu', kernel_regularizer = l2(lambda_))(flat)
    drop1 = Dropout(rate = drop_rate)(hid1)

    hid2 = Dense(nbr_hid2, activation = 'relu', kernel_regularizer = l2(lambda_))(drop1)
    drop2 = Dropout(rate = drop_rate)(hid2)

    ################################################ OUTPUT ################################################

    output = Dense(dim_labels, activation = 'relu', kernel_regularizer = l2(lambda_))(drop2)

    model = Model(inputs = visible, outputs = output)

    # Compile model 
    model.compile(optimizer = 'SGD', loss = 'mean_squared_error')

    return model

def step_decay(epoch):
    #We half the the learning rate each 10 epochs
    initial_l_r = 0.01 
    drop = 0.5 
    epochs_drop = 10
    lrate = initial_l_r * np.power(drop,np.floor((1 + epoch)/epochs_drop))
    # if you want to avoid the learning rate decay, uncomment this line.
    #lrate = initial_l_r
    #print('The learning rate is : ', lrate)
    return lrate

def CreateFit_ameri_like(x_train,y_train,x_val,y_val,params):
    
    lambda_ = params['lambda_']
    drop_rate = params['drop_rate']
    nbr_hid1 = params['nbr_filters_hid1']
    nbr_hid2 = params['nbr_filters_hid2']
    kernel_initializer = params['kernel_initializer']

    dim1 = params['dim1']
    dim2 = params['dim2']
    dim_labels = params['dim_labels']

    model = create_CNN_Ameri_like_model(dim1,dim2,dim_labels,  lambda_, drop_rate, nbr_hid1, nbr_hid2, kernel_initializer)
    
    callbacks_list = params['callbacks_list']
    epochs = params['epochs']
    batch_size = params['batch_size']

    out = model.fit(x_train,y_train, epochs = epochs, batch_size = batch_size, verbose = 0, callbacks = callbacks_list, validation_data = [x_val, y_val])
    
    return out, model

def CreateFit_ameri_like_dataset(tr_dataset,val_dataset,params):
    
    lambda_ = params['lambda_']
    drop_rate = params['drop_rate']
    nbr_hid1 = params['nbr_filters_hid1']
    nbr_hid2 = params['nbr_filters_hid2']
    kernel_initializer = params['kernel_initializer']

    dim1 = params['dim1']
    dim2 = params['dim2']
    dim_labels = params['dim_labels']

    model = create_CNN_Ameri_like_model(dim1,dim2,dim_labels,  lambda_, drop_rate, nbr_hid1, nbr_hid2, kernel_initializer)
    
    callbacks_list = params['callbacks_list']
    epochs = params['epochs']
    batch_size = params['batch_size']

    tot_correct_files_tr = params['tot_correct_files_tr']
    tot_correct_files_val = params['tot_correct_files_val']

    out = model.fit(tr_dataset, steps_per_epoch= int(np.floor(tot_correct_files_tr / batch_size))-1,
                  validation_data=val_dataset,
                  validation_steps=int(np.floor(tot_correct_files_val / batch_size)),    
                  callbacks=callbacks_list,
                  epochs=epochs,
                   )

    return out, model
###########################################################################################

