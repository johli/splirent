from __future__ import print_function
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM, ConvLSTM2D, BatchNormalization
from keras.layers import Concatenate, Reshape
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from keras import backend as K
import keras.losses

import tensorflow as tf

import pandas as pd

import os
import pickle
import numpy as np

import scipy.sparse as sp
import scipy.io as spio

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import isolearn.keras as iso

from splirent.losses import *

from splirent.data.splirent_data_5ss import load_data
from splirent.model.splirent_model_only_random_regions import load_splirent_model


#Wrapper function to execute SPLIRENT trainer
def run_trainer(load_data_func, load_model_func, load_saved_model, save_dir_path, load_name_suffix, save_name_suffix, epochs, batch_size, valid_set_size, test_set_size, sequence_padding, file_path, data_version, use_shifter, targeted_a5ss_file_path, drop, chosen_optimizer) :

    #Load plasmid data #_w_array_part_1
    data_gens = load_data_func(batch_size=batch_size, valid_set_size=valid_set_size, test_set_size=test_set_size, sequence_padding=sequence_padding, file_path=file_path, data_version=data_version, use_shifter=use_shifter, targeted_a5ss_file_path=targeted_a5ss_file_path)

    #Load model definition
    models = load_model_func(sequence_padding, drop=drop)
    _, loss_model = models[-1]


    #Optimizer code
    save_dir = os.path.join(os.getcwd(), save_dir_path)

    checkpoint_dir = os.path.join(os.getcwd(), 'model_checkpoints')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if load_saved_model :
        for model_name_prefix, model in models[:-1] :
            model_name = 'splirent_' + model_name_prefix + '_' + load_name_suffix + '.h5'
            model_path = os.path.join(save_dir, model_name)
            saved_model = load_model(model_path)
            
            model.set_weights(saved_model.get_weights())

    opt = None
    if chosen_optimizer == 'sgd' :
        opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    elif chosen_optimizer == 'adam' :
        opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    loss_model.compile(loss=lambda true, pred: pred, optimizer=opt)

    callbacks = [
        ModelCheckpoint(os.path.join(checkpoint_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1),
        EarlyStopping(monitor='val_loss', min_delta=0.0002, patience=3, verbose=0, mode='auto')
    ]

    loss_model.fit_generator(generator=data_gens['train'],
                        validation_data=data_gens['valid'],
                        epochs=epochs,
                        use_multiprocessing=True,
                        workers=12,
                        callbacks=callbacks)


    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for model_name_prefix, model in models[:-1] :
        model_name = 'aparent_' + model_name_prefix + '_' + save_name_suffix + '.h5'
        model_path = os.path.join(save_dir, model_name)
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)


#Execute Trainer if called from cmd-line
if __name__ == "__main__" :

    #Trainer parameters
    save_dir_path = '../../saved_models'
    save_name_suffix = 'sgd'#'sgd_targeted_a5ss'#'adam_targeted_a5ss_neg_rate_1'#'adam'#'adam_neg_rate_1'#'sgd'
    epochs = 15#10
    batch_size = 32

    file_path = '../../data/a5ss/processed_data/'
    data_version = ''#''#'_neg_rate_1'
    targeted_a5ss_file_path = None#'../../data/targeted_a5ss/processed_data/'#None#'../../data/targeted_a5ss/processed_data/'

    sequence_padding = 5
    use_shifter = False#False#True
    drop = 0.2
    chosen_optimizer = 'sgd'

    valid_set_size = 0.05#10000
    test_set_size = 0.05#10000

    run_trainer(load_data, load_splirent_model, False, save_dir_path, save_name_suffix, save_name_suffix, epochs, batch_size, valid_set_size, test_set_size, sequence_padding, file_path, data_version, use_shifter, targeted_a5ss_file_path, drop, chosen_optimizer)
