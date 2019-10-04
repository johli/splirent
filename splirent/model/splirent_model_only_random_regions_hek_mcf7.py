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

import isolearn.keras as iso

from splirent.losses import *

def load_splirent_model(sequence_padding, drop=0.2) :

    #Inputs
    seq_input_1 = Input(shape=(25 + 2 * sequence_padding, 4))
    seq_input_2 = Input(shape=(25 + 2 * sequence_padding, 4))

    #Outputs
    true_usage_hek = Input(shape=(1,))
    true_usage_hela = Input(shape=(1,))
    true_usage_mcf7 = Input(shape=(1,))
    true_usage_cho = Input(shape=(1,))

    trainable_hek = Input(shape=(1,))
    trainable_hela = Input(shape=(1,))
    trainable_mcf7 = Input(shape=(1,))
    trainable_cho = Input(shape=(1,))

    #Shared Model Definition (Applied to each randomized sequence region)
    conv_layer_1 = Conv1D(96, 8, padding='same', activation='relu')
    pool_layer_1 = MaxPooling1D(pool_size=2)
    conv_layer_2 = Conv1D(128, 6, padding='same', activation='relu')

    def shared_model(seq_input) :
        return Flatten()(
            conv_layer_2(
                pool_layer_1(
                    conv_layer_1(
                        seq_input
                    )
                )
            )
        )

    
    shared_out_1 = shared_model(seq_input_1)
    shared_out_2 = shared_model(seq_input_2)

    #Layers applied to the concatenated hidden representation
    layer_dense = Dense(256, activation='relu')
    layer_drop = Dropout(drop)

    concat_out = Concatenate(axis=-1)([shared_out_1, shared_out_2])

    dense_out = layer_dense(concat_out)
    dropped_out = layer_drop(dense_out)

    #Final cell-line specific regression layers

    layer_usage_hek = Dense(1, activation='sigmoid', kernel_initializer='zeros')
    layer_usage_mcf7 = Dense(1, activation='sigmoid', kernel_initializer='zeros')

    pred_usage_hek = layer_usage_hek(dropped_out)
    pred_usage_mcf7 = layer_usage_mcf7(dropped_out)

    #Compile Splicing Model
    splicing_model = Model(
        inputs=[
            seq_input_1,
            seq_input_2
        ],
        outputs=[
            pred_usage_hek,
            pred_usage_mcf7
        ]
    )

    sigmoid_kl_divergence = get_sigmoid_kl_divergence()

    #Loss Model Definition

    loss_hek = Lambda(sigmoid_kl_divergence, output_shape = (1,))([true_usage_hek, pred_usage_hek])
    loss_mcf7 = Lambda(sigmoid_kl_divergence, output_shape = (1,))([true_usage_mcf7, pred_usage_mcf7])


    total_loss = Lambda(
        lambda l: (l[0] * l[0 + 2] + l[1] * l[1 + 2]) / (l[0 + 2] + l[1 + 2]),
        output_shape = (1,)
    )(
        [
            loss_hek,
            loss_mcf7,
            trainable_hek,
            trainable_mcf7
        ]
    )

    #Must be the same order as defined in the data generators
    loss_model = Model([
        #Inputs
        seq_input_1,
        seq_input_2,
        
        #Target SD Usages
        true_usage_hek,
        true_usage_hela,
        true_usage_mcf7,
        true_usage_cho,

        #Trainable cell types
        trainable_hek,
        trainable_hela,
        trainable_mcf7,
        trainable_cho
    ], total_loss)

    return [ ('splirent_only_random_regions_drop_' + str(drop).replace(".", "") + '_hek_mcf7', splicing_model), ('loss', loss_model) ]





