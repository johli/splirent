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
    seq_input = Input(shape=(240, 4))
    lib_input = Input(shape=(100,))

    #Outputs
    true_cuts_hek = Input(shape=(241,))
    true_cuts_hela = Input(shape=(241,))
    true_cuts_mcf7 = Input(shape=(241,))
    true_cuts_cho = Input(shape=(241,))

    trainable_hek = Input(shape=(1,))
    trainable_hela = Input(shape=(1,))
    trainable_mcf7 = Input(shape=(1,))
    trainable_cho = Input(shape=(1,))

    #Shared Model Definition (Applied to each randomized sequence region)
    conv_layer_1 = Conv1D(96, 8, padding='same')
    batch_norm_1 = BatchNormalization()
    relu_1 = Lambda(lambda x: K.relu(x))

    pool_layer_1 = MaxPooling1D(pool_size=2)

    conv_layer_2 = Conv1D(128, 6, padding='same')
    batch_norm_2 = BatchNormalization()
    relu_2 = Lambda(lambda x: K.relu(x))

    def shared_model(seq_input) :
        return Flatten()(
            relu_2(
                batch_norm_2(
                    conv_layer_2(
                        pool_layer_1(
                            relu_1(
                                batch_norm_1(
                                    conv_layer_1(
                                        seq_input
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

    
    shared_out = shared_model(seq_input)

    #Layers applied to the concatenated hidden representation
    layer_dense_1 = Dense(256)
    batch_norm_dense_1 = BatchNormalization()
    relu_dense_1 = Lambda(lambda x: K.relu(x))
    layer_drop_1 = Dropout(drop)

    layer_dense_2 = Dense(256)
    layer_drop_2 = Dropout(drop)
    batch_norm_dense_2 = BatchNormalization()
    relu_dense_2 = Lambda(lambda x: K.relu(x))

    dense_out_1 = relu_dense_1(batch_norm_dense_1(layer_dense_1(shared_out)))
    dropped_out_1 = layer_drop_1(dense_out_1)
    dense_out_2 = relu_dense_2(batch_norm_dense_2(layer_dense_2(dropped_out_1)))
    dropped_out_2 = layer_drop_2(dense_out_2)

    concat_out = Concatenate(axis=-1)([dropped_out_2, lib_input])

    #Final cell-line specific regression layers

    layer_cuts_hek = Dense(241, activation='softmax', kernel_initializer='zeros')
    layer_cuts_hela = Dense(241, activation='softmax', kernel_initializer='zeros')
    layer_cuts_mcf7 = Dense(241, activation='softmax', kernel_initializer='zeros')
    layer_cuts_cho = Dense(241, activation='softmax', kernel_initializer='zeros')

    pred_cuts_hek = layer_cuts_hek(concat_out)
    pred_cuts_hela = layer_cuts_hela(concat_out)
    pred_cuts_mcf7 = layer_cuts_mcf7(concat_out)
    pred_cuts_cho = layer_cuts_cho(concat_out)

    #Compile Splicing Model
    splicing_model = Model(
        inputs=[
            seq_input,
            lib_input
        ],
        outputs=[
            pred_cuts_hek,
            pred_cuts_hela,
            pred_cuts_mcf7,
            pred_cuts_cho
        ]
    )

    kl_divergence = get_kl_divergence()

    #Loss Model Definition
    loss_hek = Lambda(kl_divergence, output_shape = (1,))([true_cuts_hek, pred_cuts_hek])
    loss_hela = Lambda(kl_divergence, output_shape = (1,))([true_cuts_hela, pred_cuts_hela])
    loss_mcf7 = Lambda(kl_divergence, output_shape = (1,))([true_cuts_mcf7, pred_cuts_mcf7])
    loss_cho = Lambda(kl_divergence, output_shape = (1,))([true_cuts_cho, pred_cuts_cho])


    total_loss = Lambda(
        lambda l: (l[0] * l[0 + 4] + l[1] * l[1 + 4] + l[2] * l[2 + 4] + l[3] * l[3 + 4]) / (l[0 + 4] + l[1 + 4] + l[2 + 4] + l[3 + 4]),
        output_shape = (1,)
    )(
        [
            loss_hek,
            loss_hela,
            loss_mcf7,
            loss_cho,
            trainable_hek,
            trainable_hela,
            trainable_mcf7,
            trainable_cho
        ]
    )

    #Must be the same order as defined in the data generators
    loss_model = Model([
        #Inputs
        seq_input,
        lib_input,
        
        #Target SD Usages
        true_cuts_hek,
        true_cuts_hela,
        true_cuts_mcf7,
        true_cuts_cho,

        #Trainable cell types
        trainable_hek,
        trainable_hela,
        trainable_mcf7,
        trainable_cho
    ], total_loss)

    return [ ('splirent_aligned_cuts_large_batchnorm_drop_' + str(drop).replace(".", ""), splicing_model), ('loss', loss_model) ]





