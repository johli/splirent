from __future__ import print_function
import keras
from keras import backend as K
import keras.losses

import tensorflow as tf

#Keras loss functions

def get_cross_entropy() :
    
    def cross_entropy(inputs) :
        y_true, y_pred = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())

        return -K.sum(y_true * K.log(y_pred), axis=-1)
    
    return cross_entropy

def get_mean_cross_entropy() :
    
    def mean_cross_entropy(inputs) :
        y_true, y_pred = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())

        return -K.mean(y_true * K.log(y_pred), axis=-1)
    
    return mean_cross_entropy

def get_sigmoid_entropy() :
    
    def sigmoid_entropy(inputs) :
        y_true, y_pred = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())

        return -K.sum(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred), axis=-1)
    
    return sigmoid_entropy

def get_mean_sigmoid_entropy() :
    
    def mean_sigmoid_entropy(inputs) :
        y_true, y_pred = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())

        return -K.mean(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred), axis=-1)
    
    return mean_sigmoid_entropy

def get_kl_divergence() :
    
    def kl_divergence(inputs) :
        y_true, y_pred = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())

        return K.sum(y_true * K.log(y_true / y_pred), axis=-1)
    
    return kl_divergence

def get_mean_kl_divergence() :
    
    def mean_kl_divergence(inputs) :
        y_true, y_pred = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())

        return K.mean(y_true * K.log(y_true / y_pred), axis=-1)
    
    return mean_kl_divergence

def get_sigmoid_kl_divergence() :
    
    def sigmoid_kl_divergence(inputs) :
        y_true, y_pred = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())

        return K.sum(y_true * K.log(y_true / y_pred) + (1.0 - y_true) * K.log((1.0 - y_true) / (1.0 - y_pred)), axis=-1)
    
    return sigmoid_kl_divergence

def get_mean_sigmoid_kl_divergence() :
    
    def mean_sigmoid_kl_divergence(inputs) :
        y_true, y_pred = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())

        return K.mean(y_true * K.log(y_true / y_pred) + (1.0 - y_true) * K.log((1.0 - y_true) / (1.0 - y_pred)), axis=-1)
    
    return mean_sigmoid_kl_divergence

def get_symmetric_kl_divergence() :
    
    def symmetric_kl_divergence(inputs) :
        y_true, y_pred = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())

        return K.sum(y_true * K.log(y_true / y_pred) + y_pred * K.log(y_pred / y_true), axis=-1)
    
    return symmetric_kl_divergence

def get_mean_symmetric_kl_divergence() :
    
    def mean_symmetric_kl_divergence(inputs) :
        y_true, y_pred = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())

        return K.mean(y_true * K.log(y_true / y_pred) + y_pred * K.log(y_pred / y_true), axis=-1)
    
    return mean_symmetric_kl_divergence

def get_symmetric_sigmoid_kl_divergence() :
    
    def symmetric_sigmoid_kl_divergence(inputs) :
        y_true, y_pred = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())

        return K.sum(y_true * K.log(y_true / y_pred) + (1.0 - y_true) * K.log((1.0 - y_true) / (1.0 - y_pred)) + y_pred * K.log(y_pred / y_true) + (1.0 - y_pred) * K.log((1.0 - y_pred) / (1.0 - y_true)), axis=-1)
    
    return symmetric_sigmoid_kl_divergence

def get_mean_symmetric_sigmoid_kl_divergence() :
    
    def mean_symmetric_sigmoid_kl_divergence(inputs) :
        y_true, y_pred = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())

        return K.mean(y_true * K.log(y_true / y_pred) + (1.0 - y_true) * K.log((1.0 - y_true) / (1.0 - y_pred)) + y_pred * K.log(y_pred / y_true) + (1.0 - y_pred) * K.log((1.0 - y_pred) / (1.0 - y_true)), axis=-1)
    
    return mean_symmetric_sigmoid_kl_divergence

