from __future__ import print_function
import keras
from keras import backend as K

import tensorflow as tf

import pandas as pd

import os
import pickle
import numpy as np

import scipy.sparse as sp
import scipy.io as spio

import isolearn.io as isoio
import isolearn.keras as iso

def iso_normalizer(t) :
    iso = 0.0
    if np.sum(t) > 0.0 :
        iso = t[120] / np.sum(t)
    
    return iso

def cut_normalizer(t) :
    cuts = np.concatenate([np.zeros(240), np.array([1.0])])
    if np.sum(t) > 0.0 :
        cuts = t / np.sum(t)
    
    return cuts

def prepend_targeted_a5ss_data(plasmid_dict, file_path='') :

    logic_df = pd.read_csv(file_path + 'a5ss_splice_logic_library_mapped.csv', sep='\t')
    agg_splice_dict = spio.loadmat(file_path + 'a5ss_splice_logic_counts.mat')

    up_background = 'ttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaagtgaacttcaagatccgccacaacat'.upper()
    dn_background = 'CTTATCACCTTCGTGGCTacagagtttccttatttgtctctgttgccggcttatatggacaagcatatcacagccatttatcggagcgcctccgtacacgctattatcggacgcctcgcgagatcaatacgtatacca'.upper()

    logic_df = logic_df[['sequence']].copy()
    logic_df['padded_seq'] = up_background + logic_df['sequence'] + dn_background
    logic_df['padded_seq'] = logic_df['padded_seq'].str.slice(0, 260)
    logic_df = logic_df.rename(columns={'padded_seq' : 'seq'})

    logic_df['library'] = 1000
    logic_df['origin'] = ['a5ss_logic'] * len(logic_df)

    padded_c_HEK, padded_c_HELA, padded_c_MCF7, padded_c_CHO = [
        sp.csr_matrix(
            sp.hstack([
                sp.csc_matrix((agg_splice_dict[cell_line].shape[0], 130)),
                agg_splice_dict[cell_line][:, 32:115],
                sp.csc_matrix((agg_splice_dict[cell_line].shape[0], 47)),
                sp.csc_matrix(np.array(agg_splice_dict[cell_line][:, 115].todense()).reshape(-1, 1))
            ])
        )
        for cell_line in ['hek', 'hela', 'mcf7', 'cho']
    ]

    print("Aligned targeted a5ss data.")
    
    print('padded_c_HEK.shape = ' + str(padded_c_HEK.shape))
    print('padded_c_HELA.shape = ' + str(padded_c_HELA.shape))
    print('padded_c_MCF7.shape = ' + str(padded_c_MCF7.shape))
    print('padded_c_CHO.shape = ' + str(padded_c_CHO.shape))

    plasmid_dict['min_df'] = pd.concat([logic_df, plasmid_dict['min_df']])
    plasmid_dict['min_hek_count'] = sp.vstack([padded_c_HEK, plasmid_dict['min_hek_count']])
    plasmid_dict['min_hela_count'] = sp.vstack([padded_c_HELA, plasmid_dict['min_hela_count']])
    plasmid_dict['min_mcf7_count'] = sp.vstack([padded_c_MCF7, plasmid_dict['min_mcf7_count']])
    plasmid_dict['min_cho_count'] = sp.vstack([padded_c_CHO, plasmid_dict['min_cho_count']])

    return plasmid_dict

def get_splice_shifter() :
    shift_range = (np.arange(21, dtype=np.int) - 10)
    shift_probs = np.zeros(shift_range.shape[0])

    shift_probs[int(shift_range.shape[0]/2)] = 0.75

    shift_probs[int(shift_range.shape[0]/2) - 1] = 0.05
    shift_probs[int(shift_range.shape[0]/2) + 1] = 0.05

    shift_probs[int(shift_range.shape[0]/2) - 2] = 0.025
    shift_probs[int(shift_range.shape[0]/2) + 2] = 0.025

    shift_probs[int(shift_range.shape[0]/2) - 3] = 0.0125
    shift_probs[int(shift_range.shape[0]/2) + 3] = 0.0125
    shift_probs[int(shift_range.shape[0]/2) - 4] = 0.0125
    shift_probs[int(shift_range.shape[0]/2) + 4] = 0.0125

    shift_probs[int(shift_range.shape[0]/2) - 5] = 0.005
    shift_probs[int(shift_range.shape[0]/2) + 5] = 0.005
    shift_probs[int(shift_range.shape[0]/2) - 6] = 0.005
    shift_probs[int(shift_range.shape[0]/2) + 6] = 0.005
    shift_probs[int(shift_range.shape[0]/2) - 7] = 0.005
    shift_probs[int(shift_range.shape[0]/2) + 7] = 0.005
    shift_probs[int(shift_range.shape[0]/2) - 8] = 0.005
    shift_probs[int(shift_range.shape[0]/2) + 8] = 0.005
    shift_probs[int(shift_range.shape[0]/2) - 9] = 0.005
    shift_probs[int(shift_range.shape[0]/2) + 9] = 0.005
    shift_probs[int(shift_range.shape[0]/2) - 10] = 0.005
    shift_probs[int(shift_range.shape[0]/2) + 10] = 0.005

    shift_probs /= np.sum(shift_probs)
    
    return iso.PositionShifter(shift_range, shift_probs)

def load_data(batch_size=32, valid_set_size=10000, test_set_size=10000, sequence_padding=5, file_path='', data_version='', use_shifter=False, targeted_a5ss_file_path=None) :

    #Load plasmid data
    plasmid_dict = pickle.load(open(file_path + 'alt_5ss_data_aligned' + data_version + '.pickle', 'rb'))

    if targeted_a5ss_file_path is not None :
        plasmid_dict = prepend_targeted_a5ss_data(plasmid_dict, file_path=targeted_a5ss_file_path)
    
    #unique_libraries = sorted(plasmid_dict['min_df']['library'].unique())
    unique_libraries = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1000]

    #Generate training and test set indexes
    plasmid_index = np.arange(len(plasmid_dict['min_df']), dtype=np.int)

    if valid_set_size < 1.0 and test_set_size < 1.0 :
        train_index = plasmid_index[:-int(len(plasmid_dict['min_df']) * (valid_set_size + test_set_size))]
        valid_index = plasmid_index[train_index.shape[0]:-int(len(plasmid_dict['min_df']) * test_set_size)]
        test_index = plasmid_index[train_index.shape[0] + valid_index.shape[0]:]
    else :
        train_index = plasmid_index[:-int(valid_set_size + test_set_size)]
        valid_index = plasmid_index[train_index.shape[0]:-int(test_set_size)]
        test_index = plasmid_index[train_index.shape[0] + valid_index.shape[0]:]

    print('Training set size = ' + str(train_index.shape[0]))
    print('Validation set size = ' + str(valid_index.shape[0]))
    print('Test set size = ' + str(test_index.shape[0]))

    pos_shifter = get_splice_shifter()

    splicing_gens = {
        gen_id : iso.DataGenerator(
            idx,
            {
                'df' : plasmid_dict['min_df'],
                'hek_count' : plasmid_dict['min_hek_count'],
                'hela_count' : plasmid_dict['min_hela_count'],
                'mcf7_count' : plasmid_dict['min_mcf7_count'],
                'cho_count' : plasmid_dict['min_cho_count'],
            },
            batch_size=32,
            inputs = [
                {
                    'id' : 'full_sequence',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : iso.SequenceExtractor('seq', start_pos=10, end_pos=250, shifter=pos_shifter if gen_id == 'train' and use_shifter else None),
                    'encoder' : iso.OneHotEncoder(seq_length=240),
                    'dim' : (240, 4),
                    'sparsify' : False
                },
                {
                    'id' : 'lib',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['library'],
                    'encoder' : iso.CategoricalEncoder(n_categories=len(unique_libraries), categories=unique_libraries),
                    'dim' : (100,),
                    'sparsify' : False
                }
            ],
            outputs = [
                {
                    'id' : cell_type + '_sd1_usage',
                    'source_type' : 'matrix',
                    'source' : cell_type + '_count',
                    'extractor' : iso.CountExtractor(start_pos=10, end_pos=250, static_poses=[-1], shifter=pos_shifter if gen_id == 'train' and use_shifter else None, sparse_source=False),
                    'transformer' : lambda t: cut_normalizer(t),
                    'dim' : (241,)
                } for cell_type in ['hek', 'hela', 'mcf7', 'cho']
            ] + [
                {
                    'id' : cell_type + '_trainable',
                    'source_type' : 'matrix',
                    'source' : cell_type + '_count',
                    'extractor' : iso.CountExtractor(start_pos=10, end_pos=250, static_poses=[-1], shifter=pos_shifter if gen_id == 'train' and use_shifter else None, sparse_source=False),
                    'transformer' : lambda t: 1 if np.sum(t) > 0.0 else 0,
                    'dim' : (1,)
                } for cell_type in ['hek', 'hela', 'mcf7', 'cho']
            ],
            randomizers = [pos_shifter] if gen_id == 'train' and use_shifter else [],
            shuffle = True if gen_id in ['train'] else False,
            densify_batch_matrices=True,
            move_outputs_to_inputs=True if gen_id in ['train', 'valid'] else False
        ) for gen_id, idx in [('train', train_index), ('valid', valid_index), ('test', test_index)]
    }

    return splicing_gens
