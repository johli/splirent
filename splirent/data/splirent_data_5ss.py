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
        iso = t[0] / np.sum(t)
    
    return iso

def cut_normalizer(t) :
    cuts = np.concatenate([np.zeros(100), np.array([1.0])])
    if np.sum(t) > 0.0 :
        cuts = t / np.sum(t)
    
    return cuts

def prepend_targeted_a5ss_data(plasmid_dict, file_path='') :

    logic_df = pd.read_csv(file_path + 'a5ss_splice_logic_library_mapped.csv', sep='\t')
    agg_splice_dict = spio.loadmat(file_path + 'a5ss_splice_logic_counts.mat')

    up_background = 'gggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaagtgaacttcaagatccgccacaacat'.upper()
    dn_background = 'CTTATCACCTTCGTGGCTacagagtttccttatttgtctctgttgccggcttatatggacaagcatatcacagccatttatcggagcgcctccgtacacgctattatcggacgcctcgcgagatcaatacgtatacca'.upper()

    logic_df = logic_df[['sequence']].copy()
    logic_df['padded_seq'] = up_background + logic_df['sequence'] + dn_background

    padded_c_HEK, padded_c_HELA, padded_c_MCF7, padded_c_CHO = [
        sp.csr_matrix(
            sp.hstack([
                sp.csc_matrix((agg_splice_dict[cell_line].shape[0], 140)),
                agg_splice_dict[cell_line][:, 32:115],
                sp.csc_matrix((agg_splice_dict[cell_line].shape[0], 120 + 18)),
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

def load_data(batch_size=32, valid_set_size=10000, test_set_size=10000, sequence_padding=5, file_path='', data_version='', use_shifter=False, targeted_a5ss_file_path=None) :

    #Load plasmid data
    plasmid_dict = pickle.load(open(file_path + 'alt_5ss_data' + data_version + '.pickle', 'rb'))

    if targeted_a5ss_file_path is not None :
        plasmid_dict = prepend_targeted_a5ss_data(plasmid_dict, file_path=targeted_a5ss_file_path)
    
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
    

    seq_start_1 = 147
    seq_start_2 = 147 + 25 + 18
    splice_donor_pos = 140

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
                    'extractor' : lambda row, index: row['padded_seq'][seq_start_1 - sequence_padding: seq_start_2 + 25 + sequence_padding],
                    'encoder' : iso.OneHotEncoder(seq_length=(seq_start_2 + 25 - seq_start_1) + 2 * sequence_padding),
                    'dim' : ((seq_start_2 + 25 - seq_start_1) + 2 * sequence_padding, 4),
                    'sparsify' : False
                }
            ],
            outputs = [
                {
                    'id' : cell_type + '_sd1_usage',
                    'source_type' : 'matrix',
                    'source' : cell_type + '_count',
                    'extractor' : iso.CountExtractor(start_pos=splice_donor_pos, end_pos=splice_donor_pos + 100, static_poses=[-1], sparse_source=False),
                    'transformer' : lambda t: iso_normalizer(t),
                    'dim' : (1,)
                } for cell_type in ['hek', 'hela', 'mcf7', 'cho']
            ] + [
                {
                    'id' : cell_type + '_trainable',
                    'source_type' : 'matrix',
                    'source' : cell_type + '_count',
                    'extractor' : iso.CountExtractor(start_pos=splice_donor_pos, end_pos=splice_donor_pos + 100, static_poses=[-1], sparse_source=False),
                    'transformer' : lambda t: 1 if np.sum(t) > 0.0 else 0,
                    'dim' : (1,)
                } for cell_type in ['hek', 'hela', 'mcf7', 'cho']
            ],
            randomizers = [],
            shuffle = True if gen_id in ['train'] else False,
            densify_batch_matrices=True,
            move_outputs_to_inputs=True if gen_id in ['train', 'valid'] else False
        ) for gen_id, idx in [('train', train_index), ('valid', valid_index), ('test', test_index)]
    }

    return splicing_gens
