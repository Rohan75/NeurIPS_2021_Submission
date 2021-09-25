# imports -----------------------------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
import sklearn

import tfomics
from tfomics import moana

import os, shutil
from six.moves import cPickle

import h5py, io
import requests as rq

import utils
import models

import ushuffle

import matplotlib.pyplot as plt

owd = os.getcwd()

# load data -------------------------------------------------------------------------------------------------------------

def load_dataset(file_path):
    
    global X, Y, x_train, y_train, x_valid, y_valid, x_test, y_test, L

    with h5py.File(file_path, 'r') as dataset:
    	X = np.array(dataset['X'])
    	Y = np.array(dataset['Y'])
    	L = np.array(dataset['L'])

    train = int(len(X) * 0.7)
    valid = train + int(len(X) * 0.1 )
    test = valid + int(len(X) * 0.2)

    x_train = X[:train]
    x_valid = X[train:valid]
    x_test = X[valid:test]

    y_train = Y[:train]
    y_valid = Y[train:valid]
    y_test = Y[valid:test]
    
def load_gia_sequences(file_path):
    
    global indep, inter
    
    with h5py.File(file_path, 'r') as dataset:
        indep = np.array(dataset['independent'])
        inter = np.array(dataset['interactions'])
    
    
    
import logomaker
import pandas as pd

def plot_filters(W, fig, num_cols=8, alphabet='ACGT', names=None, fontsize=12):
    """plot 1st layer convolutional filters"""

    num_filter, filter_len, A = W.shape
    num_rows = np.ceil(num_filter/num_cols).astype(int)

    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    for n, w in enumerate(W):
        ax = fig.add_subplot(num_rows,num_cols,n+1)

        # Calculate sequence logo heights -- information
        I = np.log2(4) + np.sum(w * np.log2(w+1e-7), axis=1, keepdims=True)
        logo = I*w

        # Create DataFrame for logomaker
        counts_df = pd.DataFrame(data=0.0, columns=list(alphabet), index=list(range(filter_len)))
        for a in range(A):
            for l in range(filter_len):
                counts_df.iloc[l,a] = logo[l,a]

        logomaker.Logo(counts_df, ax=ax)
        ax = plt.gca()
        ax.set_ylim(0,2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        plt.xticks([])
        plt.yticks([])
        if names is not None:
            plt.ylabel(names[n], fontsize=fontsize)    






# pipeline -------------------------------------------------------------------------------------------------------------
    
def run_pipeline(model, baseline, category, variant, trial, motifs, batch_size=200, epochs=100):
    
    jaspar_ids, motif_names, expecteds = motifs
    
    global x_train, y_train, x_valid, y_valid, x_test, y_test, indep, inter

    # Create directories
    model_dir = os.path.abspath(f'{baseline}/models/{category}/model-{variant}')
    motif_dir = os.path.abspath(f'{baseline}/motifs/{category}/model-{variant}')
    tomtom_dir = os.path.abspath(f'{baseline}/tomtom/{category}/model-{variant}')
    stats_dir = os.path.abspath(f'{baseline}/stats/{category}/model-{variant}')
    logs_dir = os.path.abspath(f'{baseline}/history/{category}/model-{variant}')
    ppms_dir = os.path.abspath(f'{baseline}/ppms/{category}/model-{variant}')
    heat_dir = os.path.abspath(f'{baseline}/heat_maps/{category}/model-{variant}')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(motif_dir):
        os.makedirs(motif_dir)
    if not os.path.exists(tomtom_dir):
        os.makedirs(tomtom_dir)
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    if not os.path.exists(ppms_dir):
        os.makedirs(ppms_dir)
    if not os.path.exists(heat_dir):
        os.makedirs(heat_dir)
    
    model_dir += f'/trial-{trial}/weights'
    motif_dir += f'/trial-{trial}.txt'
    tomtom_dir += f'/trial-{trial}'
    stats_dir += f'/trial-{trial}.npy'
    logs_dir += f'/trial-{trial}.pickle'
    ppms_dir += f'/trial-{trial}.pdf'
    heat_dir += f'/trial-{trial}.pdf'
    
    if os.path.exists(tomtom_dir):
        shutil.rmtree(tomtom_dir)
    
    # get important indices
    lays = [type(i) for i in model.layers]
    c_index = lays.index(tf.keras.layers.MaxPool1D)
    mha_index = lays.index(tfomics.layers.MultiHeadAttention)
    
    # train model ------------------------------------------------------------------------------------------------------
    
    auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
    model.compile(
        tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=[auroc, aupr]
    )

    lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patient=5, verbose=1, min_lr=1e-7, mode='min')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min', restore_best_weights=True)
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_valid, y_valid), callbacks=[lr_decay, early_stop], verbose=2, batch_size=batch_size)
    
    model.save_weights(model_dir) # save model weights
    
    with open(logs_dir, 'wb') as handle:
        cPickle.dump(history.history, handle) # save model history
    
    # evaluate performance --------------------------------------------------------------------------------------------
    
    loss, auc_roc, auc_pr = model.evaluate(x_test, y_test)
    
    # hierachical clustering

    o, att_maps = utils.get_layer_output(model, mha_index, x_test)
    att_maps = np.amax(att_maps, axis=1)
    for i in range(len(att_maps)):
        np.fill_diagonal(att_maps[i], 0)
        
    all_attention_values = att_maps.reshape(-1)
    
    # dinuc
    alphabet = np.array([b'A', b'C', b'G', b'T'])

    seqs = np.where(x_test == 1)[2].reshape(x_test.shape[:2])
    seqs = alphabet[seqs]

    N = 20000
    sample = 2500

    alphabet = ['A', 'C', 'G', 'T']
    shuffled_seqs = []
    for i in range(len(seqs)):
        seq = seqs[i].tobytes()
        for j in range(N // len(seqs)):
            shuffled = ushuffle.shuffle(seq, 2).decode('UTF-8')
            newseq = np.zeros((len(shuffled), 4))
            ones = [alphabet.index(shuffled[k]) for k in range(len(shuffled))]
            pos = np.arange(len(shuffled))
            newseq[pos, ones] = 1
            shuffled_seqs.append(newseq)
    shuffled_seqs = np.array(shuffled_seqs)
    np.random.shuffle(shuffled_seqs)
    shuffled_seqs = shuffled_seqs[:sample]

    # Get feature maps
    feature_maps = utils.get_layer_output(model, c_index, shuffled_seqs)
    num_filters = feature_maps.shape[2]
    print('obtained feature maps')
    # Get key and queries
    q, k = utils.get_queries_keys(model, mha_index, utils.get_layer_output(model, mha_index-1, shuffled_seqs))
    print('computed keys and queries')
    # Compute attention map
    att_maps = utils.get_attention_maps(q, k, concat=tf.math.reduce_max).numpy()
    for i in range(len(att_maps)):
        np.fill_diagonal(att_maps[i], 0)
    print('generated attention maps')
    # Flatten
    dinuc_attention_values = np.reshape(att_maps, -1)
    dinuc_attention_values.sort()
    
    stats = np.array([all_attention_values, dinuc_attention_values])
    
    np.save(stats_dir, stats)
    
    







































