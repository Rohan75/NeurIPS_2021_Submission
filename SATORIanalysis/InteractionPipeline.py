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
    maps_dir = os.path.abspath(f'{baseline}/maps/{category}/model-{variant}')
    ppms_dir = os.path.abspath(f'{baseline}/ppms/{category}/model-{variant}')

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
    if not os.path.exists(maps_dir):
        os.makedirs(maps_dir)
    if not os.path.exists(ppms_dir):
        os.makedirs(ppms_dir)
    
    model_dir += f'/trial-{trial}/weights'
    motif_dir += f'/trial-{trial}.txt'
    tomtom_dir += f'/trial-{trial}'
    stats_dir += f'/trial-{trial}.npy'
    logs_dir += f'/trial-{trial}.pickle'
    maps_dir += f'/trial-{trial}.pdf'
    ppms_dir += f'/trial-{trial}.pdf'
    
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
    
    # filter interpretability -----------------------------------------------------------------------------------------
    
    # get ppms
    ppms = utils.get_ppms(model, x_test)
    moana.meme_generate(ppms, output_file=motif_dir) # save filter PPMs
    print('generated PPMs')
    
    # filter interpretablity
    utils.tomtom(motif_dir, tomtom_dir) # save tomtom files

    match_fraction, match_any, filter_match, filter_qvalue, min_qvalue, num_counts, coverage = utils.get_tomtom_stats(tomtom_dir + '/tomtom.tsv', 32)
    filter_matches = np.array(filter_match)
    tomtom_tpr = match_fraction
    tomtom_fpr = match_any - match_fraction
    tomtom_cov = coverage
    qvals = filter_qvalue
    
    
    print('TomTom TPR: ', tomtom_tpr)
    print('TomTom FPR: ', tomtom_fpr)
    print('Filter percent coverage: ', tomtom_cov)
    print('completed TomTom analysis')
    
    # hierachical clustering
    
    feature_maps = utils.get_layer_output(model, c_index, x_test[:5000])

    order = np.argsort((filter_matches == '').astype(int))
    fmaps = feature_maps.transpose()[order].transpose()
    filter_labels = filter_matches[order]
    junk_cutoff = np.where((filter_labels == '').astype(int) == 0)[0]
    if len(junk_cutoff) > 0:
        junk_cutoff = junk_cutoff[-1]+1
    else:
        junk_cutoff = len(filter_labels)

    c_fmaps = fmaps.transpose()[:junk_cutoff].transpose()
    c_filter_labels = filter_labels[:junk_cutoff]
    labels, labels_order, Z, groups, group_names, clustered_fmaps = utils.hierarchichal_clustering(c_fmaps, 0.7, c_filter_labels)

    fmaps = np.vstack([c_fmaps.transpose()[labels_order], fmaps.transpose()[junk_cutoff:]]).transpose()
    filter_labels_ordered = np.concatenate([c_filter_labels[labels_order], filter_labels[junk_cutoff:]])

    reorder = np.argsort(filter_labels_ordered[:junk_cutoff])
    fmaps = np.vstack([fmaps.transpose()[reorder], fmaps.transpose()[junk_cutoff:]]).transpose()
    filter_labels_ordered = np.concatenate([filter_labels_ordered[reorder], filter_labels_ordered[junk_cutoff:]])

    ppm_size = 25
    filter_ppms = []
    for i in range(len(ppms)):
        padded = np.vstack([ppms[i], np.zeros((ppm_size-len(ppms[i]), 4))+0.25])
        filter_ppms.append(padded)
    filter_ppms = np.array(filter_ppms)

    filter_ppms = np.concatenate([filter_ppms[order][:junk_cutoff][labels_order][reorder], filter_ppms[order][junk_cutoff:]])

    information = []
    for i in range(len(filter_ppms)):
        I = np.log2(4) + np.sum(filter_ppms[i] * np.log2(filter_ppms[i] + 1e-7), axis=1)
        information.append(I)
    information = np.sum(information, axis=1)

    empty_filters = np.where(information < 1)[0]
    filter_labels_ordered[empty_filters] = 'empty'
    
    print('completed clustering')
    
    # dinuc
    alphabet = np.array([b'A', b'C', b'G', b'T'])

    seqs = np.where(x_test == 1)[2].reshape(x_test.shape[:2])
    seqs = alphabet[seqs]

    N = 20000
    sample = 2000

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
    print('generated attention maps')
    # Flatten
    dinuc_attention_values = np.reshape(att_maps, -1)
    dinuc_attention_values.sort()

    significance = np.percentile(dinuc_attention_values, 95)
    print('finished dinuc shuffle')
    
    # satori
    
    sample = x_test[:2000]

    feature_maps = fmaps[:len(sample)]
    num_filters = feature_maps.shape[2]
    print('predicted feature maps')

    filter_activations = utils.get_filter_activations(model, c_index, sample)
    filter_activations = tf.math.reduce_max(filter_activations, axis=-1)/2
    filter_activations = filter_activations.numpy()
    filter_activations = np.concatenate([filter_activations[order][:junk_cutoff][labels_order], filter_activations[order][junk_cutoff:]])
    print('computed filter activations')

    mha_input = utils.get_layer_output(model, mha_index-1, sample)
    q, k = utils.get_queries_keys(model, mha_index, mha_input)
    print('obtained keys and queries')

    att_maps = utils.get_attention_maps(q, k, concat=tf.math.reduce_max).numpy()
    for i in range(len(att_maps)):
        np.fill_diagonal(att_maps[i], 0)

    thresh = significance
    position_interactions = utils.get_position_interactions(att_maps, thresh, limit=100000)
    print('found position interactions')

    filter_interactions = utils.get_filter_interactions(feature_maps, position_interactions, filter_activations)
    print('converted to filter interactions\n')

    motif_interactions = utils.get_motif_interactions(filter_interactions, filter_labels_ordered)

    ppv, fpr, tpr = utils.get_interaction_stats(motif_interactions, expecteds)
    junk_hit_frac = len(np.unique(np.where(motif_interactions == '')[0]))/len(motif_interactions)
    empty_hit_frac = len(np.unique(np.where(motif_interactions == 'empty')[0]))/len(motif_interactions)
    true_hit_frac = utils.get_interaction_stats(motif_interactions, expecteds)[0]
    rando_hit_frac = 1 - true_hit_frac - empty_hit_frac

    print(f'PPV {ppv} | TPR {tpr} | FPR {fpr}')
    print(f'empty {empty_hit_frac} | rando {rando_hit_frac} | junk {junk_hit_frac}')

    filter_map = (np.arange(num_filters ** 2) * 0).reshape((num_filters, num_filters))
    for j in range(len(filter_interactions)):
        filter_map[filter_interactions[j][0], filter_interactions[j][1]] += 1
    filter_map = np.array(filter_map)
    filter_map = np.amax([filter_map, filter_map.transpose()], axis=0)
    
    ind = np.array(np.tril_indices(len(filter_map), k=-1)).transpose()
    motif_interactions = filter_labels_ordered[ind]
    filter_interactions_values = filter_map[ind.transpose().tolist()]
    mask = ~(motif_interactions.transpose() == motif_interactions.transpose()[::-1])[0]
    motif_interactions = motif_interactions[mask]
    filter_interactions_values = filter_interactions_values[mask]
    motif_interactions.sort()

    matches = []
    for i in range(len(expecteds)):
        match = (expecteds[i] == motif_interactions).astype(int).transpose()
        match = match[0] * match[1]
        matches.append(match)
    TPs = np.amax(matches, axis=0).astype(bool)

    trues = filter_interactions_values[TPs]
    falses = filter_interactions_values[~TPs]

    k = 100
    signal = np.mean(trues)
    noise = np.sort(falses)[::-1][:k]
    noise = noise[np.where(noise > 0)]
    noise = np.mean(noise)
    satori_snr = signal/(noise + np.finfo(float).eps)

    y_pred = np.hstack([trues, falses])
    y_true = np.hstack([np.ones(trues.shape), np.zeros(falses.shape)])
    precision, recall, ts = sklearn.metrics.precision_recall_curve(y_true, y_pred)
    satori_aupr = sklearn.metrics.auc(recall, precision)
    specificity, sensitivity, ts = sklearn.metrics.roc_curve(y_true, y_pred)
    satori_auroc = sklearn.metrics.auc(specificity, sensitivity)

    print(f'satori AUPR {satori_aupr} | AUROC {satori_auroc} | SNR {satori_snr}')
    
    print('finished interpretability')
    
    # save all statistics
    
    stats = [ppv, fpr, tpr, satori_aupr, satori_auroc, satori_snr, junk_hit_frac, empty_hit_frac, rando_hit_frac, filter_map, tomtom_tpr, tomtom_fpr, tomtom_cov, qvals, auc_roc, auc_pr, significance, dinuc_attention_values]
    
    np.save(stats_dir, stats) # save stats
    print('saved statistics')
    
    filter_labels_ordered = [f'{i}-junk' if filter_labels_ordered[i] == '' else f'i-{filter_labels_ordered[i]}' for i in range(len(filter_labels_ordered))]
    
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.subplots()
    plt.xticks(list(range(len(filter_labels_ordered))), labels=filter_labels_ordered, rotation=90)
    plt.yticks(list(range(len(filter_labels_ordered))), labels=filter_labels_ordered)
    c = plt.imshow(filter_map, cmap='bwr_r', vmin=-np.max(filter_map))
    plt.colorbar(c, fraction=0.046, pad=0.04)
    plt.savefig(maps_dir, format='pdf', dpi=200)
    
    ppm_size = 25
    filter_ppms = []
    for i in range(len(ppms)):
        padded = np.vstack([ppms[i], np.zeros((ppm_size-len(ppms[i]), 4))+0.25])
        filter_ppms.append(padded)
    ppms = np.array(filter_ppms)
    fig = plt.figure(figsize=(25,8))
    plot_filters(ppms, fig, num_cols=8, names=filter_matches, fontsize=14)
    fig.savefig(ppms_dir, format='pdf', dpi=200, bbox_inches='tight')







































