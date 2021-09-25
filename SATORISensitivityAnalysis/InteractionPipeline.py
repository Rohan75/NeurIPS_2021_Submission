# imports -----------------------------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf

import tfomics
from tfomics import moana

import os, shutil
from six.moves import cPickle

import h5py

import utils
import models

import ushuffle

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
    
    model_dir += f'/trial-{trial}/weights'
    motif_dir += f'/trial-{trial}.txt'
    tomtom_dir += f'/trial-{trial}'
    stats_dir += f'/trial-{trial}.npy'
    logs_dir += f'/trial-{trial}.pickle'
    
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
    utils.tomtom(motif_dir, tomtom_dir, root=False) # save tomtom files
    names = ['ELF1', 'SIX3', 'ESR1', 'FOXN', 'CEBPB', 'YY1', 'GATA1', 'IRF1', 'SP1', 'NFIB', 'TEAD', 'TAL1']
    jaspar_ids = [utils.elf, utils.six, utils.esr, utils.foxn, utils.cebpb, utils.yy1, utils.gata, utils.irf, utils.sp1, utils.nfib, utils.tead, utils.tal]

    match_fraction, match_any, filter_match, filter_qvalue, min_qvalue, num_counts = utils.get_tomtom_stats(tomtom_dir + '/tomtom.tsv', 32)
    filter_matches = np.array(filter_match)
    tomtom_tpr = match_fraction
    tomtom_fpr = match_any - match_fraction
    tomtom_cov = np.sum([1 if i in filter_matches else 0 for i in names])/len(names)
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

    good_filters = np.where(information > 0.5)[0]

    corr_fmaps = fmaps.transpose()[good_filters].transpose()
    corr_filter_labels_ordered = filter_labels_ordered[good_filters]
    
    print('completed clustering')
    
    
    #satori
    
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
    print('generated attention maps')

    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    satori_ppvs = []
    satori_tprs = []
    filter_maps = []

    for i in range(len(thresholds)):

        thresh = thresholds[i]
        position_interactions = utils.get_position_interactions(att_maps, thresh, limit=50000)
        print('found position interactions')

        filter_interactions = utils.get_filter_interactions(feature_maps, position_interactions, filter_activations)
        print('converted to filter interactions')

        motif_interactions = utils.get_motif_interactions(filter_interactions, filter_labels_ordered)

        ppv, fpr, tpr = utils.get_interaction_stats(motif_interactions, expecteds)
        satori_ppvs.append(ppv)
        satori_tprs.append(tpr)

        filter_map = (np.arange(num_filters ** 2) * 0).reshape((num_filters, num_filters))
        for j in range(len(filter_interactions)):
            filter_map[filter_interactions[j][0], filter_interactions[j][1]] += 1
        filter_map = np.array(filter_map)
        filter_map = np.amax([filter_map, filter_map.transpose()], axis=0)

        filter_maps.append(filter_map)

    satori_ppvs = np.expand_dims(satori_ppvs, axis=1)
    satori_tprs = np.expand_dims(satori_tprs, axis=1)
    filter_maps = np.array(filter_maps)
    
    print('completed satori interpretability')
    
    # save all statistics
    
    stats = [satori_ppvs, satori_tprs, filter_maps, tomtom_tpr, tomtom_fpr, tomtom_cov, auc_roc, auc_pr]
    
    np.save(stats_dir, stats) # save stats
    print('saved statistics')







































