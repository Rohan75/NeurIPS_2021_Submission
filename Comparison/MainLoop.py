import InteractionPipeline as pipeline
import models
import utils
import os
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("-b", type=str, default='baseline', help="baseline")
parser.add_argument("-c", type=str, default='category', help="category")
parser.add_argument("-v", type=str, default='variant', help="variant")
parser.add_argument("-a", type=str, default='relu', help="activation")
parser.add_argument("-bn", type=int, default=1, help="batch norm")
parser.add_argument("-p", type=int, default=4, help="pool size")
parser.add_argument("-ln", type=int, default=0, help="layer norm")
parser.add_argument("-nh", type=int, default=8, help="heads")
parser.add_argument("-k", type=int, default=32, help="vector size")
parser.add_argument("-lna", type=int, default=0, help="layer norm after")
parser.add_argument("-d", type=int, default=512, help="dense units")
parser.add_argument("-t", type=int, default=1, help="trials")
parser.add_argument("-st", type=int, default=1, help="start trial")
args = parser.parse_args()

baseline = args.b
category = args.c
variant = args.v
activation = args.a
batch_norm = bool(args.bn)
pool_size = args.p
layer_norm = bool(args.ln)
heads = args.nh
vector_size = args.k
layer_norm_after = bool(args.lna)
dense_units = args.d
trials = args.t
start_trial = args.st

trials = np.arange(start_trial, start_trial+trials)



file_path = 'synthetic_interactions_dataset.h5'
pipeline.load_dataset(file_path)



file_path = 'GIA_sequences.h5'
pipeline.load_gia_sequences(file_path)




names = ['ELF1', 'SIX3', 'ESR1', 'FOXN', 'CEBPB', 'YY1', 'GATA1', 'IRF1', 'SP1', 'NFIB', 'TEAD', 'TAL1']
jaspar_ids = [utils.elf, utils.six, utils.esr, utils.foxn, utils.cebpb, utils.yy1, utils.gata, utils.irf, utils.sp1, utils.nfib, utils.tead, utils.tal]
expecteds = [['ELF1', 'SIX3'], ['ESR1', 'FOXN'], ['GATA1', 'TAL1'], ['IRF1', 'SP1']]

motifs = [jaspar_ids, names, expecteds]



for t in trials:
    print(baseline, category, variant, t)
    
    model = models.CNN_ATT(
        in_shape = (200, 4),
        num_filters = 32,
        batch_norm = batch_norm,
        activation = activation,
        pool_size = pool_size,
        layer_norm = layer_norm,
        heads = heads,
        vector_size = vector_size,
        layer_norm_after = layer_norm_after,
        dense_units = dense_units,
        num_out = 12
    )
    
    model.summary()
    
    pipeline.run_pipeline(model, baseline, category, variant, t, motifs, batch_size=200, epochs=100)

            
            












