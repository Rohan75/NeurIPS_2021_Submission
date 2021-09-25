import InteractionPipeline as pipeline
import models
import utils
import os
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("-m", type=str, default='test', help="model name")
parser.add_argument("-a", type=str, default='relu', help="activation")
parser.add_argument("-bn", type=int, default=1, help="batch norm")
parser.add_argument("-p", type=int, default=4, help="pool size")
parser.add_argument("-ln", type=int, default=0, help="layer norm")
parser.add_argument("-lna", type=int, default=0, help="layer norm after")
parser.add_argument("-t", type=int, default=1, help="trials")
parser.add_argument("-st", type=int, default=1, help="start trial")
args = parser.parse_args()

model_name = args.m
activation = args.a
batch_norm = bool(args.bn)
pool_size = args.p
layer_norm = bool(args.ln)
layer_norm_after = bool(args.lna)
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
    print(model_name, t)
    
    model = models.CNN_ATT(
        in_shape = (200, 4),
        num_filters = 32,
        batch_norm = batch_norm,
        activation = activation,
        pool_size = pool_size,
        layer_norm = layer_norm,
        heads = 8,
        vector_size = 32,
        layer_norm_after = layer_norm_after,
        dense_units = 512,
        num_out = 12
    )
    
    model.summary()
    
    pipeline.run_pipeline(model, 'DATA2', model_name, 'model', t, motifs, batch_size=200, epochs=100)

            
            












