import os, io, h5py
import requests as rq
import numpy as np

url = 'https://www.dropbox.com/s/214w76b616qclqr/complex_synthetic_dataset.h5?raw=true'

data = rq.get(url)
data.raise_for_status()

with h5py.File(io.BytesIO(data.content), 'r') as dataset:
    X = np.array(dataset['X'])
    Y = np.array(dataset['Y'])
    L = np.array(dataset['L'])
    
    
    
file_name = 'synthetic_interactions_dataset.h5'
    
if os.path.exists(file_name):
    os.remove(file_name)
    
hf = h5py.File(file_name, 'w')
hf.create_dataset('X', data=X)
hf.create_dataset('Y', data=Y)
hf.create_dataset('L', data=L)
hf.close()






url = 'https://www.dropbox.com/s/g6zyq8kjp79umrt/GIA_sequences.h5?raw=true'

data = rq.get(url)
data.raise_for_status()

with h5py.File(io.BytesIO(data.content), 'r') as dataset:
    indep = np.array(dataset['independent'])
    inter = np.array(dataset['interactions'])
    
    

file_name = 'GIA_sequences.h5'
    
if os.path.exists(file_name):
    os.remove(file_name)
    
hf = h5py.File(file_name, 'w')
hf.create_dataset('independent', data=indep)
hf.create_dataset('interactions', data=inter)
hf.close()