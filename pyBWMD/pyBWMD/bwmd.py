import functools
import numpy as np
from scipy.sparse import csr_matrix, vstack
from multiprocessing import Pool, cpu_count
import os
import scipy

# Assuming suffixArray is a Cython module optimized for performance
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import suffixArray

def is_file(s):
    return isinstance(s, str) and os.path.isfile(s)

def make_matrix(vecs):
    if np.sum([isinstance(v, np.ndarray) for v in vecs]) > len(vecs) / 2:
        # Use numpy's efficient operations for handling dense matrices
        vecs = [v if isinstance(v, np.ndarray) else v.toarray() for v in vecs]
        return np.vstack(vecs)
    else:
        # Efficiently stack sparse matrices without converting to dense
        return vstack(vecs, format='csr')

def vectorize(b, alphabet_size=26, processes=8):
    if isinstance(b, list):
        if processes < 0:
            processes = cpu_count()
        with Pool(processes) as pool:
            to_ret = pool.map(functools.partial(vectorize, alphabet_size=alphabet_size), b)
        return make_matrix(to_ret)
    
    if is_file(b):
        with open(b, "rb") as in_file:
            b = in_file.read()
    elif isinstance(b, str):
        b = str.encode(b)
    elif not isinstance(b, bytes):
        raise ValueError('Input was not a byte array, or could not be converted to one.')
    
    vec = suffixArray.bytes_to_raw_vec(b, alphabet_size)
    vec = np.sqrt(vec) / np.sqrt(2.0)
    
    if np.sum(vec != 0) * 2 < vec.shape[0]:
        return csr_matrix(vec)
    return vec
