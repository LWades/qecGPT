import h5py
from args import args
from module import log

with h5py.File(args.file, 'r') as f:
    log(f['args.key'][()].shape[0])