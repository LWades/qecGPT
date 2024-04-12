import random

from tqdm import tqdm

from args import args
import torch
import sys
from os.path import abspath, dirname

sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import Errormodel, mod2, Loading_code, read_code, log
from pymatching import Matching
import h5py
import numpy as np
import random

pwd_trndt = '/root/Surface_code_and_Toric_code/sur_pe/'
d, p = args.d, args.p
n = 10000
l = 2 * d - 1
d, k, seed, c_type, trnsz, device, dtype = args.d, args.k, args.seed, args.c_type, args.trnsz, 'cpu', torch.float64
info = read_code(d, k, seed, c_type=c_type)
code = Loading_code(info)
mod2 = mod2(device=device, dtype=dtype)

filename_test_data = pwd_trndt + '{}_d{}_p{}_trnsz{}_imgsdr_eval_seed{}.hdf5'.format(args.c_type, args.d,
                                                                                     format(p, '.3f'), trnsz,
                                                                                     args.eval_seed)
s = ""
for s_type in args.sym:
    s = s + "_" + s_type
filename_test_data_sym = pwd_trndt + '{}_d{}_p{}_trnsz{}_imgsdr_eval_seed{}_sym_{}.hdf5'.format(args.c_type, args.d,
                                                                                             format(p, '.3f'), n,
                                                                                             args.eval_seed, s)
filename_test_data_base = pwd_trndt + '{}_d{}_p{}_trnsz{}_imgsdr_eval_seed{}_base_{}.hdf5'.format(args.c_type, args.d,
                                                                                             format(p, '.3f'), n,
                                                                                             args.eval_seed, s)
log("test_data: {}".format(filename_test_data))
log("test_data_sym: {}".format(filename_test_data_sym))

random.seed(78)


def get_center(xs, ys):
    return int(np.mean(xs)), int(np.mean(ys))


def xsys2imgsdr(xs, ys):
    image_syndrome = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            if i % 2 == 0 and j % 2 == 1:
                image_syndrome[i, j] = 1
            if i % 2 == 1 and j % 2 == 0:
                image_syndrome[i, j] = 1
    image_syndrome[xs, ys] = -1
    return image_syndrome


def imgsdr2ZX(image_syndrome):
    syndrome = np.zeros(2 * d ** 2 - 2 * d)
    k = 0
    for i in range(l):
        for j in range(l):
            if i % 2 == 0 and j % 2 == 1:
                syndrome[k] = 1 if image_syndrome[i, j] == -1 else 0
                k += 1
            if i % 2 == 1 and j % 2 == 0:
                syndrome[k] = 1 if image_syndrome[i, j] == -1 else 0
                k += 1
    return syndrome


def check(a):
    return 0 <= a < l


def translation_imgsdr(image_syndrome):
    xs = np.where(image_syndrome == -1)[0]
    ys = np.where(image_syndrome == -1)[1]
    # log("xs: {}, ys: {}".format(xs, ys))
    top_x, bottom_x, left_y, right_y = min(xs), max(xs), min(ys), max(ys)
    # log("top_x: {}, bottom_x: {}, left_y: {}, right_y: {}".format(top_x, bottom_x, left_y, right_y))
    reverse_x, reverse_y = 1, 1
    if random.choice([0, 1]) == 0:
        # log("top")
        offset_x = random.randint(0, top_x)
        reverse_x = -1
    else:
        # log("bottom")
        offset_x = random.randint(0, l - 1 - bottom_x)
    if random.choice([0, 1]) == 0:
        # log("left")
        offset_y = random.randint(0, left_y)
        reverse_y = -1
    else:
        # log("right")
        offset_y = random.randint(0, l - 1 - right_y)
    if (offset_x + offset_y) % 2 != 0:
        if offset_x > offset_y:
            offset_x -= 1
        else:
            offset_y -= 1
    offset_x *= reverse_x
    offset_y *= reverse_y
    # log("offset_x: {}, offset_y: {}".format(offset_x, offset_y))
    new_xs, new_ys = xs + offset_x, ys + offset_y
    new_imgsdr = xsys2imgsdr(new_xs, new_ys)
    return new_imgsdr, True


def reflection_imgsdr(image_syndrome):
    xs = np.where(image_syndrome == -1)[0]
    ys = np.where(image_syndrome == -1)[1]
    n = len(xs)
    center = get_center(xs, ys)
    r = random.randint(0, 3)
    # log("r: {}".format(r))
    new_xs, new_ys = np.zeros_like(xs), np.zeros_like(ys)
    if r == 0:
        for i in range(n):
            new_xs[i] = xs[i]
            new_ys[i] = (2 * center[1] - ys[i]) % (2 * d)
            if (not check(new_xs[i])) or (not check(new_ys[i])):
                return np.empty(1), False
    elif r == 1:
        for i in range(n):
            new_xs[i] = (2 * center[1] - xs[i]) % (2 * d)
            new_ys[i] = ys[i]
            if (not check(new_xs[i])) or (not check(new_ys[i])):
                return np.empty(1), False
    elif r == 2:
        for i in range(n):
            t = center[0] - center[1]
            new_xs[i], new_ys[i] = ys[i] + t, xs[i] - t
            if (not check(new_xs[i])) or (not check(new_ys[i])):
                return np.empty(1), False
    elif r == 3:
        for i in range(n):
            t = center[0] + center[1]
            new_xs[i], new_ys[i] = t - ys[i], t - xs[i]
            if (not check(new_xs[i])) or (not check(new_ys[i])):
                return np.empty(1), False
    new_imgsdr = xsys2imgsdr(new_xs, new_ys)
    return new_imgsdr, True


def rotation_imgsdr(image_syndrome):
    xs = np.where(image_syndrome == -1)[0]
    ys = np.where(image_syndrome == -1)[1]
    n = len(xs)
    center = get_center(xs, ys)
    r = random.randint(0, 2)
    # log("r: {}".format(r))
    new_xs, new_ys = np.zeros_like(xs), np.zeros_like(ys)
    if r == 0:
        for i in range(n):
            new_xs[i] = ys[i] + center[0] - center[1]
            new_ys[i] = center[0] + center[1] - xs[i]
            if (not check(new_xs[i])) or (not check(new_ys[i])):
                return np.empty(1), False
    elif r == 1:
        for i in range(n):
            new_xs[i] = 2 * center[0] - xs[i]
            new_ys[i] = 2 * center[1] - ys[i]
            if (not check(new_xs[i])) or (not check(new_ys[i])):
                return np.empty(1), False
    elif r == 2:
        for i in range(n):
            new_xs[i] = center[0] + center[1] - ys[i]
            new_ys[i] = center[1] - center[0] + xs[i]
            if (not check(new_xs[i])) or (not check(new_ys[i])):
                return np.empty(1), False
    new_imgsdr = xsys2imgsdr(new_xs, new_ys)
    return new_imgsdr, True


test_s = np.array([[0, 1, 0, 1, 0],
                  [1, 0, -1, 0, 1],
                  [0, -1, 0, 1, 0],
                  [1, 0, -1, 0, 1],
                  [0, 1, 0, -1, 0]])
# test_s = np.array([[0, 1, 0, -1, 0],
#                   [1, 0, -1, 0, 1],
#                   [0, 1, 0, -1, 0],
#                   [1, 0, 1, 0, -1],
#                   [0, 1, 0, 1, 0]])
# a, b = rotation_imgsdr(test_s)
# a, b = reflection_imgsdr(test_s)
# a, b = translation_imgsdr(test_s)
# log("a: \n{}".format(a))
# log("b:\n{}".format(b))
# exit(0)

log("2 - Init MWPM...")
log("2.1 - Init weights...")
p_tensor = torch.tensor(p)
weights = torch.ones(2 * code.n) * torch.log((1 - p_tensor) / p_tensor)
log("2.1 - Init weights... Done.")
log("2.2 - Init MWPM Decoder...")
decoder_mwpm = Matching(code.PCM, weights=weights)
log("2.2 - Init MWPM Decoder... Done.")

with h5py.File(filename_test_data_sym, 'w') as f_sym, h5py.File(filename_test_data_base, 'w') as f_base:
    dataset_syndr = f_sym.create_dataset('image_syndromes', shape=(0, 2 * d - 1, 2 * d - 1),
                                         maxshape=(None, 2 * d - 1, 2 * d - 1), chunks=True, compression="gzip")
    dataset_le = f_sym.create_dataset('logical_errors', shape=(0, 1), maxshape=(None, 1), chunks=True,
                                      compression="gzip")
    dataset_syndr_base = f_base.create_dataset('image_syndromes', shape=(0, 2 * d - 1, 2 * d - 1),
                                         maxshape=(None, 2 * d - 1, 2 * d - 1), chunks=True, compression="gzip")
    dataset_le_base = f_base.create_dataset('logical_errors', shape=(0, 1), maxshape=(None, 1), chunks=True,
                                      compression="gzip")
    image_syndrome_bases = np.empty((n, 2 * d - 1, 2 * d - 1))
    image_syndrome_syms = np.empty((n, 2 * d - 1, 2 * d - 1))
    logical_error_bases = np.empty((n, 1))
    logical_error_syms = np.empty((n, 1))
    c = 0
    with h5py.File(filename_test_data, 'r') as f:
        test_syndrome = f['image_syndromes'][()]
        test_logical_error = f['logical_errors'][()]
        syms = [translation_imgsdr, reflection_imgsdr, rotation_imgsdr]
        E = Errormodel(e_rate=p, e_model='depolarized')
        for i in tqdm(range(test_syndrome.shape[0])):
            if c >= n:
                break
            # test_syndrome[i] = test_s
            if args.sym == 'all':
                r = random.randint(0, 2)
                image_syndrome_sym, success = syms[r](test_syndrome[i])
            elif args.sym == 'tl':
                image_syndrome_sym, success = translation_imgsdr(test_syndrome[i])
            elif args.sym == 'rf':
                image_syndrome_sym, success = reflection_imgsdr(test_syndrome[i])
            elif args.sym == 'rt':
                image_syndrome_sym, success = rotation_imgsdr(test_syndrome[i])

            if success:
                image_syndrome_bases[c] = test_syndrome[i]
                logical_error_bases[c] = test_logical_error[i]
                image_syndrome_syms[c] = image_syndrome_sym

                syndrome = imgsdr2ZX(image_syndrome_sym)
                error = decoder_mwpm.decode(syndrome)
                error_zero = np.zeros_like(error)
                errors = torch.vstack((torch.from_numpy(error), torch.from_numpy(error_zero)))
                errors_xyz = mod2.xyz(errors)
                syndrome_zero = np.zeros_like(syndrome)
                syndromes = torch.vstack((torch.from_numpy(syndrome), torch.from_numpy(syndrome_zero)))

                pes = E.pure(code.pure_es, syndromes, device=device, dtype=dtype)
                checks = mod2.opt_prod(pes, errors_xyz)
                logical_errors = mod2.commute(checks, code.logical_opt)
                logical_errors_1d = logical_errors[:, 0] * 2 + logical_errors[:, 1]
                logical_errors_1d = logical_errors_1d.unsqueeze(1)
                logical_error_syms[c] = logical_errors_1d[0]

                c += 1
        if c == n:
            log("done")
        else:
            log("data not enough")
            exit(1)
        dataset_syndr = image_syndrome_syms
        dataset_le = logical_error_syms
        dataset_syndr_base = image_syndrome_bases
        dataset_le_base = logical_error_bases
log("All finished")
# nohup python3 sym_construct.py --c_type sur --d 11 --k 1 --p 0.1 --sym all --seed 0 --trnsz 50000 --eval_seed 3 > logs/sym_construct_11.log
# nohup python3 sym_construct.py --c_type sur --d 11 --k 1 --p 0.1 --sym all --seed 0 --trnsz 10000 --eval_seed 3 > logs/sym_construct_15.log &
# nohup python3 sym_construct.py --c_type sur --d 11 --k 1 --p 0.1 --sym tl --seed 0 --trnsz 50000 --eval_seed 3 > logs/sym_construct_12.log
# nohup python3 sym_construct.py --c_type sur --d 11 --k 1 --p 0.1 --sym rt --seed 0 --trnsz 50000 --eval_seed 3 > logs/sym_construct_13.log
# nohup python3 sym_construct.py --c_type sur --d 11 --k 1 --p 0.1 --sym rf --seed 0 --trnsz 50000 --eval_seed 3 > logs/sym_construct_14.log

