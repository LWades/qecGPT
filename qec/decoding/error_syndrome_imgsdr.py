from args import args
import torch
import sys
from os.path import abspath, dirname

sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import Errormodel, mod2, Loading_code, read_code, log
from tqdm import tqdm
import numpy as np
import h5py
import gc

s = [0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.]


def ZX2image_sur(d, zxsyndrome):
    image_syndrome = np.zeros((2 * d - 1, 2 * d - 1))
    m = 2 * d ** 2 - 2 * d
    side = 2 * d - 1

    for i in range(m):
        # log("i = {}".format(i))
        a = i % side
        b = i // side
        if a < d - 1:
            # log(f"({b*2}, {a * 2 + 1})")
            image_syndrome[b * 2, a * 2 + 1] = 1 if int(zxsyndrome[i]) == 0 else -1
            # image_syndrome[b * 2, (a % 2) * 2 + 1] = 1 if int(zxsyndrome[i]) == 0 else -1
        else:
            # log(f"({b * 2 + 1}, {(a - d + 1) * 2})")
            image_syndrome[b * 2 + 1, (a - d + 1) * 2] = 1 if int(zxsyndrome[i]) == 0 else -1
    return image_syndrome


def error2image(d, error):
    image_error = np.zeros((2 * d - 1, 2 * d - 1))
    side = 2 * d - 1
    for i in range(len(error)):
        a, b = i // side, i % side
        if b < d:
            x = 2 * a
            y = b * 2
        else:
            x = 2 * a + 1
            y = 2 * (b - d) + 1
        image_error[x, y] = error[i]
    return image_error

# error = [1, 0, 1, 2, 0, 2, 0, 0, 0, 0, 3, 0, 0]
# error = [2, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 2, 3, 0, 0,
#          0, 0, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3]
# log(error2image(args.d, error))
# exit(0)


device, dtype = 'cpu', torch.float64
trnsz = args.trnsz
# c_type='tor'
c_type = args.c_type
# c_type = 'sur'
# d, k, seed = 3, 2, 0
d, k, seed = args.d, args.k, args.seed
# d, k, seed = 3, 1, 0
code_seed = 0
eval_seed = args.eval_seed

info = read_code(d, k, code_seed, c_type=c_type)
code = Loading_code(info)
mod2 = mod2(device=device, dtype=dtype)

if args.single_p > 0:
    ps = torch.tensor([args.single_p])
else:
    ps = torch.linspace(args.low_p, args.high_p, args.num_p)
# ps = [0.09]
log("error rate list: {}".format(ps))
log("Code: {}, d: {}, seed: {}, error model: {}, trnsz: {}".format(c_type, d, code_seed, 'depolarized', trnsz))

batch_size = 10000

def gen_img_batch():
    log("gen_img_batch")
    import h5py
    batch_nums = trnsz // batch_size
    log("batch_size: {}".format(batch_size))
    log("Error model seed: {}".format(seed))
    for p in ps:
        log("Error rate now: {}".format(format(p, '.2f')))
        if args.gtp == 'gibe':
            file_name = "/root/Surface_code_and_Toric_code/{}_pe/{}_d{}_p{}_trnsz{}_imger_eval_seed{}".format(c_type, c_type, d, format(p, '.3f'), trnsz, eval_seed)
        else:
            file_name = "/root/Surface_code_and_Toric_code/{}_pe/{}_d{}_p{}_trnsz{}_imger_seed{}".format(c_type, c_type, d, format(p, '.3f'), trnsz, seed)
        with h5py.File(file_name + ".hdf5", 'w') as f:
            log("Open h5py file success")
            dataset_error = f.create_dataset('image_errors', shape=(0, 2*d-1, 2*d-1), maxshape=(None, 2*d-1, 2*d-1), chunks=True, compression="gzip")
            log("Dataset create success")
            for i in tqdm(range(batch_nums)):
                E = Errormodel(e_rate=p, e_model='depolarized')
                log("Errors generating start...")
                errors = E.generate_error(n=code.n, m=batch_size, seed=seed)     # 这里改成了 batch_size
                log("Errors generating end.")
                image_errors = np.empty((batch_size, 2 * d - 1, 2 * d - 1))
                for j in range(batch_size):
                    image_errors[j] = error2image(d, errors[j])

                log("Writing image syndromes to h5py file start...")
                dataset_error.resize(dataset_error.shape[0] + batch_size, axis=0)
                dataset_error[-batch_size:] = image_errors
                log("Writing image syndromes to h5py file end.")

        log("Error rate {} success!".format(p))


gen_img_batch()

# nohup python error_syndrome_imgsdr.py --c_type 'sur' --d 11 --k 1 --trnsz 10000000 --seed 0 --gtp gb --single_p 0.100 > logs/error_syndrome_imgsdr_sur_d11.log &
# nohup python error_syndrome_imgsdr.py --c_type 'sur' --gtp gibe --d 11 --k 1 --trnsz 10000 --eval_seed 1 --gtp gb --single_p 0.100 > logs/error_syndrome_imgsdr_eval_sur_d11.log &
