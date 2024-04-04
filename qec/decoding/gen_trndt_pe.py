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


def trans_org_syndr_to_image_syndr(L, syndr):
    # log("org_syndr(L={}, shape={}):\n{}".format(L, syndr.shape, syndr))
    image_syndr = torch.zeros(2 * L - 1, 2 * L - 1, dtype=torch.int)
    t = 0
    for i in range(2 * L - 1):
        for j in range(2 * L - 1):
            if i % 2 == 0:
                if j % 2 == 1:  # 偶数行奇数列 Z 稳定子
                    if syndr[t] == int(0):
                        image_syndr[i, j] = 1
                    else:
                        image_syndr[i, j] = -1
                    t += 1
            else:
                if j % 2 == 0:  # 奇数行偶数列 X 稳定子
                    if syndr[t] == int(0):
                        image_syndr[i, j] = 1
                    else:
                        image_syndr[i, j] = -1
                    t += 1
    # log("image_syndr(L={}, shape={}):\n{}".format(L, image_syndr.shape, image_syndr))
    return image_syndr


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

# p = 0.1
# ps = torch.linspace(0.01, 0.10, 10)
# ps = torch.linspace(0.16, 0.20, 5)
# ps = torch.linspace(0.11, 0.15, 5)
if args.single_p > 0:
    ps = torch.tensor([args.single_p])
else:
    ps = torch.linspace(args.low_p, args.high_p, args.num_p)
# ps = [0.09]
log("error rate list: {}".format(ps))
log("Code: {}, d: {}, seed: {}, error model: {}, trnsz: {}".format(c_type, d, code_seed, 'depolarized', trnsz))

batch_size = 10000

# for p in ps:
#     # log("pure_es: (shape={})\n{}".format(code.pure_es.shape, code.pure_es))
#     log("Error rate now: {}".format(p))
#     E = Errormodel(e_rate=p, e_model='depolarized')
#     log("Errors generating start...")
#     errors = E.generate_error(n=code.n, m=trnsz, seed=seed)
#     log("Errors generating end.")
#     log("Syndromes generating start...")
#     syndromes = mod2.commute(errors, code.g_stabilizer)
#     log("Syndromes generating end.")
#     log("Pure errors generating start...")
#     pes = E.pure(code.pure_es, syndromes, device=device, dtype=dtype)  # 得到可以恢复错误症状翻转的纯错误
#     log("Pure errors generating end.")
#     # recovers = pes
#     log("Check generating start...")
#     check = mod2.opt_prod(pes, errors)  # 矩阵加法
#     log("Check generating end.")
#     log("Logical errors generating start...")
#     logical_errors = mod2.commute(check, code.logical_opt)   # 看看有无逻辑错误
#     log("Logical errors generating end.")
#
#     log("logical errors trans to 1d start...")
#     logical_errors_1d = logical_errors[:, 0] * 2 + logical_errors[:, 1]
#     logical_errors_1d = logical_errors_1d.unsqueeze(1)
#     log("logical errors trans to 1d end.")
#
#     log("Origin syndrome trans to image syndrome start...")
#     image_syndromes = torch.empty(trnsz, 2*d-1, 2*d-1)
#     for j in range(len(syndromes)):
#         image_syndromes[j] = trans_org_syndr_to_image_syndr(d, syndromes[j])
#     log("Origin syndrome trans to image syndrome end.")
#     # log("image_syndrome(shape={}):\n{}".format(image_syndromes.shape, image_syndrome))
#     # log("syndromes.type: {}".format(type(syndromes)))
#     # log("logical_errors.type: {}".format(type(logical_errors)))
#     # log("commute:\n{}".format(logical_errors))
#     log("Writing to npz file start...")
#     file_name = "../trndt/{}_d{}_p{}_trnsz{}_imgsdr".format(c_type, d, format(p, '.3f'), trnsz)
#     np.savez_compressed(file_name, image_syndromes=image_syndromes.cpu().numpy(),
#                         logical_errors=logical_errors_1d.cpu().numpy())
#     log("Writing to npz file end.")
# nohup python gen_trndt_pe.py > logs/gen_trndt_pe.log &
# nohup python gen_trndt_pe.py -c_type 'sur' --d 3 --trnsz 10000000 > logs/gen_trndt_pe_03.log &
# nohup python gen_trndt_pe.py -c_type 'sur' --d 5 --trnsz 10000000 > logs/gen_trndt_pe_05.log &
# nohup python gen_trndt_pe.py -c_type 'sur' --d 3 --trnsz 10000 > logs/gen_trndt_pe_04.log &
# python gen_trndt_pe.py -c_type 'sur' --d 5 --trnsz 10000000

def gen_batch_torc_eval():
    import h5py
    batch_nums = trnsz // batch_size
    log("batch_size: {}".format(batch_size))
    log("Error model seed: {}".format(eval_seed))
    for p in ps:
        log("Error rate now: {}".format(p))
        file_name = "/root/Surface_code_and_Toric_code/{}_pe/{}_d{}_p{}_trnsz{}_eval_seed{}".format(c_type, c_type, d, format(p, '.3f'), trnsz, eval_seed)
        log("file name: {}".format(file_name))
        with h5py.File(file_name + ".hdf5", 'w') as f:
            log("Open h5py file success")
            dataset_syndr = f.create_dataset('syndromes', shape=(0, 2*d**2-2), maxshape=(None, 2*d**2-2), chunks=True, compression="gzip")
            dataset_le = f.create_dataset('logical_errors', shape=(0, 4), maxshape=(None, 4), chunks=True, compression="gzip")
            log("Dataset create success")
            for i in tqdm(range(batch_nums)):
                # log("Num {} batch start..., {} in all".format(i, batch_nums))

                E = Errormodel(e_rate=p, e_model='depolarized')
                # log("Errors generating start...")
                errors = E.generate_error(n=code.n, m=batch_size, seed=eval_seed)     # 这里改成了 batch_size
                # log("Errors generating end.")

                # log("Syndromes generating start...")
                syndromes = mod2.commute(errors, code.g_stabilizer)
                # log("Syndromes generating end.")

                # log("Pure errors generating start...")
                pes = E.pure(code.pure_es, syndromes, device=device, dtype=dtype)  # 得到可以恢复错误症状翻转的纯错误
                # log("Pure errors generating end.")

                # recovers = pes
                # log("Check generating start...")
                check = mod2.opt_prod(pes, errors)  # 矩阵加法
                # log("Check generating end.")
                # log("Logical errors generating start...")
                logical_errors = mod2.commute(check, code.logical_opt)  # 看看有无逻辑错误
                # log("Logical errors generating end.")

                # log("Writing syndromes to h5py file start...")
                dataset_syndr.resize(dataset_syndr.shape[0] + batch_size, axis=0)
                dataset_syndr[-batch_size:] = syndromes
                # log("Writing syndromes to h5py file end.")

                # log("Writing logical errors to h5py file start...")
                dataset_le.resize(dataset_le.shape[0] + batch_size, axis=0)
                dataset_le[-batch_size:] = logical_errors.cpu().numpy()
                # log("Writing logical errors to h5py file end.")

                # log("Num {} batch end.".format(i))
        log("Error rate {} success!".format(p))

def gen_batch_torc():
    import h5py
    batch_nums = trnsz // batch_size
    # log("batch_size: {}".format(batch_size))
    for p in ps:
        log("Error rate now: {}".format(format(p, '.2f')))
        log("Error model seed: {}".format(seed))
        file_name = "/root/Surface_code_and_Toric_code/{}_pe/{}_d{}_p{}_trnsz{}_seed{}".format(c_type, c_type, d, format(p, '.3f'), trnsz, seed)
        with h5py.File(file_name + ".hdf5", 'w') as f:
            log("Open h5py file success")
            dataset_syndr = f.create_dataset('syndromes', shape=(0, 2*d**2-2), maxshape=(None, 2*d**2-2), chunks=True, compression="gzip")
            dataset_le = f.create_dataset('logical_errors', shape=(0, 4), maxshape=(None, 4), chunks=True, compression="gzip")
            log("Dataset create success")
            for i in tqdm(range(batch_nums)):
                # log("Num {} batch start..., {} in all".format(i, batch_nums))

                E = Errormodel(e_rate=p, e_model='depolarized')
                # log("Errors generating start...")
                errors = E.generate_error(n=code.n, m=batch_size, seed=seed)     # 这里改成了 batch_size
                # log("Errors generating end.")

                # log("Syndromes generating start...")
                syndromes = mod2.commute(errors, code.g_stabilizer)
                # log("Syndromes generating end.")

                # log("Pure errors generating start...")
                pes = E.pure(code.pure_es, syndromes, device=device, dtype=dtype)  # 得到可以恢复错误症状翻转的纯错误
                # log("Pure errors generating end.")

                # recovers = pes
                # log("Check generating start...")
                check = mod2.opt_prod(pes, errors)  # 矩阵加法
                # log("Check generating end.")
                # log("Logical errors generating start...")
                logical_errors = mod2.commute(check, code.logical_opt)  # 看看有无逻辑错误
                # log("Logical errors generating end.")

                # log("Writing syndromes to h5py file start...")
                dataset_syndr.resize(dataset_syndr.shape[0] + batch_size, axis=0)
                dataset_syndr[-batch_size:] = syndromes
                # log("Writing syndromes to h5py file end.")

                # log("Writing logical errors to h5py file start...")
                dataset_le.resize(dataset_le.shape[0] + batch_size, axis=0)
                dataset_le[-batch_size:] = logical_errors.cpu().numpy()
                # log("Writing logical errors to h5py file end.")

                # log("Num {} batch end.".format(i))
        log("Error rate {} success!".format(p))


def gen_img_batch():
    log("gen_img_batch")
    import h5py
    batch_nums = trnsz // batch_size
    log("batch_size: {}".format(batch_size))
    log("Error model seed: {}".format(seed))
    for p in ps:
        log("Error rate now: {}".format(format(p, '.2f')))
        file_name = "/root/Surface_code_and_Toric_code/{}_pe/{}_d{}_p{}_trnsz{}_imgsdr_seed{}".format(c_type, c_type, d, format(p, '.3f'), trnsz, seed)
        with h5py.File(file_name + ".hdf5", 'w') as f:
            log("Open h5py file success")
            dataset_syndr = f.create_dataset('image_syndromes', shape=(0, 2*d-1, 2*d-1), maxshape=(None, 2*d-1, 2*d-1), chunks=True, compression="gzip")
            dataset_le = f.create_dataset('logical_errors', shape=(0, 1), maxshape=(None, 1), chunks=True, compression="gzip")
            log("Dataset create success")
            for i in tqdm(range(batch_nums)):
                # log("Num {} batch start..., {} in all".format(i, batch_nums))

                E = Errormodel(e_rate=p, e_model='depolarized')
                # log("Errors generating start...")
                errors = E.generate_error(n=code.n, m=batch_size, seed=seed)     # 这里改成了 batch_size
                # log("Errors generating end.")

                # log("Syndromes generating start...")
                syndromes = mod2.commute(errors, code.g_stabilizer)
                # log("Syndromes generating end.")

                # log("Pure errors generating start...")
                pes = E.pure(code.pure_es, syndromes, device=device, dtype=dtype)  # 得到可以恢复错误症状翻转的纯错误
                # log("Pure errors generating end.")

                # recovers = pes
                # log("Check generating start...")
                check = mod2.opt_prod(pes, errors)  # 矩阵加法
                # log("Check generating end.")
                # log("Logical errors generating start...")
                logical_errors = mod2.commute(check, code.logical_opt)  # 看看有无逻辑错误
                # log("Logical errors generating end.")

                # log("logical errors trans to 1d start...")
                logical_errors_1d = logical_errors[:, 0] * 2 + logical_errors[:, 1]
                logical_errors_1d = logical_errors_1d.unsqueeze(1)
                # log("logical errors trans to 1d end.")

                # log("Origin syndrome trans to image syndrome start...")
                image_syndromes = np.empty((batch_size, 2 * d - 1, 2 * d - 1))
                for j in range(len(syndromes)):
                    image_syndromes[j] = ZX2image_sur(d, syndromes[j])
                # log("Origin syndrome trans to image syndrome end.")

                # log("Writing image syndromes to h5py file start...")
                dataset_syndr.resize(dataset_syndr.shape[0] + batch_size, axis=0)
                dataset_syndr[-batch_size:] = image_syndromes
                # log("Writing image syndromes to h5py file end.")

                # log("Writing logical errors to h5py file start...")
                dataset_le.resize(dataset_le.shape[0] + batch_size, axis=0)
                dataset_le[-batch_size:] = logical_errors_1d.cpu().numpy()
                # log("Writing logical errors to h5py file end.")

                # log("Num {} batch end.".format(i))
        log("Error rate {} success!".format(p))


def gen_img_batch_eval():
    log("gen_img_batch_eval")
    batch_nums = trnsz // batch_size
    log("batch_size: {}".format(batch_size))
    log("Error model seed: {}".format(eval_seed))
    for p in ps:
        log("Error rate now: {}".format(format(p, '.2f')))
        file_name = "/root/Surface_code_and_Toric_code/{}_pe/{}_d{}_p{}_trnsz{}_imgsdr_eval_seed{}".format(c_type, c_type, d, format(p, '.3f'), trnsz, eval_seed)
        with h5py.File(file_name + ".hdf5", 'w') as f:
            log("Open h5py file success")
            dataset_syndr = f.create_dataset('image_syndromes', shape=(0, 2*d-1, 2*d-1), maxshape=(None, 2*d-1, 2*d-1), chunks=True, compression="gzip")
            dataset_le = f.create_dataset('logical_errors', shape=(0, 1), maxshape=(None, 1), chunks=True, compression="gzip")
            log("Dataset create success")
            for i in tqdm(range(batch_nums)):
                log("Num {} batch start..., {} in all".format(i, batch_nums))

                E = Errormodel(e_rate=p, e_model='depolarized')
                log("Errors generating start...")
                errors = E.generate_error(n=code.n, m=batch_size, seed=eval_seed)     # 这里改成了 batch_size
                log("Errors generating end.")

                log("Syndromes generating start...")
                syndromes = mod2.commute(errors, code.g_stabilizer)
                log("Syndromes generating end.")

                log("Pure errors generating start...")
                pes = E.pure(code.pure_es, syndromes, device=device, dtype=dtype)  # 得到可以恢复错误症状翻转的纯错误
                log("Pure errors generating end.")

                # recovers = pes
                log("Check generating start...")
                check = mod2.opt_prod(pes, errors)  # 矩阵加法
                log("Check generating end.")
                log("Logical errors generating start...")
                logical_errors = mod2.commute(check, code.logical_opt)  # 看看有无逻辑错误
                log("Logical errors generating end.")

                log("logical errors trans to 1d start...")
                logical_errors_1d = logical_errors[:, 0] * 2 + logical_errors[:, 1]
                logical_errors_1d = logical_errors_1d.unsqueeze(1)
                log("logical errors trans to 1d end.")

                log("Origin syndrome trans to image syndrome start...")
                image_syndromes = np.empty((batch_size, 2 * d - 1, 2 * d - 1))
                for j in range(len(syndromes)):
                    image_syndromes[j] = ZX2image_sur(d, syndromes[j])
                log("Origin syndrome trans to image syndrome end.")

                log("Writing image syndromes to h5py file start...")
                dataset_syndr.resize(dataset_syndr.shape[0] + batch_size, axis=0)
                dataset_syndr[-batch_size:] = image_syndromes
                log("Writing image syndromes to h5py file end.")

                log("Writing logical errors to h5py file start...")
                dataset_le.resize(dataset_le.shape[0] + batch_size, axis=0)
                dataset_le[-batch_size:] = logical_errors_1d.cpu().numpy()
                log("Writing logical errors to h5py file end.")

                log("Num {} batch end.".format(i))
        log("Error rate {} success!".format(p))



def gen_batch():
    log("gen_img_batch")
    import h5py
    batch_nums = trnsz // batch_size
    log("batch_size: {}".format(batch_size))
    log("Error model seed: {}".format(seed))
    for p in ps:
        log("Error rate now: {}".format(format(p, '.2f')))
        file_name = "/root/Surface_code_and_Toric_code/{}_base_pe/{}_d{}_p{}_trnsz{}_seed{}".format(c_type, c_type, d, format(p, '.3f'), trnsz, seed)
        with h5py.File(file_name + ".hdf5", 'w') as f:
            log("Open h5py file success")
            dataset_syndr = f.create_dataset('syndromes', shape=(0, 2*d**2-2*d), maxshape=(None, 2*d**2-2*d), chunks=True, compression="gzip")
            dataset_le = f.create_dataset('logical_errors', shape=(0, 1), maxshape=(None, 1), chunks=True, compression="gzip")
            log("Dataset create success")
            for i in tqdm(range(batch_nums)):
                # log("Num {} batch start..., {} in all".format(i, batch_nums))

                E = Errormodel(e_rate=p, e_model='depolarized')
                # log("Errors generating start...")
                errors = E.generate_error(n=code.n, m=batch_size, seed=seed)     # 这里改成了 batch_size
                # log("Errors generating end.")

                # log("Syndromes generating start...")
                syndromes = mod2.commute(errors, code.g_stabilizer)
                # log("Syndromes generating end.")

                # log("Pure errors generating start...")
                pes = E.pure(code.pure_es, syndromes, device=device, dtype=dtype)  # 得到可以恢复错误症状翻转的纯错误
                # log("Pure errors generating end.")

                # recovers = pes
                # log("Check generating start...")
                check = mod2.opt_prod(pes, errors)  # 矩阵加法
                # log("Check generating end.")
                # log("Logical errors generating start...")
                logical_errors = mod2.commute(check, code.logical_opt)  # 看看有无逻辑错误
                # log("Logical errors generating end.")

                # log("logical errors trans to 1d start...")
                logical_errors_1d = logical_errors[:, 0] * 2 + logical_errors[:, 1]
                logical_errors_1d = logical_errors_1d.unsqueeze(1)
                # log("logical errors trans to 1d end.")

                # log("Writing syndromes to h5py file start...")
                dataset_syndr.resize(dataset_syndr.shape[0] + batch_size, axis=0)
                dataset_syndr[-batch_size:] = syndromes
                # log("Writing syndromes to h5py file end.")

                # log("Writing logical errors to h5py file start...")
                dataset_le.resize(dataset_le.shape[0] + batch_size, axis=0)
                dataset_le[-batch_size:] = logical_errors_1d.cpu().numpy()
                # log("Writing logical errors to h5py file end.")

                # log("Num {} batch end.".format(i))
        log("Error rate {} success!".format(p))

def gen_batch_eval():
    log("gen_img_batch")
    import h5py
    batch_nums = trnsz // batch_size
    log("batch_size: {}".format(batch_size))
    log("Error model seed: {}".format(eval_seed))
    for p in ps:
        log("Error rate now: {}".format(format(p, '.2f')))
        file_name = "/root/Surface_code_and_Toric_code/{}_base_pe/{}_d{}_p{}_trnsz{}_seed{}".format(c_type, c_type, d, format(p, '.3f'), trnsz, eval_seed)
        with h5py.File(file_name + ".hdf5", 'w') as f:
            log("Open h5py file success")
            dataset_syndr = f.create_dataset('syndromes', shape=(0, 2*d**2-2*d), maxshape=(None, 2*d**2-2*d), chunks=True, compression="gzip")
            dataset_le = f.create_dataset('logical_errors', shape=(0, 1), maxshape=(None, 1), chunks=True, compression="gzip")
            log("Dataset create success")
            for i in range(batch_nums):
                log("Num {} batch start..., {} in all".format(i, batch_nums))

                E = Errormodel(e_rate=p, e_model='depolarized')
                log("Errors generating start...")
                errors = E.generate_error(n=code.n, m=batch_size, seed=eval_seed)     # 这里改成了 batch_size
                log("Errors generating end.")

                log("Syndromes generating start...")
                syndromes = mod2.commute(errors, code.g_stabilizer)
                log("Syndromes generating end.")

                log("Pure errors generating start...")
                pes = E.pure(code.pure_es, syndromes, device=device, dtype=dtype)  # 得到可以恢复错误症状翻转的纯错误
                log("Pure errors generating end.")

                # recovers = pes
                log("Check generating start...")
                check = mod2.opt_prod(pes, errors)  # 矩阵加法
                log("Check generating end.")
                log("Logical errors generating start...")
                logical_errors = mod2.commute(check, code.logical_opt)  # 看看有无逻辑错误
                log("Logical errors generating end.")

                log("logical errors trans to 1d start...")
                logical_errors_1d = logical_errors[:, 0] * 2 + logical_errors[:, 1]
                logical_errors_1d = logical_errors_1d.unsqueeze(1)
                log("logical errors trans to 1d end.")

                log("Writing image syndromes to h5py file start...")
                dataset_syndr.resize(dataset_syndr.shape[0] + batch_size, axis=0)
                dataset_syndr[-batch_size:] = syndromes
                log("Writing image syndromes to h5py file end.")

                log("Writing logical errors to h5py file start...")
                dataset_le.resize(dataset_le.shape[0] + batch_size, axis=0)
                dataset_le[-batch_size:] = logical_errors_1d.cpu().numpy()
                log("Writing logical errors to h5py file end.")

                log("Num {} batch end.".format(i))
        log("Error rate {} success!".format(p))


gtp = args.gtp
if gtp == 'gib':
    gen_img_batch()
elif gtp == 'gibe':
    gen_img_batch_eval()
elif gtp == 'gbt':
    gen_batch_torc()
elif gtp == 'gbte':
    gen_batch_torc_eval()
elif gtp == 'gb':
    gen_batch()
elif gtp == 'gbe':
    gen_batch_eval()

# nohup python gen_trndt_pe.py --c_type 'sur' --d 9 --k 1 --trnsz 10000000 --seed 0 --gtp gb --low_p 0.01 --high_p 0.04 --num_p 4 > logs/gen_trndt_pe_sur_d9_s0.log &
# nohup python gen_trndt_pe.py --c_type 'sur' --d 9 --k 1 --trnsz 10000000 --seed 0 --gtp gb --low_p 0.05 --high_p 0.09 --num_p 5 > logs/gen_trndt_pe_sur_d9_s0_1.log &
# nohup python gen_trndt_pe.py --c_type 'sur' --d 9 --k 1 --trnsz 10000000 --seed 0 --gtp gb --low_p 0.10 --high_p 0.14 --num_p 5 > logs/gen_trndt_pe_sur_d9_s0_2.log &
# nohup python gen_trndt_pe.py --c_type 'sur' --d 9 --k 1 --trnsz 10000000 --seed 0 --gtp gb --low_p 0.15 --high_p 0.20 --num_p 6 > logs/gen_trndt_pe_sur_d9_s0_3.log &

# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 5 --k 2 --trnsz 10000000 --single_p 0.1 --seed 0 > logs/gen_trndt_pe_torc_d51345.log &
# nohup python gen_trndt_pe.py --gtp gbte --c_type 'torc' --d 5 --k 2 --trnsz 10000000 --single_p 0.1 --eval_seed 1 > logs/gen_trndt_pe_torc_d534342.log &

# 7
# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 7 --k 2 --trnsz 10000000 --single_p 0.1 --seed 0 > logs/gen_trndt_pe_torc_d51sfs345.log &
# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 7 --k 2 --trnsz 10000000 --single_p 0.01 --seed 0 > logs/gen_trndt_pe_torc_d51sfs345.log &
# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 7 --k 2 --trnsz 10000000 --single_p 0.05 --seed 0 > logs/gen_trndt_pe_torc_d51s1fs345.log &
# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 7 --k 2 --trnsz 10000000 --single_p 0.15 --seed 0 > logs/gen_trndt_pe_torc_d51sf2s345.log &
# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 7 --k 2 --trnsz 10000000 --single_p 0.20 --seed 0 > logs/gen_trndt_pe_torc_d51sfs3345.log &
# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 7 --k 2 --trnsz 10000000 --single_p 0.01 --seed 0 > logs/gen_trndt_pe_torc_d51sfs3445.log &
# nohup python gen_trndt_pe.py --gtp gbte --c_type 'torc' --d 7 --k 2 --trnsz 10000000 --single_p 0.01 --eval_seed 1 > logs/gen_trndt_pe_torc_d51sfs345.log &
# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 7 --k 2 --trnsz 10000000 --single_p 0.05 --seed 0 > logs/gen_trndt_pe_torc_d51sfs345.log &
# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 7 --k 2 --trnsz 10000000 --single_p 0.15 --seed 0 > logs/gen_trndt_pe_torc_d51sfs345.log &


# 9
# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 9 --k 2 --trnsz 10000000 --single_p 0.01 --seed 0 > logs/gen_trndt_pe_torc_d51sfsfs345.log &
# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 9 --k 2 --trnsz 10000000 --single_p 0.05 --seed 0 > logs/gen_trndt_pe_torc_d51sf1sfs345.log &
# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 9 --k 2 --trnsz 10000000 --single_p 0.10 --seed 0 > logs/gen_trndt_pe_torc_d51sf2sfs345.log &
# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 9 --k 2 --trnsz 10000000 --single_p 0.15 --seed 0 > logs/gen_trndt_pe_torc_d51sf3sfs345.log &

# sur 7
# nohup python gen_trndt_pe.py --gtp gib --c_type 'sur' --d 7 --k 1 --trnsz 10000000 --single_p 0.01 --seed 0 > logs/gen_trndt_pe_torc_d51274345.log &
# nohup python gen_trndt_pe.py --gtp gib --c_type 'sur' --d 7 --k 1 --trnsz 10000000 --single_p 0.05 --seed 0 > logs/gen_trndt_pe_torc_d5127s4345.log &
# nohup python gen_trndt_pe.py --gtp gib --c_type 'sur' --d 7 --k 1 --trnsz 10000000 --single_p 0.10 --seed 0 > logs/gen_trndt_pe_torc_d5127ss4345.log &
# nohup python gen_trndt_pe.py --gtp gib --c_type 'sur' --d 7 --k 1 --trnsz 10000000 --single_p 0.15 --seed 0 > logs/gen_trndt_pe_torc_d5127sss4345.log &
# eval
# nohup python gen_trndt_pe.py --gtp gibe --c_type 'sur' --d 7 --k 1 --trnsz 10000 --single_p 0.01 > logs/gen_trndt_pe_torc_d512743e45.log &
# nohup python gen_trndt_pe.py --gtp gibe --c_type 'sur' --d 7 --k 1 --trnsz 10000 --single_p 0.05 > logs/gen_trndt_pe_torc_d512743e45.log &
# nohup python gen_trndt_pe.py --gtp gibe --c_type 'sur' --d 7 --k 1 --trnsz 10000 --single_p 0.10 > logs/gen_trndt_pe_torc_d512743e45.log &
# nohup python gen_trndt_pe.py --gtp gibe --c_type 'sur' --d 7 --k 1 --trnsz 10000 --single_p 0.15 > logs/gen_trndt_pe_torc_d512743e45.log &
# nohup python gen_trndt_pe.py --gtp gibe --c_type 'sur' --d 7 --k 1 --trnsz 10000 --single_p 0.20 > logs/gen_trndt_pe_torc_d512743e45.log &

# sur 11
# nohup python gen_trndt_pe.py --gtp gib --c_type 'sur' --d 11 --k 1 --trnsz 10000000 --single_p 0.01 --seed 0 > logs/gen_trndt_pe_torc_d51274345.log &
# nohup python gen_trndt_pe.py --gtp gib --c_type 'sur' --d 11 --k 1 --trnsz 10000000 --single_p 0.05 --seed 0 > logs/gen_trndt_pe_torc_d5127sab4345.log &
# nohup python gen_trndt_pe.py --gtp gib --c_type 'sur' --d 11 --k 1 --trnsz 10000000 --single_p 0.10 --seed 0 > logs/gen_trndt_pe_torc_d5127sabb4345.log &
# nohup python gen_trndt_pe.py --gtp gib --c_type 'sur' --d 11 --k 1 --trnsz 10000000 --single_p 0.15 --seed 0 > logs/gen_trndt_pe_torc_d5127sabbb4345.log &
# nohup python gen_trndt_pe.py --gtp gibe --c_type 'sur' --d 11 --k 1 --trnsz 10000 --low_p 0.01 --high_p 0.20 --num_p 20 > logs/gen_trndt_pe_torc_040204.log &


# 3
# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 3 --k 2 --trnsz 10000000 --single_p 0.01 --seed 0 > logs/gen_trndt_pe_torc_d3sfs345.log &
# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 3 --k 2 --trnsz 10000000 --single_p 0.05 --seed 0 > logs/gen_trndt_pe_torc_d3sfs345.log &
# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 3 --k 2 --trnsz 10000000 --single_p 0.10 --seed 0 > logs/gen_trndt_pe_torc_d3s1fs345.log &
# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 5 --k 2 --trnsz 10000000 --single_p 0.10 --seed 0 > logs/gen_trndt_pe_torc_d3s1fs3ee45.log &
# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 3 --k 2 --trnsz 10000000 --single_p 0.15 --seed 0 > logs/gen_trndt_pe_torc_d3sf2s345.log &
# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 3 --k 2 --trnsz 10000000 --single_p 0.20 --seed 0 > logs/gen_trndt_pe_torc_d3sfs3345.log &

# nohup python gen_trndt_pe.py --gtp gibe --c_type 'sur' --d 5 --k 1 --trnsz 10000 --low_p 0.01 --high_p 0.20 --num_p 20 > logs/gen_trndt_pe_torc_040201.log &
# nohup python gen_trndt_pe.py --gtp gbte --c_type 'torc' --d 3 --k 2 --trnsz 10000 --low_p 0.01 --high_p 0.20 --num_p 20 > logs/gen_trndt_pe_torc_040102.log &
# nohup python gen_trndt_pe.py --gtp gbte --c_type 'torc' --d 5 --k 2 --trnsz 10000 --low_p 0.01 --high_p 0.20 --num_p 20 > logs/gen_trndt_pe_torc_040103.log &
# nohup python gen_trndt_pe.py --gtp gbte --c_type 'torc' --d 7 --k 2 --trnsz 10000 --low_p 0.01 --high_p 0.20 --num_p 20 > logs/gen_trndt_pe_torc_040104.log &

# sur 7
# nohup python gen_trndt_pe.py --gtp gib --c_type 'sur' --d 5 --k 1 --trnsz 10000000 --single_p 0.01 --seed 0 > logs/gen_trndt_pe_torc_d51274345.log &
# nohup python gen_trndt_pe.py --gtp gib --c_type 'sur' --d 5 --k 1 --trnsz 10000000 --single_p 0.05 --seed 0 > logs/gen_trndt_pe_torc_d512sab4345.log &
# nohup python gen_trndt_pe.py --gtp gib --c_type 'sur' --d 5 --k 1 --trnsz 10000000 --single_p 0.10 --seed 0 > logs/gen_trndt_pe_torc_d517sabb4345.log &
# nohup python gen_trndt_pe.py --gtp gib --c_type 'sur' --d 5 --k 1 --trnsz 10000000 --single_p 0.15 --seed 0 > logs/gen_trndt_pe_torc_d527sabbb4345.log &

# nohup python gen_trndt_pe.py --gtp gib --c_type 'sur' --d 5 --k 1 --trnsz 10000000 --single_p 0.20 --seed 0 > logs/gen_trndt_pe_torc_d5270sabbb4345.log &
# nohup python gen_trndt_pe.py --gtp gib --c_type 'sur' --d 7 --k 1 --trnsz 10000000 --single_p 0.20 --seed 0 > logs/gen_trndt_pe_torc_d527s0abbb4345.log &
# nohup python gen_trndt_pe.py --gtp gib --c_type 'sur' --d 11 --k 1 --trnsz 10000000 --single_p 0.20 --seed 0 > logs/gen_trndt_pe_torc_d527sa0bbb4345.log &

# nohup python gen_trndt_pe.py --gtp gib --c_type 'sur' --d 7 --k 1 --trnsz 10000000 --single_p 0.15 --seed 0 > logs/gen_trndt_pe_torc_d5127sss4345.log &
# nohup python gen_trndt_pe.py --gtp gib --c_type 'sur' --d 9 --k 1 --trnsz 10000000 --single_p 0.20 --seed 0 > logs/gen_trndt_pe_torc_d5127ss345.log &
# nohup python gen_trndt_pe.py --gtp gib --c_type 'sur' --d 3 --k 1 --trnsz 10000000 --single_p 0.20 --seed 0 > logs/gen_trndt_pe_torc_d51827ss345.log &

# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 5 --k 2 --trnsz 10000000 --single_p 0.20 --seed 0 > logs/gen_trndt_pe_torc_d3d3345.log &
# nohup python gen_trndt_pe.py --gtp gbt --c_type 'torc' --d 7 --k 2 --trnsz 10000000 --single_p 0.20 --seed 0 > logs/gen_trndt_pe_torc_d3ffs3345.log &

# sur gb 3
# nohup python gen_trndt_pe.py --gtp gb --c_type 'sur' --d 3 --k 1 --trnsz 10000000 --single_p 0.01 --seed 0 > logs/gen_trndt_pe_torc_d31sfsffgb45.log &
# nohup python gen_trndt_pe.py --gtp gb --c_type 'sur' --d 3 --k 1 --trnsz 10000000 --single_p 0.05 --seed 0 > logs/gen_trndt_pe_torc_d31sf1sffgb45.log &
# nohup python gen_trndt_pe.py --gtp gb --c_type 'sur' --d 3 --k 1 --trnsz 10000000 --single_p 0.10 --seed 0 > logs/gen_trndt_pe_torc_d31sf2sffgb45.log &
# nohup python gen_trndt_pe.py --gtp gb --c_type 'sur' --d 3 --k 1 --trnsz 10000000 --single_p 0.15 --seed 0 > logs/gen_trndt_pe_torc_d31sf3sffgb45.log &
# nohup python gen_trndt_pe.py --gtp gb --c_type 'sur' --d 3 --k 1 --trnsz 10000000 --single_p 0.20 --seed 0 > logs/gen_trndt_pe_torc_d31sf4sffgb45.log &

# sur gb 5
# nohup python gen_trndt_pe.py --gtp gb --c_type 'sur' --d 5 --k 1 --trnsz 10000000 --single_p 0.01 --seed 0 > logs/gen_trndt_pe_torc_d31sfsffgb45.log &
# nohup python gen_trndt_pe.py --gtp gb --c_type 'sur' --d 5 --k 1 --trnsz 10000000 --single_p 0.05 --seed 0 > logs/gen_trndt_pe_torc_d31sf11sffgb45.log &
# nohup python gen_trndt_pe.py --gtp gb --c_type 'sur' --d 5 --k 1 --trnsz 10000000 --single_p 0.10 --seed 0 > logs/gen_trndt_pe_torc_d31sf2s1ffgb45.log &
# nohup python gen_trndt_pe.py --gtp gb --c_type 'sur' --d 5 --k 1 --trnsz 10000000 --single_p 0.15 --seed 0 > logs/gen_trndt_pe_torc_d31sf3sf1fgb45.log &
# nohup python gen_trndt_pe.py --gtp gb --c_type 'sur' --d 5 --k 1 --trnsz 10000000 --single_p 0.20 --seed 0 > logs/gen_trndt_pe_torc_d31sf4sff1gb45.log &

# sur gb 7
# nohup python gen_trndt_pe.py --gtp gb --c_type 'sur' --d 7 --k 1 --trnsz 10000000 --single_p 0.01 --seed 0 > logs/gen_trndt_pe_torc_d31sfsffgb45.log &
# nohup python gen_trndt_pe.py --gtp gb --c_type 'sur' --d 7 --k 1 --trnsz 10000000 --single_p 0.05 --seed 0 > logs/gen_trndt_pe_torc_d31sfffgb45.log &
# nohup python gen_trndt_pe.py --gtp gb --c_type 'sur' --d 7 --k 1 --trnsz 10000000 --single_p 0.10 --seed 0 > logs/gen_trndt_pe_torc_d31sf2fgb45.log &
# nohup python gen_trndt_pe.py --gtp gb --c_type 'sur' --d 7 --k 1 --trnsz 10000000 --single_p 0.15 --seed 0 > logs/gen_trndt_pe_torc_d31sf3fgb45.log &
# nohup python gen_trndt_pe.py --gtp gb --c_type 'sur' --d 7 --k 1 --trnsz 10000000 --single_p 0.20 --seed 0 > logs/gen_trndt_pe_torc_d31sf4sfb45.log &

# sur gb 9
# nohup python gen_trndt_pe.py --gtp gb --c_type 'sur' --d 9 --k 1 --trnsz 10000000 --single_p 0.01 --seed 0 > logs/gen_trndt_pe_torc_d319fsffgb45.log &
# nohup python gen_trndt_pe.py --gtp gb --c_type 'sur' --d 9 --k 1 --trnsz 10000000 --single_p 0.05 --seed 0 > logs/gen_trndt_pe_torc_d31s9fffgb45.log &
# nohup python gen_trndt_pe.py --gtp gb --c_type 'sur' --d 9 --k 1 --trnsz 10000000 --single_p 0.10 --seed 0 > logs/gen_trndt_pe_torc_d31sf92fgb45.log &
# nohup python gen_trndt_pe.py --gtp gb --c_type 'sur' --d 9 --k 1 --trnsz 10000000 --single_p 0.15 --seed 0 > logs/gen_trndt_pe_torc_d31sf39fgb45.log &
# nohup python gen_trndt_pe.py --gtp gb --c_type 'sur' --d 9 --k 1 --trnsz 10000000 --single_p 0.20 --seed 0 > logs/gen_trndt_pe_torc_d31sf4s9fb45.log &
