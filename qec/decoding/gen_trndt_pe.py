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


def ZX2image(d, zxsyndrome):
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

info = read_code(d, k, code_seed, c_type=c_type)
code = Loading_code(info)
mod2 = mod2(device=device, dtype=dtype)

# p = 0.1
# ps = torch.linspace(0.01, 0.10, 10)
# ps = torch.linspace(0.16, 0.20, 5)
# ps = torch.linspace(0.11, 0.15, 5)
ps = torch.linspace(0.01, 0.20, 20)
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

def gen_batch_eval():
    import h5py
    batch_nums = trnsz // batch_size
    # log("batch_size: {}".format(batch_size))
    for p in ps:
        log("Error rate now: {}".format(p))
        file_name = "/root/Surface_code_and_Toric_code/{}_pe/{}_d{}_p{}_trnsz{}_eval_seed{}".format(c_type, c_type, d, format(p, '.3f'), trnsz, seed)
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

def gen_batch():
    import h5py
    batch_nums = trnsz // batch_size
    # log("batch_size: {}".format(batch_size))
    for p in ps:
        log("Error rate now: {}".format(format(p, '.2f')))
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
    import h5py
    batch_nums = trnsz // batch_size
    log("batch_size: {}".format(batch_size))
    for p in ps:
        log("Error rate now: {}".format(format(p, '.2f')))
        file_name = "/root/Surface_code_and_Toric_code/{}_pe/{}_d{}_p{}_trnsz{}_imgsdr_seed{}".format(c_type, c_type, d, format(p, '.3f'), trnsz, seed)
        with h5py.File(file_name + ".hdf5", 'w') as f:
            log("Open h5py file success")
            dataset_syndr = f.create_dataset('image_syndromes', shape=(0, 2*d-1, 2*d-1), maxshape=(None, 2*d-1, 2*d-1), chunks=True, compression="gzip")
            dataset_le = f.create_dataset('logical_errors', shape=(0, 1), maxshape=(None, 1), chunks=True, compression="gzip")
            log("Dataset create success")
            for i in range(batch_nums):
                log("Num {} batch start..., {} in all".format(i, batch_nums))

                E = Errormodel(e_rate=p, e_model='depolarized')
                log("Errors generating start...")
                errors = E.generate_error(n=code.n, m=batch_size, seed=seed)     # 这里改成了 batch_size
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
                image_syndromes = torch.empty(batch_size, 2 * d - 1, 2 * d - 1)
                for j in range(len(syndromes)):
                    image_syndromes[j] = trans_org_syndr_to_image_syndr(d, syndromes[j])
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


def gen_img_batch_eval():
    batch_nums = trnsz // batch_size
    log("batch_size: {}".format(batch_size))
    for p in ps:
        log("Error rate now: {}".format(format(p, '.2f')))
        file_name = "/root/Surface_code_and_Toric_code/{}_pe_img/{}_d{}_p{}_trnsz{}_imgsdr_eval_seed{}".format(c_type, c_type, d, format(p, '.3f'), trnsz, seed)
        with h5py.File(file_name + ".hdf5", 'w') as f:
            log("Open h5py file success")
            dataset_syndr = f.create_dataset('image_syndromes', shape=(0, 2*d-1, 2*d-1), maxshape=(None, 2*d-1, 2*d-1), chunks=True, compression="gzip")
            dataset_le = f.create_dataset('logical_errors', shape=(0, 1), maxshape=(None, 1), chunks=True, compression="gzip")
            log("Dataset create success")
            for i in tqdm(range(batch_nums)):
                log("Num {} batch start..., {} in all".format(i, batch_nums))

                E = Errormodel(e_rate=p, e_model='depolarized')
                log("Errors generating start...")
                errors = E.generate_error(n=code.n, m=batch_size, seed=seed)     # 这里改成了 batch_size
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
                    image_syndromes[j] = ZX2image(d, syndromes[j])
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


gen_img_batch_eval()
# gen_img_batch_eval()
# gen_batch_eval()
# nohup python gen_trndt_pe.py -c_type 'sur' --d 5 --trnsz 10000000 > logs/gen_trndt_pe_batch.log &
# nohup python gen_trndt_pe.py -c_type 'sur' --d 5 --trnsz 10000000 > logs/gen_trndt_pe_batch_d5_20to11.log &
# nohup python gen_trndt_pe.py -c_type 'sur' --d 7 --trnsz 10000000 > logs/gen_trndt_pe_batch_d7_20to16.log &
# nohup python gen_trndt_pe.py -c_type 'sur' --d 7 --trnsz 10000000 > 4logs/gen_trndt_pe_batch_d7_15to11.log &
# nohup python gen_trndt_pe.py -c_type 'sur' --d 7 --trnsz 10000000 > logs/gen_trndt_pe_batch_d7_10to06.log &
# nohup python gen_trndt_pe.py -c_type 'sur' --d 5 --trnsz 10000000 > logs/gen_trndt_pe_d5_0.130.log &
# nohup python gen_trndt_pe.py -c_type 'sur' --d 5 --trnsz 10000000 > logs/gen_trndt_pe_d5_0.09.log &
# nohup python gen_trndt_pe.py --c_type 'sur' --d 5 --trnsz 10000 > logs/gen_trndt_pe_d5_eval.log &
# nohup python gen_trndt_pe.py --c_type 'torc' --d 3 --k 2 --trnsz 10000000 > logs/gen_trndt_pe_torc_d3.log &
# nohup python gen_trndt_pe.py --c_type 'torc' --d 5 --k 2 --trnsz 10000000 > logs/gen_trndt_pe_torc_d5.log &
# nohup python gen_trndt_pe.py --c_type 'torc' --d 3 --k 2 --trnsz 10000 > logs/gen_trndt_pe_torc_d3_eval.log &
# nohup python gen_trndt_pe.py --c_type 'torc' --d 3 --k 2 --trnsz 10000 --seed 1 > logs/gen_trndt_pe_torc_d3_eval_s1.log &
# nohup python gen_trndt_pe.py --c_type 'sur' --d 3 --k 1 --trnsz 10000 --seed 1 > logs/gen_trndt_pe_sur_d3_eval_s1.log &
# nohup python gen_trndt_pe.py --c_type 'torc' --d 5 --k 2 --trnsz 10000 --seed 1 > logs/gen_trndt_pe_torc_d5_eval_s1.log &
# nohup python gen_trndt_pe.py --c_type 'sur' --d 5 --k 1 --trnsz 10000 --seed 1 > logs/gen_trndt_pe_sur_d5_eval_s1.log &
