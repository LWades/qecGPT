from args import args
import torch
import time
import sys
from os.path import abspath, dirname, exists

sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import Errormodel, mod2, Loading_code, read_code, log, Toricode, Surfacecode


def trans_org_syndr_to_image_syndr(L, syndr):
    log("org_syndr(L={}, shape={}):\n{}".format(L, syndr.shape, syndr))
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
    log("image_syndr(L={}, shape={}):\n{}".format(L, image_syndr.shape, image_syndr))
    return image_syndr


device, dtype = 'cpu', torch.float64
# device, dtype = 'cuda:5', torch.float64
trials = 2
# c_type = 'torc'
# c_type = 'tor'
c_type = 'sur'
# c_type = 'sur'
# d, k, seed = 3, 2, 0
d, k, seed = 5, 1, 0
# d, k, seed = 5, 1, 0
t_p = 5
error_seed = 10000

info = read_code(d, k, seed, c_type=c_type)
code = Loading_code(info)
mod2 = mod2(device=device, dtype=dtype)

# code = Surfacecode(3)
# code = Toricode(3)
# code = Torcode(5)
error_rate = torch.linspace(0.2, 0.2, 1)
# error_rate = torch.linspace(0.01, 0.368, 19)
lo_rate = []

log("pure_es: (shape={})\n{}".format(code.pure_es.shape, code.pure_es))
log("g_stab: (shape={})\n{}".format(code.g_stabilizer.shape, code.g_stabilizer))
log("logical_opt:(shape={})\n{}".format(code.logical_opt.shape, code.logical_opt))
log("stabilizer & logical commute: \n{}".format(mod2.commute(code.g_stabilizer, code.logical_opt)))
log("pure_es & stabilizer commute: \n{}".format(mod2.commute(code.g_stabilizer, code.pure_es)))
log("pure_es & logical_opt commute: \n{}".format(mod2.commute(code.logical_opt, code.pure_es)))
for i in range(len(error_rate)):
    # for i in range(error_rate):
    E = Errormodel(e_rate=error_rate[i])
    log("E: {}".format(E))

    error = E.generate_error(n=code.n, m=trials, seed=seed)
    # error = torch.tensor([[1, 0, 1, 2, 0, 2, 0, 0, 0, 0, 3, 0, 0],
    #                       [1, 0, 1, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0]])
    log("error type: {}".format(type(error)))
    log("error(shape={}):\n{}".format(error.shape, error))



    syndrome = mod2.commute(error, code.g_stabilizer)
    # syndrome = torch.tensor([[1,1,0,1,0,0,0,1,0,0,1,0],[1,1,0,1,0,0,0,0,0,1,1,0]])
    # syndrome = torch.tensor([[1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
    #                          [1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.]])
    log("syndrome(shape={}):\n{}".format(syndrome.shape, syndrome))

    pe = E.pure(code.pure_es, syndrome, device=device, dtype=dtype)
    log("pe(shape={}):\n{}".format(pe.shape, pe))

    recover = pe
    check = mod2.opt_prod(recover, error)
    log("check(shape={}):\n{}".format(check.shape, check))

    ss = mod2.commute(check, code.g_stabilizer)
    # ss = code.g_stabilizer @ check.T
    log("ss(shape={}):\n{}".format(ss.shape, ss))

    commute = mod2.commute(check, code.logical_opt)
    log("commute(shape={}):\n{}".format(commute.shape, commute))

    fail = torch.count_nonzero(commute.sum(1)).cpu().item()
    log("fail: {}".format(fail))

    logical_error_rate = fail / trials
    log("logical_error_rate: {}".format(logical_error_rate))

    # image_syndrome = torch.empty(trials, 2*d-1, 2*d-1)
    # for j in range(len(syndrome)):
    #     image_syndrome[j] = trans_org_syndr_to_image_syndr(d, syndrome[j])
    # log("image_syndrome(shape={}):\n{}".format(image_syndrome.shape, image_syndrome))

    lo_rate.append(logical_error_rate)

print(lo_rate)
