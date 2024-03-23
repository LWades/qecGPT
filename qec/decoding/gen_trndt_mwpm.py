from args import args
import torch
import sys
from os.path import abspath, dirname

sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import Errormodel, mod2, Loading_code, read_code, log
from tqdm import tqdm
from pymatching import Matching
import numpy as np
from rich import print
from rich.panel import Panel

d, k, seed, c_type, trnsz, device, dtype = args.d, args.k, args.seed, args.c_type, args.trnsz, 'cpu', torch.float64

info = read_code(d, k, seed, c_type=c_type)
code = Loading_code(info)
mod2 = mod2(device=device, dtype=dtype)

ps = torch.linspace(0.01, 0.20, 20)
# ps = [0.09]
log("error rate list: {}".format(ps))
print(Panel.fit("Code: {}, d: {}, seed: {}, error model: {}, trnsz: {}".format(c_type, d, seed, 'depolarized', trnsz), title='Parameters'))
print(Panel.fit('Base information'))
log(f"g_stabilizer (shape={code.g_stabilizer.shape}):\n{code.g_stabilizer}")
log(f'PCM (shape={code.PCM.shape}):\n{code.PCM}')
log(f'logical_opt (shape={code.logical_opt.shape}):\n{code.logical_opt}')

for p in ps:
    log("0 - Error rate now: {}".format(p))
    p_tensor = torch.tensor(p)

    log("1 - Init Error model...")
    E = Errormodel(e_rate=p, e_model='depolarized')
    log("1 - Init Error model... Done.")

    log("2 - Init MWPM...")
    log("2.1 - Init weights...")
    weights = torch.ones(2 * code.n) * torch.log((1 - p_tensor) / p_tensor)
    log("2.1 - Init weights... Done.")
    log("2.2 - Init MWPM Decoder...")
    decoder_mwpm = Matching(code.PCM, weights=weights)
    log("2.2 - Init MWPM Decoder... Done.")

    log("3 - Errors generating...")
    errors = E.generate_error(n=code.n, m=trnsz, seed=seed)
    log("3 - Errors generating... Done.")

    log("4 - Syndromes generating...")
    syndromes = mod2.commute(errors, code.g_stabilizer)
    log("4 - Syndromes generating... Done.")

    log("5 - Errors preprocess...")
    errors_post = mod2.rep(errors).squeeze().int().numpy()
    log("5 - Errors preprocess... Done.")

    # log("5 - Pure errors generating... Done")
    # pes = E.pure(code.pure_es, syndromes, device=device, dtype=dtype)
    # log("5 - Pure errors generating end.")
    checks = torch.empty(trnsz, code.n * 2)
    log("5 - MWPM Decoding...")
    for j in tqdm(range(trnsz)):
        error = errors_post[j]
        syndrome = syndromes[j]
        recover = decoder_mwpm.decode(syndrome)
        check = (recover + error) % 2
        checks[j] = torch.tensor(check)

    log("5 - MWPM Decoding... Done.")
    log("5.1 - Test MWPM decoding result...")
    if (mod2.commute(mod2.xyz(checks), code.g_stabilizer).sum() == 0).item():
        log("5.1.1 - Perfect.")
    else:
        log("5.1.1 - So sad..")
    log("5.1 - Test MWPM decoding result... Done.")
    # log("Check generating start...")
    # check = mod2.opt_prod(pes, errors)
    # log("Check generating end.")
    log("6 - Logical errors generating...")
    logical_errors = mod2.commute(mod2.xyz(checks), code.logical_opt)
    log("6 - Logical errors generating... Done.")

    # log("logical errors trans to 1d start...")
    # logical_errors_1d = logical_errors[:, 0] * 2 + logical_errors[:, 1]
    # logical_errors_1d = logical_errors_1d.unsqueeze(1)
    # log("logical errors trans to 1d end.")

    log("7 - Writing to npz file...")
    file_name = "../trndt/toricode/{}_d{}_p{}_trnsz{}_eval".format(c_type, d, format(p, '.3f'), trnsz)
    np.savez_compressed(file_name, syndromes=syndromes.cpu().numpy(),
                        logical_errors=logical_errors.cpu().numpy())
    log("7 - Writing to npz file end... Done.")
# nohup python gen_trndt_pe.py -c_type 'sur' --d 5 --trnsz 10000000 > logs/gen_trndt_pe_05.log &
# nohup python gen_trndt_mwpm.py --c_type 'torc' --d 3 --trnsz 10000 > logs/gen_trndt_mwpm_d3_eval.log &
