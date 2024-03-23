from args import args
import torch
import random
import sys
from os.path import abspath, dirname, exists

sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import Surfacecode, Abstractcode, Toric, Rotated_Surfacecode, Sur_3D, QuasiCyclicCode, Toricode
from module import mod2, read_code, Loading_code

mod2 = mod2()  # dtype = torch.float32, device='cuda:0'

def code_generator(d, k, seed, c_type='sur'):
    '''
        generate and save code
        code: code class
        d: label the original code.
        k: if k < code.k , this function will delete some stabilizers from original code. And the actuall distance d' of new code will be changed.
    '''
    path = abspath(dirname(__file__)).strip('decoding') + 'code/' + c_type + '_d{}_k{}_seed{}'.format(d, k, seed)
    print(path)
    if False:
    # if exists(path):
        print('code exists')
        info = read_code(d, k, seed, c_type=c_type)
        A = Loading_code(info)
        m = A.m
    else:
        print("创建新的")
        if c_type == 'sur':
            S = Surfacecode(d)
        elif c_type == 'rsur':
            S = Rotated_Surfacecode(d)
        elif c_type == 'tor':
            S = Toric(d)
        elif c_type == 'torc':
            S = Toricode(d)
        elif c_type == '3d':
            S = Sur_3D(d)
            print(S.g_stabilizer.size())

        m = S.n - k
        print(m)
        if k == 1 and (c_type == 'sur'):
            A = S
            A.logical_opt = A.logical_opt[[1, 2]]   # 在这里，Surface code 的逻辑算符只取逻辑 Z 和 X，
        elif k == 1 and c_type == 'rsur':
            A = S
        elif k == 2 and c_type == 'torc':
            A = S
            A.logical_opt = A.logical_opt[[1, 2, 4, 5]]
        else:
            def random_remove(code, k, seed=0):
                random.seed(seed)
                sta_list = list(range(code.m))
                indices = random.sample(sta_list, m)
                indices.sort()
                return indices

            sta_list = random_remove(S, k, seed)
            print(sta_list)
            reduce_g = S.g_stabilizer[sta_list]
            A = Abstractcode(reduce_g, complete=True)

        torch.save((A.g_stabilizer, A.logical_opt, A.pure_es), path)

    print('Commute--stabilizer with logical :', (mod2.commute(A.g_stabilizer, A.logical_opt).sum() == 0).item())
    print('Commute--pure error with logical :', (mod2.commute(A.pure_es, A.logical_opt).sum() == 0).item())
    print('Anticommute--pure error with stabilizer :',
          ((mod2.commute(A.g_stabilizer, A.pure_es) - torch.eye(m)).sum() == 0).item())
    print('Commute--pure error with pure error :', (mod2.commute(A.pure_es, A.pure_es).sum() == 0).item())
    print('Anti-Commute--logicals : ')
    print(mod2.commute(A.logical_opt, A.logical_opt))
    # print(mod2.commute(A.pure_es, A.pure_es))


def qcc_generator(l, m, polynomial_a, polynomial_b):
    C = QuasiCyclicCode(l, m, polynomial_a, polynomial_b)
    A = Abstractcode(C.stabilizers)
    n, k = A.n, A.n - A.m
    print('n:', n)
    print('k:', k)

    print('Commute--stabilizer with logical :', (mod2.commute(A.g_stabilizer, A.logical_opt).sum() == 0).item())
    print('Commute--pure error with logical :', (mod2.commute(A.pure_es, A.logical_opt).sum() == 0).item())
    print('Anticommute--pure error with stabilizer :',
          ((mod2.commute(A.g_stabilizer, A.pure_es) - torch.eye(A.m)).sum() == 0).item())
    print('Commute--pure error with pure error :', (mod2.commute(A.pure_es, A.pure_es).sum() == 0).item())
    print('Anti-Commute--logicals : ')
    print(mod2.commute(A.logical_opt, A.logical_opt))

    path = abspath(dirname(__file__)).strip('decoding') + 'code/' + c_type + '_n{}_k{}'.format(n, k)
    if exists(path):
        None
    # print(path)
    else:
        torch.save((A.g_stabilizer, A.logical_opt, A.pure_es), path)


print(args)

c_type = args.c_type  # args.c_type
# c_type = 'qcc'  # args.c_type
if c_type == 'qcc':
    l, m, polynomial_a, polynomial_b = 12, 12, [3, 2, 7], [3, 1, 2]
    qcc_generator(l, m, polynomial_a, polynomial_b)
else:
    d, k, seed = args.d, args.k, args.seed
    code_generator(d, k, seed, c_type=c_type)
