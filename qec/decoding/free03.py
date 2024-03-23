import numpy as np
import sys
from os.path import abspath, dirname, exists

sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import Errormodel, mod2, Loading_code, read_code, log, Toricode, Surfacecode
from ldpc import mod2
# 示例矩阵
# matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
code = Toricode(3)
# code = Toricode(7)
log(f"PCM:\n{code.PCM}")
# log(f"PCM[0]:\n{code.PCM[0]}")
# print(type(code.PCM))
# 计算矩阵的秩
rank = mod2.rank(code.PCM.numpy())
# rank = np.linalg.matrix_rank(code.PCM)
print(f"NumPy - Matrix rank: {rank}")
