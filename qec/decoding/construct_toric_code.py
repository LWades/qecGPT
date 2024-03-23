from args import args
import torch
import random
import sys
from os.path import abspath, dirname, exists

sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import Surfacecode, Abstractcode, Toric, Rotated_Surfacecode, Sur_3D, QuasiCyclicCode, Toricode
from module import mod2, read_code, Loading_code, log

import matplotlib.pyplot as plt
import numpy as np

toric = Toricode(3)
log("g_stablizer (shape={}): \n{}".format(toric.g_stabilizer.shape, toric.g_stabilizer))
log("PCM (shape={}): \n{}".format(toric.PCM.shape, toric.PCM))
log("logical_opt (shape={}): \n{}".format(toric.logical_opt.shape, toric.logical_opt))
log("pure_es (shape={}): \n{}".format(toric.pure_es.shape, toric.pure_es))

# plt.imshow(toric.g_stabilizer.numpy(), cmap='gray_r')  # 'viridis'是一种颜色映射
# plt.colorbar()  # 显示颜色条
# plt.xticks(np.arange(0, toric.g_stabilizer.shape[0], 1))  # x轴刻度
# plt.yticks(np.arange(0, toric.g_stabilizer.shape[0], 1))  # y轴刻度
# plt.show()  # 显示图表

plt.imshow(toric.PCM.numpy(), cmap='gray_r')  # 'viridis'是一种颜色映射
# plt.colorbar()  # 显示颜色条
plt.xticks(np.arange(0, toric.PCM.shape[1], 1))  # x轴刻度
plt.yticks(np.arange(0, toric.PCM.shape[0], 1))  # y轴刻度
plt.show()  # 显示图表
