import numpy as np

def mod2_gauss_jordan_elimination(H):
    m, n = H.shape
    assert m <= n, "H should have more columns than rows."

    # 构建增广矩阵 [H | I_m]
    I_m = np.eye(m, dtype=int)
    aug_matrix = np.concatenate((H, I_m), axis=1)

    # 高斯-约旦消元法在模 2 下进行
    for i in range(m):
        if aug_matrix[i, i] == 0:
            for j in range(i+1, m):
                if aug_matrix[j, i] == 1:
                    aug_matrix[[i, j]] = aug_matrix[[j, i]]  # 交换行
                    break

        for j in range(m):
            if j != i and aug_matrix[j, i] == 1:
                aug_matrix[j] = (aug_matrix[j] + aug_matrix[i]) % 2

    # 提取 B^T 并返回 B
    B_T = aug_matrix[:, n:]
    return B_T.T

# 示例
m, n = 12, 13  # m 和 n 的值应根据实际情况设置
H = np.random.randint(0, 2, (m, 2*n))

B = mod2_gauss_jordan_elimination(H)
print("Matrix B:")
print(B)
print(B.shape[0], "x", B.shape[1])
