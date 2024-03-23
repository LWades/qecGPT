import numpy as np

def find_matrix_B(H, iterations=1000):
    m, n = H.shape
    B = np.random.randint(0, 2, (2 * n, m))  # 初始化 B 为随机 0 和 1

    for _ in range(iterations):
        # 计算 H * B^T，并进行模 2 运算
        product = (H @ B.T) % 2

        # 计算误差（这里简单地使用单位矩阵和 product 的差）
        error = (np.eye(m, dtype=int) - product) % 2

        # 根据误差调整 B，这里需要一个有效的策略
        # 例如，随机选择 B 中的元素进行翻转
        for i in range(len(error)):
            if np.any(error[i] != 0):  # 如果当前行有误差
                B[:, i] = (B[:, i] + 1) % 2  # 翻转 B 的对应列

    return B

# 示例
m, n = 3, 3  # m 和 n 的值应根据实际情况设置
H = np.random.randint(0, 2, (m, 2 * n))

B = find_matrix_B(H)
print("Matrix B:")
print(B)