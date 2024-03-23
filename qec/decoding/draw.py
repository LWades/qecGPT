import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np

x_stabilizer_tangle_line_color = '#FFE137'
z_stabilizer_tangle_line_color = '#12D273'
color_X_Z_stabilizer_font = '#808080'


def get_lower_left_corner(center, width, height):
    return center[0] - width / 2, center[1] - height / 2


def draw_toric_code(d):
    n = 2 * d
    fig, axes = plt.subplots(dpi=600)
    axes.set_title('Toric Code (d = {})'.format(d))
    axes.invert_yaxis()

    # 计算当前码距下各元素（量子比特+纠缠线）的大小
    size_qubit = 1 / ((n-1) * 6 + 2)
    width_line = 0.8 * 4 * size_qubit
    height_line = size_qubit
    center_qubit_x = 6 * size_qubit
    center_qubit_y = 6 * size_qubit
    center_tangle_line_horizontal_x = 6 * size_qubit
    center_tangle_line_horizontal_y = 6 * size_qubit
    center_tangle_line_vertical_x = 6 * size_qubit
    center_tangle_line_vertical_y = 6 * size_qubit
    font_size = 12 * 5 / n
    offset_font_correct_y = 0.115 * size_qubit
    offset_font_correct_x = 0.02 * size_qubit

    # 绘制量子比特（数据 + 测量）
    for i in range(n):
        for j in range(n):
            center_qubit = (center_qubit_x * i + size_qubit, center_qubit_y * j + size_qubit)
            if (i + j) % 2 == 0:    # 测量量子比特
                data_qubit = Circle(center_qubit, size_qubit, facecolor='black', edgecolor='black')
            else:
                data_qubit = Circle(center_qubit, size_qubit, facecolor='white', edgecolor='black')
            axes.add_patch(data_qubit)

    # 绘制纠缠线
    for i in range(n):  # 水平
        for j in range(n-1):
            tangle_line_center = (center_tangle_line_horizontal_x * j + 4 * size_qubit, center_tangle_line_horizontal_y * i + size_qubit)
            tangle_line_lower_corner = get_lower_left_corner(tangle_line_center, width_line, height_line)
            if i % 2 == 0:  # Z 稳定子
                tangle_line = Rectangle(tangle_line_lower_corner, width_line, height_line,
                                        facecolor=z_stabilizer_tangle_line_color,
                                        edgecolor=z_stabilizer_tangle_line_color)
                axes.text(tangle_line_center[0] - offset_font_correct_x, tangle_line_center[1] + offset_font_correct_y, 'Z', ha='center', va='center',
                          fontsize=font_size, color=color_X_Z_stabilizer_font)
            else:
                tangle_line = Rectangle(tangle_line_lower_corner, width_line, height_line,
                                        facecolor=x_stabilizer_tangle_line_color,
                                        edgecolor=x_stabilizer_tangle_line_color)
                axes.text(tangle_line_center[0] - offset_font_correct_x, tangle_line_center[1] + offset_font_correct_y, 'X', ha='center', va='center',
                          fontsize=font_size, color=color_X_Z_stabilizer_font)
            axes.add_patch(tangle_line)
    for i in range(n-1):    # 垂直
        for j in range(n):
            tangle_line_center = (center_tangle_line_vertical_x * j + size_qubit, center_tangle_line_vertical_y * i + 4 * size_qubit)
            tangle_line_lower_corner = get_lower_left_corner(tangle_line_center, height_line, width_line)
            if j % 2 == 0:  # X 稳定子
                tangle_line = Rectangle(tangle_line_lower_corner, height_line, width_line,
                                        facecolor=z_stabilizer_tangle_line_color,
                                        edgecolor=z_stabilizer_tangle_line_color)
                axes.text(tangle_line_center[0] - offset_font_correct_x, tangle_line_center[1] + offset_font_correct_y, 'Z', ha='center', va='center', fontsize=font_size, color=color_X_Z_stabilizer_font)
            else:
                tangle_line = Rectangle(tangle_line_lower_corner, height_line, width_line,
                                        facecolor=x_stabilizer_tangle_line_color,
                                        edgecolor=x_stabilizer_tangle_line_color)
                axes.text(tangle_line_center[0] - offset_font_correct_x, tangle_line_center[1] + offset_font_correct_y, 'X', ha='center', va='center',
                          fontsize=font_size, color=color_X_Z_stabilizer_font)
            axes.add_patch(tangle_line)
    axes.set_aspect('equal')    # 保证图不会变形
    # plt.axis('off')
    plt.show()


draw_toric_code(3)