import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator
from matplotlib import rcParams
from matplotlib import font_manager as fm
# 设置字体为 Times New Roman
plt.rcParams.update({'font.size': 16})  # 设置全局字体大小
font_path = fm.findfont(fm.FontProperties(family="Times New Roman"))
font_name = fm.FontProperties(fname=font_path).get_name()  # 获取字体名称
rcParams['font.family'] = font_name
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = font_name
rcParams['mathtext.it'] = font_name
rcParams['mathtext.bf'] = font_name# 数据
lambda_1 = np.array([0.23, 0.24, 0.25, 0.26, 0.27])
lambda_4 = np.array([0.27, 0.275, 0.28, 0.285, 0.29])
lambda_1, lambda_4 = np.meshgrid(lambda_1, lambda_4)
acc2 = np.array([
    [88.56, 88.92, 89.02, 88.72, 87.65],
    [88.36, 88.78, 89.12, 87.95, 88.94],
    [87.65, 88.65, 89.04, 88.24, 87.84],
    [88.26, 88.41, 89.63, 88.71, 88.62],
    [88.26, 88.95, 87.95, 88.36, 87.58]
])  # 示例数据，可以替换为实际的Acc2数据

# 自定义颜色，每个柱子对应一种颜色
colors = [
    'yellow','cyan','dodgerblue','blue', 'navy',
    'yellow', 'cyan','dodgerblue','blue', 'navy',
    'yellow', 'cyan', 'dodgerblue', 'blue', 'navy',
    'yellow', 'cyan',  'dodgerblue','blue', 'navy',
    'yellow', 'cyan', 'dodgerblue', 'blue', 'navy',
]

# 创建3D图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制3D柱状图
color_idx = 0
z_base = 87  # 设置底部起点为 zlim 的最小值
for i in range(lambda_1.shape[0]):
    for j in range(lambda_1.shape[1]):
        y_offset = -0.001  # 小的y偏移量
        x_offset = 0.0001  # 小的x偏移量
        ax.bar3d(
            lambda_4[i, j]+x_offset,      # x坐标
            lambda_1[i, j]+y_offset,      # y坐标
            z_base,                   # 底部位置
            0.004,             # x宽度
            0.008,              # y宽度
            acc2[i, j]-z_base,          # 高度
            color=colors[color_idx],  # 使用自定义的颜色
            alpha=0.8, #設置透明度

        )
        color_idx += 1

# 手动设置 x 和 y 轴刻度
ax.set_xticks([0.27, 0.275, 0.28, 0.285, 0.29])  # x轴刻度
ax.set_yticks([0.23, 0.24, 0.25, 0.26, 0.27])   # y轴刻度
# 设置 z 轴范围和刻度
ax.set_zlim(87, 90)  # 设置 z 轴范围
ax.zaxis.set_major_locator(MultipleLocator(0.4))  # 每隔 0.01 设置一个刻度
# 调整视角
ax.view_init(elev=25, azim=310)#220
# 设置轴标签
ax.set_xlabel(r'$\eta_{sv}$',fontsize=18)
ax.set_ylabel(r'$\eta_{sa}$',fontsize=18)
ax.set_zlabel('Acc-2',fontsize=18)

plt.show()
