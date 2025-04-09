#两个三维特征向量
import numpy as np
import torch
from torch.nn.functional import kl_div
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})  # 设置全局字体大小
def generate_js_vectors(features):
    """将特征归一化为概率分布。"""
    return features / (features.sum(axis=-1, keepdims=True) + 1e-10)

def js_divergence(p, q, epsilon=1e-10):
    """计算两个概率分布之间的JS散度。"""
    # 避免数值问题的归一化
    p = (p + epsilon) / (p.sum() + epsilon)
    q = (q + epsilon) / (q.sum() + epsilon)

    # 计算中间分布
    m = 0.5 * (p + q)

    # 使用kl_div计算KL散度
    kl_pm = kl_div(p.log(), m, reduction='batchmean')  # KL(p || m)
    kl_qm = kl_div(q.log(), m, reduction='batchmean')  # KL(q || m)

    # 返回JS散度
    return 0.5 * (kl_pm + kl_qm)

# 加载数据
data = np.load('E:/陈俊松/博士阶段/cjs/备份代码/第一篇代码/JS散度可視化比較/MOSI+CMD-va-ta-1.npz', allow_pickle=True)
text_features = data['text_output']  # 形状: [总批次数, num_steps, num_features_text]
visual_features = data['visual_output']  # 形状: [总批次数, num_steps, num_features_visual]
audio_features = data['audio_output']  # 形状: [总批次数, num_steps, num_features_audio]



def dynamic_weight_update(text_features , visual_features,audio_features):
    # 初始化存储结果的数组
    num_batches, num_steps, _ = text_features[0].shape
    t_v_js_mean = np.zeros(len(text_features))
    t_a_js_mean = np.zeros(len(text_features))
    for idx in range(len(text_features)):
        text_out = generate_js_vectors(torch.tensor(text_features[idx], dtype=torch.float32))
        visual_out = generate_js_vectors(torch.tensor(visual_features[idx], dtype=torch.float32))
        audio_out = generate_js_vectors(torch.tensor(audio_features[idx], dtype=torch.float32))
        min_value_t = text_out.min().item()
        min_value_v =visual_out.min().item()
        min_value_a = audio_out.min().item()

        # 如果最小值小于0，加上绝对值
        if min_value_t < 0:
            text_out += abs(min_value_t)
        if min_value_v < 0:
            visual_out += abs(min_value_v)
        if min_value_a < 0:
            audio_out += abs(min_value_a)

        jsv_weights = torch.zeros(num_steps)
        jsa_weights = torch.zeros(num_steps)

        for t in range(num_steps):
            # 对当前时间步的所有batch求平均
            reference_t = text_out[:, t, :].mean(dim=0)
            visual_v = visual_out[:, t, :].mean(dim=0)
            audio_a = audio_out[:, t, :].mean(dim=0)
            # print(reference_t)
            # print(visual_v)
            # 计算JS散度
            jsv = js_divergence(visual_v, reference_t)
            jsa = js_divergence(audio_a, reference_t)

            jsv = jsv * 1e9
            jsa = jsa * 1e9
            # 处理异常情况
            jsv = 1 if np.isinf(jsv) else jsv
            jsv = 0 if np.isnan(jsv) else jsv
            jsa = 1 if np.isinf(jsa) else jsa
            jsa = 0 if np.isnan(jsa)  else jsa

            # 记录权重
            jsv_weights[t] = jsv  # 权重与JS散度成反比
            jsa_weights[t] = jsa

        # 计算每个batch的平均JS散度权重
        t_v_js_mean[idx] = jsv_weights.mean().item()
        # print(t_v_js_mean[idx])
        t_a_js_mean[idx] = jsa_weights.mean().item()
        # print(t_v_js_mean[idx])
        # print(t_a_js_mean[idx])
    return t_v_js_mean,t_a_js_mean
# 可视化平均JS散度
t_v_js_mean,t_a_js_mean = dynamic_weight_update(text_features , visual_features, audio_features)
# 设置纵坐标范围
y_min = 0  # 你可以调整这些值
y_max = 4  # 你可以调整这些值
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # 设置字体
plt.plot(t_v_js_mean, label="Text vs visual", color='r')
plt.plot(t_a_js_mean, label="Text vs audio", color='y')
plt.xlabel('Iteration Length')
plt.ylabel('Average JS Divergence')
# plt.title('Average JS Divergence Per Iteration')

# 调整纵坐标范围
plt.ylim(y_min, y_max)

# 使用plt.legend()添加图例
plt.legend()
plt.grid()
plt.savefig('JS Divergence.png', dpi=600)  # 可以设置文件格式和分辨率
plt.show()
