#两个三维特征向量
import numpy as np
import torch
from torch.nn.functional import kl_div
import matplotlib.pyplot as plt
def generate_js_vectors(features):
    """将特征归一化为概率分布。"""
    return features / (features.sum(axis=-1, keepdims=True) + 1e-10)

def js_divergence(p, q, epsilon=1e-10):
    """计算JS散度，确保结果非负（CPU版）。"""
    # 确保输入在CPU上
    p = p.cpu()
    q = q.cpu()

    # 归一化分布并避免零值
    p = torch.clamp(p + epsilon, min=epsilon) / (p.sum() + epsilon)
    q = torch.clamp(q + epsilon, min=epsilon) / (q.sum() + epsilon)
    m = 0.5 * (p + q)

    # 计算KL散度
    kl_pm = torch.nn.functional.kl_div(p.log(), m, reduction='batchmean')
    kl_qm = torch.nn.functional.kl_div(q.log(), m, reduction='batchmean')

    # 计算JS散度并确保非负
    jsd = 0.5 * (kl_pm + kl_qm)
    jsd = torch.clamp(jsd, min=0.0)  # 避免负值
    return jsd


def WCWUM(text_features , visual_features, audio_features):
    # 初始化存储结果的数组
    num_batches, seq_len, _ = text_features.shape
    # alpha = max(0.08, 0.13 * np.exp(-cg.epoch_flag / 100))
    alpha = 0.13  #0.13最好
    epsilon = 1e-6
    weights_v = torch.zeros(seq_len)
    weights_a = torch.zeros(seq_len)
    # for idx in range(len(text_features)):
    text_out = generate_js_vectors(text_features.clone().detach().cpu().float())
    visual_out = generate_js_vectors(visual_features.clone().detach().cpu().float())
    audio_out = generate_js_vectors(audio_features.clone().detach().cpu().float())

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

    jsv_weights = torch.zeros(seq_len)
    jsa_weights = torch.zeros(seq_len)

    for t in range(seq_len):
        # 对当前时间步的所有batch求平均
        reference_t = text_out[:, t, :].mean(dim=0)
        visual_v = visual_out[:, t, :].mean(dim=0)
        audio_a = audio_out[:, t, :].mean(dim=0)

        # 计算JS散度

        jsv = js_divergence(visual_v, reference_t)
        jsa = js_divergence(audio_a, reference_t)
        jsv = jsv * 1e9
        jsa = jsa * 1e9

        # 处理异常情况
        jsv = 1 if np.isinf(jsv) else jsv
        jsv = 0 if np.isnan(jsv) else jsv
        jsa = 1 if np.isinf(jsa) else jsa
        jsa = 0 if np.isnan(jsa) else jsa

        # print(jsv)
        # print(jsa)
        jsv_weights[t] = jsv  # 放大散度值  # 权重与JS散度成反比
        jsa_weights[t] = jsa  # 放大散度值
        # 记录权重
        if t == 0:
            weights_v[t] = 1 / (1 + jsv + epsilon)
            weights_a[t] = 1 / (1 + jsa + epsilon)
        else:
            weights_v[t] = alpha * weights_v[t - 1] + (1 - alpha) * (1 / (1 + jsv + epsilon))
            weights_a[t] = alpha * weights_a[t - 1] + (1 - alpha) * (1 / (1 + jsa + epsilon))

    # 动态阈值
    # sv_threshold = jsv_weights.mean() + jsv_weights.std()
    # sa_threshold = jsa_weights.mean() + jsa_weights.std()
    sv_threshold = 0.285
    sa_threshold = 0.25
    # print(sv_threshold)
    # print(sa_threshold)
    probability_v = weights_v.mean() if jsv_weights.mean() >= sv_threshold else 1.0
    probability_a = weights_a.mean() if jsa_weights.mean() >= sa_threshold else 1.0

    return probability_v, probability_a
# 可视化平均JS散度
# t_v_js_mean,t_a_js_mean = dynamic_weight_update(text_features , visual_features,audio_features)
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']  #设置字体
# plt.plot(t_v_js_mean, label="Variance of Text vs visual JS Divergence", color='r')
# plt.plot(t_a_js_mean, label="Variance of Text vs audio JS Divergence", color='y')
# plt.xlabel('Iteration Length')
# plt.ylabel('Average JS Divergence')
# plt.title('Average JS Divergence Per Iteration')
# # 使用plt.legend()添加图例
# plt.legend()
# plt.grid()
# plt.savefig('JS Divergence.png', dpi=300)  # 可以设置文件格式和分辨率
# plt.show()
