import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.head_dim = hid_dim // num_heads

        assert (
            self.head_dim * num_heads == hid_dim
        ), "Hidden dimension must be divisible by number of heads"

        self.values = nn.Linear(hid_dim, hid_dim, bias=False)
        self.keys = nn.Linear(hid_dim, hid_dim, bias=False)
        self.queries = nn.Linear(hid_dim, hid_dim, bias=False)
        self.fc_out = nn.Linear(hid_dim, hid_dim)

    def forward(self, x):
        N, seq_length, _ = x.shape  # N: batch size, seq_length: sequence length

        # Split embedding into multiple heads
        values = self.values(x).view(N, seq_length, self.num_heads, self.head_dim)
        keys = self.keys(x).view(N, seq_length, self.num_heads, self.head_dim)
        queries = self.queries(x).view(N, seq_length, self.num_heads, self.head_dim)

        # Transpose to get dimensions (N, num_heads, seq_length, head_dim)
        values = values.transpose(1, 2)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # Calculate energy scores
        energy = torch.einsum("nqhd,nkhd->nqk", [queries, keys])  # (N, num_heads, seq_length, seq_length)
        attention = F.softmax(energy / (self.head_dim ** (1 / 2)), dim=2)

        # Weighted values
        out = torch.einsum("nqk,nkhd->nqhd", [attention, values]).reshape(N, seq_length, self.hid_dim)

        return self.fc_out(out)

class GRUWithGraphAttention(nn.Module):
    def __init__(self, hid_dim, num_heads, num_layers, dropout):
        super(GRUWithGraphAttention, self).__init__()
        self.gru = nn.GRU(input_size=hid_dim, hidden_size=hid_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.attention = MultiHeadAttention(hid_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 4),
            nn.ReLU(),
            nn.Linear(hid_dim * 4, hid_dim)
        )
        self.layer_norm = nn.LayerNorm(hid_dim,eps=1e-6)
        # self.conv = nn.Conv1d(in_channels=hid_dim, out_channels=hid_dim, kernel_size=1)  # 一维卷积

    def forward(self, x):
        # GRU处理输入
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hid_dim)
        # 通过多头自注意力
        attention_out = self.attention(gru_out)  # (batch_size, seq_len, hid_dim)

        # 残差连接和层归一化
        out = self.layer_norm(gru_out + attention_out)
        # 前馈网络处理
        ff_out = self.feed_forward(out)

        out = self.layer_norm(out + ff_out)  # 残差连接和层归一化
        return out # 返回最终输出


class AFRM(nn.Module):
    def __init__(self, audio_dim, reduced_dim, pca_components=None):
        super(AFRM, self).__init__()
        self.audio_dim = audio_dim
        self.reduced_dim = reduced_dim
        self.pca = pca_components
        # 用于将MLP输出映射回原始输入的维度
        self.expand_fc = nn.Linear(self.reduced_dim, self.audio_dim)
        self.audio_model = GRUWithGraphAttention(self.reduced_dim,8,2,0)   # num_heads=8

    def apply_pca(self, x):
        # 将PyTorch张量转换为NumPy数组，形状：[batch_size * seq_len, visual_dim]
        x_np = x.detach().cpu().numpy().reshape(-1, self.audio_dim)  # 将数据转到CPU并转换为NumPy

        # 对输入进行PCA降维，保留前reduced_dim个主成分
        x_pca = self.pca.transform(x_np)  # [batch_size * seq_len, reduced_dim]

        # 将降维后的数据转换回PyTorch张量并重新reshape，移动到GPU
        x_pca_torch = torch.tensor(x_pca, dtype=torch.float32, device=x.device)
        x_pca_torch = x_pca_torch.reshape(x.shape[0], x.shape[1],
                                          self.reduced_dim)  # [batch_size, seq_len, reduced_dim]
        return x_pca_torch

    def forward(self, x):
        # Step 1: 先对输入特征进行PCA降维
        pca_output = self.apply_pca(x)  # [batch_size, seq_len, reduced_dim]
        pca_output = self.audio_model(pca_output) #torch.Size([64, 1])
        mlp_output_expanded = self.expand_fc(pca_output)  # [batch_size, seq_len, visual_dim]
        # print(pca_output.shape)
        return mlp_output_expanded


class VFRM(nn.Module):
    def __init__(self, visual_dim, reduced_dim,  pca_components=None):
        super(VFRM, self).__init__()
        self.visual_dim = visual_dim
        self.reduced_dim = reduced_dim

        # 使用预先拟合的 PCA 模型
        self.pca = pca_components
        # 组成成分分析
        self.component_fc = nn.Linear(reduced_dim, reduced_dim, bias=False)
        # 加性注意力机制的权重
        self.attention_fc = nn.Linear(reduced_dim, 1, bias=False)
        self.expand_fc = nn.Linear(self.reduced_dim, visual_dim)

    def apply_pca(self, x):
        # 将PyTorch张量转换为NumPy数组，形状：[batch_size * seq_len, visual_dim]
        x_np = x.detach().cpu().numpy().reshape(-1, self.visual_dim)  # 将数据转到CPU并转换为NumPy

        # 对输入进行PCA降维，保留前reduced_dim个主成分
        x_pca = self.pca.transform(x_np)  # [batch_size * seq_len, reduced_dim]

        # 将降维后的数据转换回PyTorch张量并重新reshape，移动到GPU
        x_pca_torch = torch.tensor(x_pca, dtype=torch.float32, device=x.device)
        x_pca_torch = x_pca_torch.reshape(x.shape[0], x.shape[1],
                                          self.reduced_dim)  # [batch_size, seq_len, reduced_dim]

        return x_pca_torch

    def forward(self, x):
        # Step 1: 先对输入特征进行PCA降维
        pca_output = self.apply_pca(x)  # [batch_size, seq_len, reduced_dim]

        # Step 2: 组成成分分析：对每个时间步的每个维度进行线性变换
        component_output = torch.tanh(self.component_fc(pca_output))  # 非线性激活函数
        # component_output = self.mlp(component_output)

        # Step 3: 计算注意力分数，形状为 [batch_size, seq_len, 1]
        attention_scores = self.attention_fc(component_output)

        # Step 4: 计算注意力权重，形状为 [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)
        atent_out = attention_weights * pca_output+pca_output

        # Step 5: 对原始输入特征进行加权，形状为 [batch_size, seq_len, visual_dim]
        # Step 6: 使用MLP处理PCA降维后的特征，保持序列长度不变
        # mlp_output = self.mlp(pca_output)  # [batch_size, seq_len, reduced_dim]
        # print(mlp_output.shape)
        # Step 7: 将MLP的输出映射回原始的visual_dim维度
        output = self.expand_fc(atent_out)  # [batch_size, seq_len, visual_dim]

        return  output

