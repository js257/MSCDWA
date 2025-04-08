# import torch
import torch.nn as nn
from config.global_configs import *
from torch.nn import Dropout, Linear, LayerNorm
from .WeightUpdate import WCWUM
from .transformer import TransformerEncoder
#####################################################
from .FRM import VFRM,AFRM
import joblib



class CmECM(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, layers, attn_dropout, relu_dropout, res_dropout, embed_dropout):
        super(CmECM, self).__init__()
        self.attention_norm = LayerNorm(input_dim, eps=1e-6)
        self.transformer = TextEnhancedTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            layers=layers,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
            embed_dropout=embed_dropout
        )

    def forward(self, text_input, modal_input):
        modal_input = modal_input.permute(1, 0, 2)  # [seq_len, batch_size, input_dim]
        h = modal_input
        modal_input = self.attention_norm(modal_input)
        attended_output = self.transformer(text_input.permute(1, 0, 2), modal_input, modal_input)
        return (attended_output + h).permute(1, 0, 2)

class TextEnhancedTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, attn_dropout, relu_dropout, res_dropout, embed_dropout) -> None:
        super().__init__()

        self.lower_mha = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            layers=1,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
            embed_dropout=embed_dropout,
            position_embedding=True,
            attn_mask=True
        )

        self.upper_mha = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            layers=layers,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
            embed_dropout=embed_dropout,
            position_embedding=True,
            attn_mask=True
        )

    def forward(self, query_m, key_m, text):
        c = self.lower_mha(query_m, text, text)
        return self.upper_mha(key_m, c, c)

class CmRI(nn.Module):
    def __init__(self, visual_dim=27, audio_dim=74, fusion_DIM=512):
        super(CmRI, self).__init__()
        self.visual_dim = visual_dim
        self.fusion_dim = fusion_DIM
        self.audio_dim = audio_dim
        self.hv = TextEnhancedTransformer(
            embed_dim=self.fusion_dim,
            num_heads=2,
            layers=1, attn_dropout=0, relu_dropout=0, res_dropout=0,
            embed_dropout=0)  # 自注意力层用于视觉
        self.ha = TextEnhancedTransformer(
            embed_dim=self.fusion_dim,
            num_heads=2,
            layers=1, attn_dropout=0, relu_dropout=0, res_dropout=0,
            embed_dropout=0)  # 自注意力层用于听觉
        self.cat_connect = nn.Linear(self.fusion_dim * 2, fusion_DIM)
        self.linear_visual = nn.Linear(self.visual_dim, self.fusion_dim)
        self.linear_audio = nn.Linear(self.audio_dim, self.fusion_dim)

        self.attention_norm = LayerNorm(self.fusion_dim * 2, eps=1e-6)
        self.dpper_mha = TransformerEncoder(
            embed_dim=self.fusion_dim * 2,
            num_heads=2,
            layers=2,
            position_embedding=True,
            attn_mask=True
        )

    def forward(self, visual_ids, acoustic_ids):
        # 视觉和听觉输入应为长整型索引
        visual_ids = self.linear_visual(visual_ids)
        acoustic_ids = self.linear_audio(acoustic_ids)
        visual_ = self.hv(acoustic_ids, visual_ids, visual_ids)
        acoustic_ = self.ha(visual_ids, acoustic_ids, acoustic_ids)
        visual_acoustic = torch.cat([visual_, acoustic_], dim=-1)
        BA = visual_acoustic
        visual_acoustic = self.attention_norm(visual_acoustic)
        visual_acoustic = self.dpper_mha(visual_acoustic) + BA
        shift = self.cat_connect(visual_acoustic)

        return shift

class FeatureFusion(nn.Module):
    def __init__(self, text_dim, visual_dim, audio_dim, fusion_dim):
        super(FeatureFusion, self).__init__()
        self.fusion_linear_text = nn.Linear(text_dim, fusion_dim)
        self.fusion_linear_visual = nn.Linear(visual_dim, fusion_dim)
        self.fusion_linear_audio = nn.Linear(audio_dim, fusion_dim)

    def forward(self, text, visual, audio):
        fused_text = self.fusion_linear_text(text)
        fused_visual = self.fusion_linear_visual(visual)
        fused_audio = self.fusion_linear_audio(audio)
        return fused_text, fused_visual, fused_audio

class ModalityCooperativeWeightUpdate(nn.Module):
    def __init__(self):
        super(ModalityCooperativeWeightUpdate, self).__init__()

    def forward(self, fused_text, fused_visual, fused_audio):
        t_v_r, t_a_r = WCWUM(fused_text, fused_visual, fused_audio)
        fus_visual = t_v_r * fused_visual
        fus_audio = t_a_r * fused_audio
        return fus_visual, fus_audio


class MSCDWA(nn.Module):
    def __init__(self, text_dim=TEXT_DIM, visual_dim=VISUAL_DIM, audio_dim=ACOUSTIC_DIM, fusion_dim=512):
        super(MSCDWA, self).__init__()
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.fusion_dim = fusion_dim

        # 初始化各个模块
        self.pca_attention_mlp_audio = AFRM(
            audio_dim=audio_dim,
            reduced_dim=40,  # MOSI: 40, MOSEI: 32
            pca_components=joblib.load('/media/cjs/a4208921-d8a3-41bd-9da2-9e7319a43785/cjs-first-paper/cjs-model-xiu-MOSI/pca_pkl/MOSI_a-40-pca.pkl')
        )
        self.pca_attention_mlp_visual = VFRM(
            visual_dim=visual_dim,
            reduced_dim=20,  # MOSI: 20, MOSEI: 27
            pca_components=joblib.load('/media/cjs/a4208921-d8a3-41bd-9da2-9e7319a43785/cjs-first-paper/cjs-model-xiu-MOSI/pca_pkl/MOSI_v-20-pca.pkl')
        )
        self.CmECM_v = CmECM(
            input_dim=visual_dim,
            embed_dim=visual_dim,
            num_heads=1,
            layers=6,
            attn_dropout=0,
            relu_dropout=0,
            res_dropout=0.1,
            embed_dropout=0
        )
        self.CmECM_a = CmECM(
            input_dim=audio_dim,
            embed_dim=audio_dim,
            num_heads=2,
            layers=6,
            attn_dropout=0,
            relu_dropout=0,
            res_dropout=0.1,
            embed_dropout=0
        )
        self.feature_fusion = FeatureFusion(
            text_dim=text_dim,
            visual_dim=visual_dim,
            audio_dim=audio_dim,
            fusion_dim=fusion_dim
        )
        self.modality_cooperative_weight_update = ModalityCooperativeWeightUpdate()

        # CmRI模型
        self.CmRI_v_a = CmRI(self.visual_dim, self.audio_dim, self.fusion_dim)
        self.CmRI_t_a = CmRI(self.text_dim, self.audio_dim, self.fusion_dim)

        # 线性层用于匹配维度
        self.linear_text_to_visual = nn.Linear(self.text_dim, self.visual_dim)
        self.linear_text_to_audio = nn.Linear(self.text_dim, self.audio_dim)
    def forward(self, text_embedding, visual=None, acoustic=None):
        # 视觉与音频特征冗余处理
        acoustic = self.pca_attention_mlp_audio(acoustic)
        visual = self.pca_attention_mlp_visual(visual)

        # 提取特定相关性特征
        relate_model_v_a = self.CmRI_v_a(visual, acoustic)
        relate_model_t_a = self.CmRI_t_a(text_embedding, acoustic)

        # 文本到视觉、音频的线性投影
        projected_text_to_visual = self.linear_text_to_visual(text_embedding)
        projected_text_to_audio = self.linear_text_to_audio(text_embedding)

        # 视觉与音频增强
        attended_visual = self.CmECM_v(projected_text_to_visual, visual)
        attended_audio = self.CmECM_a(projected_text_to_audio, acoustic)

        # 模态维度统一
        fused_text, fused_visual, fused_audio = self.feature_fusion(text_embedding, attended_visual, attended_audio)

        # 模态协同权重更新
        if self.training:
            fus_visual, fus_audio = self.modality_cooperative_weight_update(fused_text, fused_visual, fused_audio)
        else:
            fus_visual = fused_visual
            fus_audio = fused_audio

        # 最终输出
        fused_output = fused_text + fus_visual + fus_audio + relate_model_v_a + relate_model_t_a
        return fused_output, fused_text, fus_visual, relate_model_v_a, relate_model_t_a
