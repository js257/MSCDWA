from __future__ import absolute_import, division, print_function

import argparse
from pytorch_transformers.modeling_roberta import RobertaConfig
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from torch.nn import L1Loss, MSELoss
from pytorch_transformers import WarmupLinearSchedule, AdamW

from utils.databuilder import set_up_data_loader
from utils.set_seed import set_random_seed, seed
from utils.metric import score_model
import config.global_configs as cg
import os
from networks.subnet.all_loss import diss_loss
from networks.SentiLARE import RobertaForSequenceClassification


# ============================================================================
# 模型分析工具类
# ============================================================================
class ModelAnalyzer:
    """用于统计模型参数量、FLOPs、训练时间和推理速度"""

    def __init__(self, model_name="MSCDWA"):
        self.model_name = model_name
        self.train_start_time = None
        self.train_end_time = None
        self.epoch_times = []
        self.inference_times = []
        self.inference_samples = []

    def count_parameters(self, model, new_module_keywords=None):
        """
        统计模型参数量

        Args:
            model: 模型实例
            new_module_keywords: 新增模块的关键词列表，如 ['MSCDWA']
                              用于区分预训练参数和新增参数
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        # 按模块统计
        module_params = {}
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                module_params[name] = sum(p.numel() for p in module.parameters())

        result = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params,
            "total_params_M": total_params / 1e6,
            "trainable_params_M": trainable_params / 1e6,
            "module_params": module_params
        }

        # 统计新增模块参数量（如 MSCDWA）
        if new_module_keywords:
            new_module_params = 0
            new_module_details = {}
            for name, param in model.named_parameters():
                if any(kw in name for kw in new_module_keywords):
                    new_module_params += param.numel()
                    new_module_details[name] = param.numel()

            result["new_module_params"] = new_module_params
            result["new_module_params_M"] = new_module_params / 1e6
            result["new_module_details"] = new_module_details

            # 预训练模型参数量 = 总参数量 - 新增模块参数量
            pretrained_params = total_params - new_module_params
            result["pretrained_params"] = pretrained_params
            result["pretrained_params_M"] = pretrained_params / 1e6

        return result

    def estimate_flops(self, model, input_shapes, new_module_keywords=None):
        """
        估算模型FLOPs（基于前向传播的近似计算）

        Args:
            model: 模型实例
            input_shapes: dict with keys like 'input_ids', 'visual', 'acoustic', etc.
            new_module_keywords: 新增模块的关键词列表，如 ['MSCDWA']
                               只统计这些模块的FLOPs，不传入则统计全部
        """
        total_flops = 0
        new_module_flops = 0

        # 记录每层FLOPs
        flops_details = {}
        new_module_flops_details = {}

        def hook_fn(module, input, output, name):
            nonlocal total_flops, new_module_flops
            module_flops = 0

            if isinstance(module, nn.Linear):
                # Linear: 2 * input_features * output_features * batch_size
                batch_size = input[0].shape[0] if len(input[0].shape) > 1 else 1
                in_features = module.in_features
                out_features = module.out_features
                module_flops = 2 * batch_size * in_features * out_features

            elif isinstance(module, nn.MultiheadAttention):
                # Attention: 4 * seq_len^2 * d_model + 2 * seq_len * d_model^2 (per head, approximate)
                if len(input) >= 2:
                    batch_size = input[0].shape[0] if len(input[0].shape) > 1 else 1
                    seq_len = input[0].shape[-2] if len(input[0].shape) > 1 else input[0].shape[0]
                    d_model = module.embed_dim
                    num_heads = module.num_heads
                    module_flops = 2 * batch_size * num_heads * seq_len * seq_len * (d_model // num_heads)
                    module_flops += 2 * batch_size * seq_len * d_model * d_model

            elif isinstance(module, nn.Conv1d):
                # Conv1d: 2 * batch * out_channels * out_length * kernel_size * in_channels
                batch_size = input[0].shape[0]
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                # Approximate output length
                out_length = input[0].shape[-1] // kernel_size if input[0].shape[-1] > kernel_size else input[0].shape[-1]
                module_flops = 2 * batch_size * out_channels * out_length * kernel_size * in_channels

            elif isinstance(module, nn.LayerNorm):
                batch_size = input[0].numel() // input[0].shape[-1]
                features = input[0].shape[-1]
                module_flops = 5 * batch_size * features  # mean, var, sub, div, mul, add

            elif isinstance(module, nn.Embedding):
                # Embedding lookup is memory-bound, approximate as 0 FLOPs or minimal
                module_flops = 0

            total_flops += module_flops
            if module_flops > 0:
                flops_details[name] = module_flops

            # 判断是否为新增模块的FLOPs
            if new_module_keywords and any(kw in name for kw in new_module_keywords):
                new_module_flops += module_flops
                if module_flops > 0:
                    new_module_flops_details[name] = module_flops

        hooks = []
        for name, module in model.named_modules():
            if any(isinstance(module, t) for t in [nn.Linear, nn.MultiheadAttention, nn.Conv1d, nn.LayerNorm]):
                hook = module.register_forward_hook(
                    lambda mod, inp, out, n=name: hook_fn(mod, inp, out, n)
                )
                hooks.append(hook)

        # 执行一次前向传播来统计FLOPs
        model.eval()
        with torch.no_grad():
            try:
                # 创建dummy inputs based on input_shapes
                dummy_inputs = {}
                for key, shape in input_shapes.items():
                    if key in ['input_ids', 'visual_ids', 'acoustic_ids', 'pos_ids', 'senti_ids', 'polarity_ids']:
                        dummy_inputs[key] = torch.randint(0, 100, shape).to(cg.DEVICE)
                    elif key == 'input_mask':
                        dummy_inputs[key] = torch.ones(shape, dtype=torch.long).to(cg.DEVICE)
                    elif key == 'segment_ids':
                        dummy_inputs[key] = torch.zeros(shape, dtype=torch.long).to(cg.DEVICE)
                    elif key in ['visual', 'acoustic']:
                        dummy_inputs[key] = torch.randn(shape).to(cg.DEVICE)
                    else:
                        dummy_inputs[key] = torch.randn(shape).to(cg.DEVICE)

                # 调用模型
                if hasattr(model, 'roberta'):
                    _ = model(
                        dummy_inputs.get('input_ids'),
                        dummy_inputs.get('visual'),
                        dummy_inputs.get('acoustic'),
                        dummy_inputs.get('visual_ids'),
                        dummy_inputs.get('acoustic_ids'),
                        dummy_inputs.get('pos_ids'),
                        dummy_inputs.get('senti_ids'),
                        dummy_inputs.get('polarity_ids'),
                        attention_mask=dummy_inputs.get('input_mask'),
                        token_type_ids=dummy_inputs.get('segment_ids'),
                    )
            except Exception as e:
                print(f"FLOPs estimation warning: {e}")
                # Fallback: rough estimate based on parameter count
                total_params = sum(p.numel() for p in model.parameters())
                total_flops = total_params * 2  # Very rough estimate
                if new_module_keywords:
                    new_module_flops = sum(
                        p.numel() for n, p in model.named_parameters()
                        if any(kw in n for kw in new_module_keywords)
                    ) * 2

        # Remove hooks
        for hook in hooks:
            hook.remove()

        result = {
            "total_flops": total_flops,
            "total_flops_G": total_flops / 1e9,
            "total_flops_M": total_flops / 1e6,
            "flops_details": flops_details
        }

        # 新增模块FLOPs
        if new_module_keywords:
            result["new_module_flops"] = new_module_flops
            result["new_module_flops_G"] = new_module_flops / 1e9
            result["new_module_flops_M"] = new_module_flops / 1e6
            result["new_module_flops_details"] = new_module_flops_details

            pretrained_flops = total_flops - new_module_flops
            result["pretrained_flops"] = pretrained_flops
            result["pretrained_flops_G"] = pretrained_flops / 1e9
            result["pretrained_flops_M"] = pretrained_flops / 1e6

        return result

    def start_train_timer(self):
        """开始训练计时"""
        self.train_start_time = time.time()

    def end_train_timer(self):
        """结束训练计时"""
        self.train_end_time = time.time()

    def record_epoch_time(self, epoch_time):
        """记录每个epoch的时间"""
        self.epoch_times.append(epoch_time)

    def record_inference(self, batch_size, num_batches, total_time):
        """记录推理速度"""
        self.inference_times.append(total_time)
        self.inference_samples.append(batch_size * num_batches)

    def get_train_summary(self):
        """获取训练时间总结"""
        if self.train_start_time and self.train_end_time:
            total_train_time = self.train_end_time - self.train_start_time
        else:
            total_train_time = sum(self.epoch_times) if self.epoch_times else 0

        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0

        return {
            "total_train_time_sec": total_train_time,
            "total_train_time_min": total_train_time / 60,
            "avg_epoch_time_sec": avg_epoch_time,
            "num_epochs_recorded": len(self.epoch_times)
        }

    def get_inference_summary(self):
        """获取推理速度总结"""
        if not self.inference_times:
            return {}

        total_samples = sum(self.inference_samples)
        total_time = sum(self.inference_times)
        avg_time_per_sample = total_time / total_samples if total_samples > 0 else 0

        return {
            "total_inference_time_sec": total_time,
            "total_samples": total_samples,
            "avg_time_per_sample_sec": avg_time_per_sample,
            "samples_per_sec": total_samples / total_time if total_time > 0 else 0,
            "avg_batch_inference_time_ms": (sum(self.inference_times) / len(self.inference_times)) * 1000 / max(sum(self.inference_samples) / len(self.inference_samples), 1)
        }

    def print_report(self, model, input_shapes=None, new_module_keywords=None):
        """
        打印完整分析报告

        Args:
            model: 模型实例
            input_shapes: 输入形状字典，用于FLOPs估算
            new_module_keywords: 新增模块关键词，如 ['MSCDWA']
        """
        print("\n" + "="*70)
        print(f"           MODEL ANALYSIS REPORT: {self.model_name}")
        print("="*70)

        # 参数量统计
        params = self.count_parameters(model, new_module_keywords)
        print(f"\n[1] PARAMETER SIZE:")
        print(f"    Total Parameters:      {params['total_params']:,} ({params['total_params_M']:.2f} M)")
        print(f"    Trainable Parameters:  {params['trainable_params']:,} ({params['trainable_params_M']:.2f} M)")
        print(f"    Non-trainable Params:  {params['non_trainable_params']:,}")

        # 模块参数量
        if new_module_keywords and 'new_module_params' in params:
            print(f"\n    --- NEW MODULE ({'/'.join(new_module_keywords)}) ---")
            print(f"    New Module Params:     {params['new_module_params']:,} ({params['new_module_params_M']:.2f} M)")
            print(f"    Pretrained Params:     {params['pretrained_params']:,} ({params['pretrained_params_M']:.2f} M)")
            print(f"    New/Total Ratio:       {params['new_module_params']/params['total_params']*100:.2f}%")

        # FLOPs统计
        if input_shapes:
            flops = self.estimate_flops(model, input_shapes, new_module_keywords)
            print(f"\n[2] FLOPs (Approximate):")
            print(f"    Total FLOPs:           {flops['total_flops']:,} ({flops['total_flops_G']:.2f} G / {flops['total_flops_M']:.2f} M)")

            if new_module_keywords and 'new_module_flops' in flops:
                print(f"\n    --- NEW MODULE ({'/'.join(new_module_keywords)}) FLOPs ---")
                print(f"    New Module FLOPs:      {flops['new_module_flops']:,} ({flops['new_module_flops_G']:.2f} G / {flops['new_module_flops_M']:.2f} M)")
                print(f"    Pretrained FLOPs:      {flops['pretrained_flops']:,} ({flops['pretrained_flops_G']:.2f} G / {flops['pretrained_flops_M']:.2f} M)")
                print(f"    New/Total FLOPs Ratio: {flops['new_module_flops']/flops['total_flops']*100:.2f}%")

        # 训练时间
        train_summary = self.get_train_summary()
        print(f"\n[3] TRAINING TIME:")
        print(f"    Total Training Time:   {train_summary['total_train_time_sec']:.2f} sec ({train_summary['total_train_time_min']:.2f} min)")
        print(f"    Avg Epoch Time:        {train_summary['avg_epoch_time_sec']:.2f} sec")
        print(f"    Epochs Recorded:       {train_summary['num_epochs_recorded']}")

        # 推理速度
        inf_summary = self.get_inference_summary()
        if inf_summary:
            print(f"\n[4] FPS:")
            print(f"    Avg Time/Sample:       {inf_summary['avg_time_per_sample_sec']*1000:.2f} ms")
            print(f"    Throughput:            {inf_summary['samples_per_sec']:.2f} samples/sec")
            print(f"    Total Samples Tested:  {inf_summary['total_samples']}")

        print("\n" + "="*70 + "\n")

        # 返回汇总数据以便wandb记录
        summary = {
            "param_total_M": params['total_params_M'],
            "param_trainable_M": params['trainable_params_M'],
            "total_train_time_sec": train_summary['total_train_time_sec'],
            "avg_epoch_time_sec": train_summary['avg_epoch_time_sec'],
        }
        if 'new_module_params_M' in params:
            summary["param_new_module_M"] = params['new_module_params_M']
            summary["param_pretrained_M"] = params['pretrained_params_M']
            summary["new_module_ratio_%"] = params['new_module_params']/params['total_params']*100
        if input_shapes:
            summary["flops_total_G"] = flops['total_flops_G']
            if 'new_module_flops_G' in flops:
                summary["flops_new_module_G"] = flops['new_module_flops_G']
                summary["flops_pretrained_G"] = flops['pretrained_flops_G']
                summary["new_module_flops_ratio_%"] = flops['new_module_flops']/flops['total_flops']*100
        if inf_summary:
            summary["inference_ms_per_sample"] = inf_summary['avg_time_per_sample_sec'] * 1000
            summary["inference_samples_per_sec"] = inf_summary['samples_per_sec']

        return summary


# ============================================================================
# 全局分析器实例
# ============================================================================
model_analyzer = None


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="MSCDWA")
    parser.add_argument("--dataset", type=str,
                        choices=["mosi", "mosei", "sims", "simsv2"], default="mosi")
    parser.add_argument("--data_path", type=str, default='/media/cjs/a4208921-d8a3-41bd-9da2-9e7319a43785/cjs-first-paper/datasets/dataset/MOSI_16_sentilare_unaligned_data.pkl')
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=64) #64
    parser.add_argument("--dev_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--beta_shift", type=float, default=1.0)
    parser.add_argument("--dropout_prob", type=float, default=0.5) #0.5
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument(
        "--model",
        type=str,
        choices=["bert-base-uncased", "xlnet-base-cased", "roberta-base", "bert-base-chinese"],
        default="roberta-base")
    parser.add_argument("--model_name_or_path", default='/media/cjs/a4208921-d8a3-41bd-9da2-9e7319a43785/cjs-first-paper/pretrained-model/sentilare_model', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--learning_rate", type=float, default=6e-5) #6e-5
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--gradient_accumulation_step", type=int, default=1)
    parser.add_argument("--test_step", type=int, default=20)
    parser.add_argument("--test_flag", type=bool, default=False)
    parser.add_argument("--max_grad_norm", type=int, default=2)
    parser.add_argument("--warmup_proportion", type=float, default=0.4)
    parser.add_argument("--seed", type=seed, default=1111, help="integer or 'random'")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    # 是否打印分析报告的参数
    parser.add_argument("--print_model_analysis", type=bool, default=True,
                        help="Whether to print model parameter size, FLOPs, training time and inference speed analysis")
    return parser.parse_args()


def prep_for_training(args, num_train_optimization_steps: int):
    config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=1, finetuning_task='sst')
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config, pos_tag_embedding=True, senti_embedding=True, polarity_embedding=True)
    model.to(cg.DEVICE)

    # ============================================================================
    # 初始化模型分析器并统计参数量
    # ============================================================================
    global model_analyzer
    model_analyzer = ModelAnalyzer(args.model_name)

    # 统计总参数量
    params_info = model_analyzer.count_parameters(model)
    print(f"\n>>> Model initialized. Total parameters: {params_info['total_params']:,} ({params_info['total_params_M']:.2f}M)")
    print(f">>> Trainable parameters: {params_info['trainable_params']:,} ({params_info['trainable_params_M']:.2f}M)")

    # 统计新增模块（MSCDWA）参数量 —— 论文中通常报告这个
    params_info_new = model_analyzer.count_parameters(model, new_module_keywords=['MSCDWA'])
    if 'new_module_params' in params_info_new:
        print(f">>> NEW MODULE (MSCDWA) parameters: {params_info_new['new_module_params']:,} ({params_info_new['new_module_params_M']:.4f}M)")
        print(f">>> Pretrained (Roberta) parameters: {params_info_new['pretrained_params']:,} ({params_info_new['pretrained_params_M']:.2f}M)")
        print(f">>> NEW/TOTAL ratio: {params_info_new['new_module_params']/params_info_new['total_params']*100:.2f}%\n")
    else:
        print()

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    MultiModalAttentionFusion_params = ['MSCDWA']
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if
                not any(nd in n for nd in no_decay) and not any(nd in n for nd in MultiModalAttentionFusion_params)
            ],
            "weight_decay": args.weight_decay,
        },
        {"params": model.roberta.encoder.MSCDWA.parameters(), 'lr': args.learning_rate,
         "weight_decay": args.weight_decay},
        {
            "params": [
                p for n, p in param_optimizer if
                any(nd in n for nd in no_decay) and not any(nd in n for nd in MultiModalAttentionFusion_params)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        t_total=num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def train_epoch(args, model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model.train()
    preds = []
    labels = []
    tr_loss = 0
    nb_tr_steps = 0


    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(cg.DEVICE) for t in batch)
        input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        outputs = model(
            input_ids,
            visual,
            acoustic,
            visual_ids,
            acoustic_ids,
            pos_ids, senti_ids, polarity_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
        )
        logits = outputs[0]
        loss_fct = MSELoss()
        diss_ = diss_loss()
        loss_diss = diss_(outputs[1][1],outputs[1][2],outputs[1][3],outputs[1][4])
        # print(loss_diss)
        loss_mse = loss_fct(logits.view(-1), label_ids.view(-1))

        loss = loss_mse + 0.01*loss_diss
        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        tr_loss += loss.item()
        nb_tr_steps += 1
        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.detach().cpu().numpy()
        logits = np.squeeze(logits).tolist()
        label_ids = np.squeeze(label_ids).tolist()
        preds.extend(logits)
        labels.extend(label_ids)

    preds = np.array(preds)
    labels = np.array(labels)

    return tr_loss / nb_tr_steps, preds, labels



def evaluate_epoch(args,mode:str, model: nn.Module, dataloader: DataLoader):
    model.eval()

    preds = []
    labels = []
    loss = 0
    nb_dev_examples, nb_steps = 0, 0

    # ============================================================================
    # 推理速度计时
    # ============================================================================
    inference_start = time.time()
    total_batches = 0
    batch_size = args.dev_batch_size if mode == "dev" else args.test_batch_size

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(cg.DEVICE) for t in batch)
            input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            outputs = model(
                input_ids,
                visual,
                acoustic,
                visual_ids,
                acoustic_ids,
                pos_ids, senti_ids, polarity_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
            )

            logits = outputs[0]
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step
            loss += loss.item()
            nb_steps += 1
            total_batches += 1
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()
            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()
            preds.extend(logits)
            labels.extend(label_ids)

    # ============================================================================
    # 记录推理时间
    # ============================================================================
    inference_end = time.time()
    inference_time = inference_end - inference_start
    if model_analyzer is not None and mode == "test":
        model_analyzer.record_inference(batch_size, total_batches, inference_time)
        print(f">>> [{mode.upper()}] Inference completed: {total_batches} batches, "
              f"{inference_time:.3f}s total, "
              f"{inference_time/total_batches*1000:.2f}ms/batch, "
              f"{total_batches*batch_size/inference_time:.2f} samples/sec")
    elif model_analyzer is not None:
        print(f">>> [{mode.upper()}] Inference completed: {total_batches} batches, "
              f"{inference_time:.3f}s total")

    preds = np.array(preds)
    labels = np.array(labels)
    return loss / nb_steps, preds, labels


def train(
        args,
        model,
        train_dataloader,
        validation_dataloader,
        test_data_loader,
        optimizer,
        scheduler, ):

    curr_patience = patience = args.patience
    num_trials = 1

    best_valid_loss = float('inf')

    # ============================================================================
    # 开始训练计时
    # ============================================================================
    if model_analyzer is not None:
        model_analyzer.start_train_timer()

    for epoch_i in range(int(args.n_epochs)):
        cg.epoch_flag = epoch_i

        # ============================================================================
        # Epoch计时
        # ============================================================================
        epoch_start = time.time()

        train_loss, train_pre, train_label = train_epoch(args, model, train_dataloader, optimizer, scheduler)
        valid_loss, valid_pre, valid_label = evaluate_epoch(args,"dev", model, validation_dataloader)

        # ============================================================================
        # 记录epoch时间
        # ============================================================================
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        if model_analyzer is not None:
            model_analyzer.record_epoch_time(epoch_time)

        train_acc, train_mae, train_corr, train_f_score = score_model(train_pre, train_label)

        # ============================================================================
        # 结束训练计时
        # ============================================================================
        if model_analyzer is not None:
            model_analyzer.end_train_timer()

        valid_acc, valid_mae, valid_corr, valid_f_score = score_model(valid_pre, valid_label)

        print(
            "epoch:{}, train_loss:{}, train_acc:{}, valid_mae:{},valid_acc:{}, valid_f_score:{}, valid_loss:{}, epoch_time:{:.2f}s".format(
                epoch_i, train_loss, train_acc, valid_mae, valid_acc, valid_f_score, valid_loss, epoch_time
            )
        )

        wandb.log(
            (
                {
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "train_acc": train_acc,
                    "train_corr": train_corr,
                    "valid_acc": valid_acc,
                    "valid_corr": valid_corr,
                    "valid_f_score": valid_f_score,
                    "epoch_time_sec": epoch_time,
                }
            )
        )

        print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            # print("最好valid_f1:",best_valid_f_score)
            print("Found new best model on dev set!")
            if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
            torch.save(model.state_dict(), f'checkpoints/model_{args.model_name}.std')
            torch.save(optimizer.state_dict(), f'checkpoints/optim_{args.model_name}.std')
            curr_patience = patience
        else:
            curr_patience -= 1
            if curr_patience <= -1:
                print("Running out of patience, loading previous best model.")
                num_trials -= 1
                curr_patience = patience
                model.load_state_dict(torch.load(f'checkpoints/model_{args.model_name}.std', weights_only=True))
                optimizer.load_state_dict(torch.load(f'checkpoints/optim_{args.model_name}.std',weights_only=True))
                scheduler.step()
                print(f"Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")

        if num_trials <= 0:
            print("Running out of patience, early stopping.")
            break
         ##################################################################


    test_loss, test_pre, test_label = evaluate_epoch(args,"test",model, test_data_loader)
    test_acc, test_mae, test_corr, test_f_score = score_model(test_pre, test_label)
    non0_test_acc, _, _, non0_test_f_score = score_model(test_pre, test_label, use_zero=True)


    print(
        "test_mae:{}, test_loss:{}, test_acc:{}, test_corr:{}, test_f_score:{}".format(
             test_mae, test_loss, test_acc,test_corr, test_f_score
        )
    )

    wandb.log(
        (
            {
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_mae": test_mae,
                "test_corr": test_corr,
                "test_f_score": test_f_score,
                "non0_test_acc": non0_test_acc,
                "non0_test_f_score": non0_test_f_score,
            }
        )
    )

    # ============================================================================
    # 打印完整分析报告
    # ============================================================================
    if args.print_model_analysis and model_analyzer is not None:
        # 构建input_shapes用于FLOPs估算（基于dev batch size）
        input_shapes = {
            'input_ids': (args.dev_batch_size, args.max_seq_length),
            'visual_ids': (args.dev_batch_size, args.max_seq_length),
            'acoustic_ids': (args.dev_batch_size, args.max_seq_length),
            'pos_ids': (args.dev_batch_size, args.max_seq_length),
            'senti_ids': (args.dev_batch_size, args.max_seq_length),
            'polarity_ids': (args.dev_batch_size, args.max_seq_length),
            'visual': (args.dev_batch_size, 1, 35),  # 根据MOSI数据集常见visual特征维度
            'acoustic': (args.dev_batch_size, 1, 74),  # 根据MOSI数据集常见acoustic特征维度
            'input_mask': (args.dev_batch_size, args.max_seq_length),
            'segment_ids': (args.dev_batch_size, args.max_seq_length),
        }

        analysis_summary = model_analyzer.print_report(model, input_shapes, new_module_keywords=['MSCDWA'])

        # 将分析结果也记录到wandb
        wandb.log(analysis_summary)
        print(f">>> Model analysis metrics also logged to wandb.")



def main():
    args = parser_args()
    wandb.init(project="MSCDWA", reinit=True, save_code=False)

    set_random_seed(args.seed)
    wandb.config.update(args)

    (train_data_loader,
     dev_data_loader,
     test_data_loader,
     num_train_optimization_steps,
     ) = set_up_data_loader(args)

    model, optimizer, scheduler = prep_for_training(args, num_train_optimization_steps)

    train(
        args,
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        optimizer,
        scheduler,
    )


if __name__ == "__main__":
    main()
