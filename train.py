from __future__ import absolute_import, division, print_function

import argparse
from pytorch_transformers.modeling_roberta import RobertaConfig
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.nn import L1Loss, MSELoss
from pytorch_transformers import WarmupLinearSchedule, AdamW

from utils.databuilder import set_up_data_loader
from utils.set_seed import set_random_seed, seed
from utils.metric import score_model
import config.global_configs as cg
import os
from networks.subnet.all_loss import diss_loss
from networks.SentiLARE import RobertaForSequenceClassification
##########bert###########
from pytorch_transformers import BertConfig
########################

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="cjs_model")
    parser.add_argument("--dataset", type=str,
                        choices=["mosi", "mosei", "sims", "simsv2"], default="mosi")
    parser.add_argument("--data_path", type=str, default='/media/cjs/a4208921-d8a3-41bd-9da2-9e7319a43785/cjs-first-paper/datasets/dataset/MOSI_16_sentilare_unaligned_data.pkl')
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--dev_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--beta_shift", type=float, default=1.0)
    parser.add_argument("--dropout_prob", type=float, default=0.5) #0.5
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument(
        "--model",
        type=str,
        choices=["bert-base-uncased", "xlnet-base-cased", "roberta-base", "bert-base-chinese"],
        default="roberta-base")
    parser.add_argument("--model_name_or_path", default='/media/cjs/a4208921-d8a3-41bd-9da2-9e7319a43785/cjs-first-paper/pretrained-model/sentilare_model', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--learning_rate", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--gradient_accumulation_step", type=int, default=1)
    parser.add_argument("--test_step", type=int, default=20)
    parser.add_argument("--test_flag", type=bool, default=False)
    parser.add_argument("--max_grad_norm", type=int, default=2)
    parser.add_argument("--warmup_proportion", type=float, default=0.4)
    parser.add_argument("--seed", type=seed, default=6758, help="integer or 'random'")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    return parser.parse_args()


def prep_for_training(args, num_train_optimization_steps: int):
    config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=1, finetuning_task='sst')
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config, pos_tag_embedding=True, senti_embedding=True, polarity_embedding=True)
    model.to(cg.DEVICE)
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
    cg.change_flag = 0  #切换训练、验证、测试标志
    nb_tr_steps = 0

    all_text_outputs = []
    all_visual_outputs = []
    all_audio_outputs = []

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

        loss = loss_mse + 0.05*loss_diss
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
        ###########################################
        # all_fusion_ot.append(outputs[0].cpu().numpy())
        all_text_outputs.append(outputs[1][1].detach().cpu().numpy())
        all_visual_outputs.append(outputs[1][2].detach().cpu().numpy())
        all_audio_outputs.append(outputs[1][3].detach().cpu().numpy())

    preds = np.array(preds)
    labels = np.array(labels)

    if cg.change_flag == 0:
        return tr_loss / nb_tr_steps, preds, labels, all_text_outputs, all_visual_outputs, all_audio_outputs
    else:
        return tr_loss / nb_tr_steps, preds, labels


def evaluate_epoch(args,mode:str, model: nn.Module, dataloader: DataLoader):
    model.eval()
    cg.change_flag =1 #切换训练、验证、测试标志
    if mode == "test":
        cg.change_flag = 2  # 切换训练、验证、测试标志
        model.load_state_dict(torch.load(
            f'checkpoints/model_{args.model_name}.std', weights_only=True))

    preds = []
    labels = []
    loss = 0
    nb_dev_examples, nb_steps = 0, 0
    # 初始化用于保存每个 batch 输出特征的列表
    all_text_outputs = []
    all_visual_outputs = []
    all_audio_outputs = []

    # all_raw_text_ot = []
    # all_raw_fu_ot = []
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
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()
            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()
            preds.extend(logits)
            labels.extend(label_ids)

            # 保存当前 batch 的输出特征
            # if mode != "test":
            #     # all_fusion_ot.append(outputs[0].cpu().numpy())
            #     all_text_outputs.append(outputs[1][1].cpu().numpy())
            #     all_visual_outputs.append(outputs[1][2].cpu().numpy())
            #     all_audio_outputs.append(outputs[1][3].cpu().numpy())
        preds = np.array(preds)
        labels = np.array(labels)
        # if cg.change_flag ==1:
        #     return loss / nb_steps, preds, labels, all_text_outputs, all_visual_outputs, all_audio_outputs
        # else:
        return loss / nb_steps, preds, labels


def train(
        args,
        model,
        train_dataloader,
        validation_dataloader,
        test_data_loader,
        optimizer,
        scheduler, ):
    # valid_losses = []
    # test_accuracies = []
    # test_maes = []
    text_outputs = []
    visual_outputs = []
    audio_outputs = []
    curr_patience = patience = args.patience
    num_trials = 1

    best_valid_loss = float('inf')
    for epoch_i in range(int(args.n_epochs)):
        cg.epoch_flag = epoch_i

        train_loss, train_pre, train_label,T,V,A = train_epoch(args, model, train_dataloader, optimizer, scheduler)
        valid_loss, valid_pre, valid_label = evaluate_epoch(args,"dev", model, validation_dataloader)

        train_acc, train_mae, train_corr, train_f_score = score_model(train_pre, train_label)
        valid_acc, valid_mae, valid_corr, valid_f_score = score_model(valid_pre, valid_label)
        ###########################################################################
        text_outputs.extend(T)
        visual_outputs.extend(V)
        audio_outputs.extend(A)

        print(
            "epoch:{}, train_loss:{}, train_acc:{}, valid_mae:{},valid_acc:{}, valid_f_score:{}, valid_loss:{}".format(
                epoch_i, train_loss, train_acc, valid_mae, valid_acc, valid_f_score, valid_loss
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

    # test_accuracies.append(test_acc)
    # test_maes.append(test_mae)
    # 保存三个模态的输出特征用于后续计算
    # np.savez('./checkpoints/multimodal_batch_features.npz',
    #          text_output=text_outputs,
    #          visual_output=visual_outputs,
    #          audio_output=audio_outputs
    #          )

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




def main():
    args = parser_args()
    wandb.init(project="CJSNet", reinit=True, save_code=False)

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