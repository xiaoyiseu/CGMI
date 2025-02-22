import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import logging
import numpy as np
import pkuseg
from utils.options import get_args
import os, time
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from model.CrossAttention import Transformer, FeatureExtractor
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from sklearn.model_selection import train_test_split
from module.TrainValid import train, evaluate, print_metrics
from module.manager import Data_Indices
from utils.iotools import save_train_configs
from utils.visualization import plot_pr_curve


def Result(meters, batch_size):
    loss = meters['loss'].avg 
    acc_s = meters['correct_s'].avg / batch_size 
    acc_d = meters['correct_d'].avg / batch_size 
    return loss, acc_s, acc_d

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, valid_loader, num_severity, num_department = Data_Indices(args)


    if args.ProbW:
        if args.ProbW and args.SFD:
            input_dim_vs = 28 + args.n_comp
        else:
            input_dim_vs = 29 + args.n_comp
    elif args.SFD and not args.ProbW:       
        input_dim_vs = 28
    else:       
        input_dim_vs = 29
    
    modelVS = FeatureExtractor(input_dim=input_dim_vs, 
                               hidden_dim=args.hidden_dim, 
                               output_dim=128, 
                               num_layers=args.lstm_layers).to(device)

    model = Transformer(args,
                        input_dim=768, 
                        embed_dim=args.embed_dim, #128
                        num_heads=args.num_heads, #4
                        hidden_dim=args.hidden_dim, #256
                        num_encoder_layers=args.nec, #4
                        num_decoder_layers=args.ndc, #4
                        output_dim_s=num_severity, #4
                        output_dim_d=num_department).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-5)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0.0)

    args.output_dir = os.path.join(args.output_dir, 
                                   args.backbone+'_'
                                   + args.ImputMode+'_'
                                   +'ProbW'+str(args.ProbW)+ str(args.n_comp)+'_'
                                #    +'tail'+str(args.quantile)+'_'
                                   +'loss'+str(args.loss)+'_'
                                   +'prob'+str(args.prob)+'_'
                                #    +'filt'+str(args.filt)+'_'
                                   +'SA'+str(args.SA)+'_'
                                   +'SPNPairs'+str(args.SPNPairs)+'_'
                                   +'SFD'+str(args.SFD)+'_'
                                #    +'ProbW'+str(args.ProbW)+'_'
                                   +'CMF'+str(args.CMF)+'_'
                                   +'temp'+str(args.temp)
                                   )
    args.bestmodel = os.path.join(args.output_dir, args.bestmodel)
    args.vsmodel = os.path.join(args.output_dir, args.vsmodel)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.bestmodel), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    args.log_path = os.path.join(args.output_dir, 'training_log.txt')
    save_train_configs(args.output_dir, args)

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)     
    # 配置logging模块
    logging.basicConfig(filename=args.log_path, 
                        level=logging.INFO,  # 设置日志级别
                        format='%(asctime)s - %(message)s', 
                        filemode='a') 
    logger = logging.getLogger()

    best_score = 0.0  
    best_model_info = {}  
    early_stop_counter = 0 
    train_loss2, valid_loss, valid_acc_severity, valid_acc_department = [], [], [], []

    best_model_state = None 
    for epoch in range(1, args.num_epochs + 1):
        start_time = time.time()  # 记录开始时间
        args.mode = 'train'
        meters_train = train(args, train_loader, model, optimizer, scheduler, modelVS)
        args.mode = 'valid'
        meters_val, _, _  = evaluate(args, valid_loader, model, modelVS)

        train_loss, train_acc_s, train_acc_d = Result(meters_train, args.batch_size)
        val_loss, val_acc_s, val_acc_d = Result(meters_val, args.batch_size)
        combined_score = val_acc_s + 0.1 * val_acc_d

        # 检查是否为最优模型
        if val_acc_d > best_score:
            best_score = val_acc_d
            best_model_state = model.state_dict()
            torch.save(model, args.bestmodel)
            torch.save(modelVS, args.vsmodel)

            best_model_info = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc_s': val_acc_s,
                'val_acc_d': val_acc_d,
                'combined_score': combined_score
            }
            early_stop_counter = 0 
        else:
            early_stop_counter += 1 

        if early_stop_counter >= args.patience :
            logger.info(f"Training Finished! Early stopping at epoch {epoch}") 
            break 

        # 记录训练和验证的结果
        train_loss2.append(train_loss)
        valid_loss.append(val_loss)
        valid_acc_severity.append(val_acc_s)
        valid_acc_department.append(val_acc_d)

        # 计算每个epoch的训练时间
        epoch_time = time.time() - start_time
        log_message = (f'Epoch: {epoch:03d}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, '
                       f'val_severity_acc: {val_acc_s:.4f}, val_depart_acc: {val_acc_d:.4f}, '
                       f'time: {epoch_time:.2f} s')
        logger.info(log_message) 
        print(log_message) 

    logger.info(f"Best model found at epoch {best_model_info['epoch']}. \n"
                 f"Train Loss: {best_model_info['train_loss']:.4f}, "
                 f"Val Loss: {best_model_info['val_loss']:.4f}, "
                 f"Val Severity Acc: {best_model_info['val_acc_s']:.4f}, "
                 f"Val Depart Acc: {best_model_info['val_acc_d']:.4f}, "
                 f"Combined Score: {best_model_info['combined_score']:.4f}")
    
    
    ## 测试2
    model0 = torch.load(args.bestmodel)
    modelVS0 = torch.load(args.vsmodel)
    args.mode = 'test'
    test_loader, dic1, dic2 = Data_Indices(args)
    _, metrics, avg_time, true_s, probs_s, true_d, probs_d, cc_feat, vs_feat = evaluate(args, test_loader, model0, modelVS0)
    
    print_metrics(metrics, 
                  file_path=os.path.join(args.output_dir,'test.txt'), 
                  mode = args.mode)
    
    print(f"Ave time: {avg_time:.4f} ms")

    plot_pr_curve(
        true_labels=true_s,
        pred_probs=probs_s,
        class_dict = dic1,
        task_name="severity",
        file_path=os.path.join(args.output_dir, 'pr_curve_severity.svg')
    )
    plot_pr_curve(
        true_labels=true_d,
        pred_probs=probs_d,
        class_dict = dic2,
        task_name="department",
        file_path=os.path.join(args.output_dir, 'pr_curve_department.svg')
    )