import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.comm import synchronize
from utils.meter import AverageMeter
import time
from sklearn.metrics import *
from scipy.stats import ttest_ind
import numpy as np
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import torch
from torch.utils.data.dataloader import default_collate
from module.manager import collate_fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def CasualWeighted(vs_data, labels):
    num_classes = labels.unique().numel()  # 标签类别数
    n_feat = vs_data.shape[1]  # 生命体征特征数
    prob_matrix = torch.zeros_like(vs_data, dtype=torch.float)
    for label in range(num_classes):
        vs_subset = vs_data[labels == label]
        for col in range(n_feat):
            unique_vals, counts = vs_subset[:, col].unique(return_counts=True)
            probs = counts.float() / vs_subset.shape[0]
            for val, prob in zip(unique_vals, probs):
                mask = (vs_data[:, col] == val) & (labels == label)
                prob_matrix[mask, col] = prob
    return vs_data * prob_matrix

def MulSum(A, B):
    assert A.shape[0] == B.shape[0], "The first dimension (N) of A and B must be the same."
    B_expanded = B.unsqueeze(2)  # Shape: (N, n_feat2, 1)
    A_expanded = A.unsqueeze(1)  # Shape: (N, 1, n_feat1)
    result = A_expanded * B_expanded  # Shape: (N, n_feat2, n_feat1)
    C = result.sum(dim=1)  # Shape: (N, n_feat1)
    mean = C.mean(dim=0, keepdim=True)
    std = C.std(dim=0, keepdim=True) 
    C_normalized = (C - mean) / (std + 1e-6) 
    return C_normalized


def CasualWeightedTrain(vs_data, labels):
    num_classes = labels.unique().numel()  # 标签类别数
    n_feat = vs_data.shape[1]  # 生命体征特征数
    prob_matrix = torch.zeros_like(vs_data, dtype=torch.float)
    
    for label in range(num_classes):
        vs_subset = vs_data[labels == label]
        for col in range(n_feat):
            unique_vals, counts = vs_subset[:, col].unique(return_counts=True)
            probs = counts.float() / vs_subset.shape[0]
            for val, prob in zip(unique_vals, probs):
                mask = (vs_data[:, col] == val) & (labels == label)
                prob_matrix[mask, col] = prob
    torch.save(prob_matrix, 'train_prob_matrix.pt') 
    return prob_matrix

def CasualWeightedTest(vs_data):
    # 加载训练阶段保存的 prob_matrix
    prob_matrix = torch.load('train_prob_matrix.pt')
    
    # 确保 prob_matrix 具有相同的特征维度
    if vs_data.shape[1] != prob_matrix.shape[1]:
        raise ValueError(f"Number of features in vs_data ({vs_data.shape[1]}) "
                         f"does not match the number of features in prob_matrix ({prob_matrix.shape[1]})")
    
    # 使用广播机制，扩展 prob_matrix 的第一个维度以适应测试集的大小
    if vs_data.shape[0] != prob_matrix.shape[0]:
        prob_matrix = prob_matrix.expand(vs_data.shape[0], -1)  # 扩展维度
        print(f"Expanded prob_matrix shape: {prob_matrix.shape}")
    
    return vs_data * prob_matrix



def extract_features(args, batch, modelVS):
    """提取并处理特征的函数"""
    VitalSign = batch['VS']
    # ChiefComp = batch['CC_tokens'][:, 0, :].squeeze()

    if args.SFD:
        if args.mode== 'train':
            train_prob_matrix = CasualWeightedTrain(VitalSign, batch['Level'])
            vs_feat_cw = VitalSign * train_prob_matrix
        else:
            train_prob_matrix = CasualWeightedTest(VitalSign)
            vs_feat_cw = VitalSign * train_prob_matrix
        # vs_feat_cw = CasualWeighted(VitalSign, batch['Level'])
        # vs_feat_cw = MulSum(VitalSign, ChiefComp)
        # vs_feat_cw = torch.cat((VitalSign, ChiefComp), dim=-1)
        # vs_feat_cw = ChiefComp
    else:
        vs_feat_cw = VitalSign
    vs_feat = modelVS(vs_feat_cw) 
    return vs_feat

def train(args, train_loader, model, optimizer, scheduler, modelVS):
    if args.grade:
        head_threshold, head_subset, tail_subset = preprocess_loader(train_loader.dataset, quantile = args.quantile)
        print(f"Head threshold: {head_threshold}, Head samples: {len(head_subset)}, Tail samples: {len(tail_subset)}")
        batch_size = train_loader.batch_size
        num_workers = train_loader.num_workers if hasattr(train_loader, 'num_workers') else 4  # 默认使用4个工作线程
        head_loader = create_dataloader(head_subset, batch_size=batch_size, shuffle=True, 
                                        num_workers=num_workers, collate_fn=collate_fn)
        tail_loader = create_dataloader(tail_subset, batch_size=batch_size, shuffle=True, 
                                        num_workers=num_workers, collate_fn=collate_fn)
        meters_train = train_single_loader(args, head_loader, model, optimizer, scheduler, modelVS)
        meters_train = train_single_loader(args, tail_loader, model, optimizer, scheduler, modelVS)
    else:
        meters_train = train_single_loader(args, train_loader, model, optimizer, scheduler, modelVS)
    return meters_train

def preprocess_loader(dataset, quantile=0.3):
    label_count = defaultdict(int)
    for sample in dataset:
        if isinstance(sample, dict) and "Dept_digit" in sample:
            label = sample["Dept_digit"]
        else:
            raise ValueError(f"Dataset format not supported. Found sample: {sample}")

        label_count[label.item()] += 1 
    counts = np.array(list(label_count.values()))
    threshold = np.percentile(counts, quantile * 100)
    head_indices, tail_indices = [], []
    for idx, sample in enumerate(dataset):
        if isinstance(sample, dict) and "Dept_digit" in sample:
            label = sample["Dept_digit"]
        else:
            raise ValueError(f"Dataset format not supported. Found sample: {sample}")
        if label_count[label.item()] > threshold:
            head_indices.append(idx)
        else:
            tail_indices.append(idx)

    return threshold, Subset(dataset, head_indices), Subset(dataset, tail_indices)

def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4, collate_fn=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                      num_workers=num_workers, collate_fn=collate_fn)

def train_single_loader(args, train_loader, model, optimizer, scheduler, modelVS):
    model.train()
    modelVS.train()
    meters_train = {
        "loss": AverageMeter(),
        "kl_loss": AverageMeter(),
        "cmc_loss": AverageMeter(),
        "cel_loss": AverageMeter(),
        "pcl_loss": AverageMeter(),
        "cont_loss": AverageMeter(),
        "correct_s": AverageMeter(),
        "correct_d": AverageMeter(),
    }
    for n_iter, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad()
        modelVS.zero_grad()
        optimizer.zero_grad()

        vs_feat = extract_features(args, batch, modelVS)
        ret, _, _ = model(vs_feat, batch)
        total_loss = sum([v for k, v in ret.items() if "loss" in k])

        batch_size = batch['CC_tokens'].shape[0]
        meters_train['loss'].update(total_loss.item(), batch_size)
        meters_train['kl_loss'].update(ret.get('kl_loss', 0), batch_size)
        meters_train['cmc_loss'].update(ret.get('cmc_loss', 0), batch_size)
        meters_train['cel_loss'].update(ret.get('cel_loss', 0), batch_size)
        meters_train['pcl_loss'].update(ret.get('pcl_loss', 0), batch_size)
        meters_train['cont_loss'].update(ret.get('cont_loss', 0), batch_size)
        meters_train['correct_s'].update(ret.get('correct_s', 0), batch_size)
        meters_train['correct_d'].update(ret.get('correct_d', 0), batch_size)

        total_loss.backward()
        optimizer.step()
        synchronize()
    scheduler.step()
    return meters_train




def evaluate(args, valid_loader, model, modelVS):
    model.eval()
    modelVS.eval()
    
    meters_val = {
        "loss": AverageMeter(),
        "kl_loss": AverageMeter(),
        "cmc_loss": AverageMeter(),
        "cel_loss": AverageMeter(),
        "pcl_loss": AverageMeter(),
        "cont_loss": AverageMeter(),
        "correct_s": AverageMeter(),
        "correct_d": AverageMeter()
    }
    
    classification_metrics = {}
    all_probs_s, all_probs_d = [], []  # 存储概率分布
    all_labels_s, all_preds_s = [], []
    all_labels_d, all_preds_d = [], []
    processing_times = []  # 用于记录每条数据的处理时间
    
    cc_feat = []
    vs_feat0 = []
    with torch.no_grad():
        for batch in valid_loader:
            start_time = time.time()  
            
            batch = {k: v.to(device) for k, v in batch.items()}
            vs_feat = extract_features(args, batch, modelVS)
            ret, fusion1_cc, vs0 = model(vs_feat, batch)
            cc_feat.append(fusion1_cc)
            vs_feat0.append(vs0)
            
            total_loss = sum([v for k, v in ret.items() if "loss" in k])
            batch_size = batch['CC_tokens'].shape[0]
            
            meters_val['loss'].update(total_loss.item(), batch_size)
            meters_val['kl_loss'].update(ret.get('kl_loss', 0), batch_size)
            meters_val['cmc_loss'].update(ret.get('cmc_loss', 0), batch_size)
            meters_val['cel_loss'].update(ret.get('cel_loss', 0), batch_size)
            meters_val['pcl_loss'].update(ret.get('pcl_loss', 0), batch_size)
            meters_val['cont_loss'].update(ret.get('cont_loss', 0), batch_size)
            
            meters_val['correct_s'].update(ret.get('correct_s', 0), batch_size)
            meters_val['correct_d'].update(ret.get('correct_d', 0), batch_size)
            
            all_labels_s.extend(batch['Level'].cpu().numpy())
            all_preds_s.extend(ret.get('severity_out').argmax(dim=1).cpu().numpy())
            all_labels_d.extend(batch['Dept_digit'].cpu().numpy())
            all_preds_d.extend(ret.get('department_out').argmax(dim=1).cpu().numpy())
            
            elapsed_time = time.time() - start_time
            processing_times.append(elapsed_time / batch_size)

            if args.mode == 'test':
                all_probs_s.extend(torch.softmax(ret.get('severity_out'), dim=1).cpu().numpy())  # 保存概率
                all_probs_d.extend(torch.softmax(ret.get('department_out'), dim=1).cpu().numpy())  # 保存概率

    classification_metrics.update(
        calc_metrics(
            all_labels_s, 
            all_preds_s, 
            "severity"))
    classification_metrics.update(
        calc_metrics(
            all_labels_d, 
            all_preds_d,
            "department"))
    
    total_time = sum(processing_times)
    avg_time_per_sample = total_time * 1000 / len(processing_times)
    if args.mode == 'train' or args.mode == 'valid':
        return meters_val, classification_metrics, avg_time_per_sample
    else:
        return meters_val, classification_metrics, avg_time_per_sample, all_labels_s, all_probs_s, all_labels_d, all_probs_d, cc_feat, vs_feat0

def calc_kappa(true_labels, pred_labels):
    return cohen_kappa_score(true_labels, pred_labels)

def calc_metrics(true_labels, pred_labels, task_name):
    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    kappa = calc_kappa(true_labels, pred_labels)
    return {
        f"{task_name}_accuracy": report.get("accuracy", 0),
        f"{task_name}_f1_macro": report.get("weighted avg", {}).get("f1-score", 0),
        f"{task_name}_precision": report.get("weighted avg", {}).get("precision", 0),
        f"{task_name}_sensitivity": recall_score(true_labels, pred_labels, average="weighted", zero_division=0),
        f"{task_name}_specificity": precision_score(true_labels, pred_labels, average="weighted", zero_division=0),
        f"{task_name}_kappa": kappa
    }


def print_metrics(metrics, file_path=None, mode=None):
    lines = []
    if mode == "test":
        all_metrics = sorted({key.split("_", 1)[1] for key in metrics})
        header = "\t".join(["Metric"] + all_metrics)
        lines.append(header)
        tasks = ["severity", "department"]
        for task in tasks:
            task_values = [f"{metrics.get(f'{task}_{metric}', 0):.4f}" if isinstance(metrics.get(f'{task}_{metric}', 0), 
                                                                                     float) else str(metrics.get(f'{task}_{metric}', 0))
                           for metric in all_metrics]
            lines.append("\t".join([task.capitalize()] + task_values))
    else:
        lines.append("Evaluation Metrics:")
        lines.append("*" * 40)
        tasks = {}
        for key, value in metrics.items():
            task, metric = key.split("_", 1)
            tasks.setdefault(task, {})[metric] = value

        for task, task_metrics in tasks.items():
            lines.append(f"Task: {task.capitalize()}")
            lines.append(f"{'Metric':<30}{'Value':<15}")
            lines.append("-" * 40)
            for metric, value in task_metrics.items():
                if isinstance(value, float):
                    lines.append(f"{metric:<30}{value:<15.4f}")
                else:
                    lines.append(f"{metric:<30}{value}")
            lines.append("-" * 40)
    output = "\n".join(lines)
    print(output)
    state = "w" if mode == "test" else "a"
    if file_path:
        with open(file_path, state, encoding="utf-8") as f:
            f.write(output + "\n")
