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
class PocketModel(nn.Module):
    def __init__(self):
        super(PocketModel, self).__init__()
        self.pocket_dict = defaultdict(list)

    def update_pocket(self, features, labels):
        for feature, label in zip(features, labels):
            self.pocket_dict[label.item()].append(feature)
        save_pocket_model(self, './try/pocket_model.pth')

    def load_pocket(self, pocket_dict):
        self.pocket_dict = pocket_dict

    def get_similar_class(self, feature, top_k=1):
        similarities = []
        feature = feature.squeeze()

        all_features = []
        all_labels = []
        for label, features in self.pocket_dict.items():
            for stored_feature in features:
                all_features.append(stored_feature.squeeze())
                all_labels.append(label)
        all_features = torch.stack(all_features, dim=0)
        all_labels = torch.tensor(all_labels)
        sim = F.cosine_similarity(feature.unsqueeze(0), all_features)
        top_k_indices = torch.topk(sim, top_k).indices
        top_k_indices = top_k_indices.to(feature.device)  # 将 top_k_indices 移动到 feature 的设备上
        all_labels = all_labels.to(feature.device)  # 将 all_labels 移动到 feature 的设备上
        predicted_labels = all_labels[top_k_indices]
        return predicted_labels.tolist()

    def get_pocket_dict(self):
        return self.pocket_dict

    def forward(self, x):
        return x

def save_pocket_model(pocket_model, filepath):
    torch.save(pocket_model.get_pocket_dict(), filepath)
    print(f"Pocket model saved to {filepath}")

def load_pocket_model(filepath, device):
    pocket_model = PocketModel()
    pocket_dict = torch.load(filepath, map_location=device)
    pocket_model.load_pocket(pocket_dict)
#     print(f"Pocket model loaded from {filepath}")
    return pocket_model

def CasualWeighted(vs_data, labels=None, device=None, pocket_model=None):
    n_feat = vs_data.shape[1]
    prob_matrix = torch.zeros_like(vs_data, dtype=torch.float).to(device)

    if labels is not None and pocket_model is not None:
        for col in range(n_feat):
            unique_vals, counts = vs_data[:, col].unique(return_counts=True)
            probs = counts.float() / vs_data.shape[0]

            for label in labels.unique():
                vs_subset = vs_data[labels == label]
                for val, prob in zip(unique_vals, probs):
                    mask = (vs_data[:, col] == val) & (labels == label)
                    prob_matrix[mask, col] = prob

    elif labels is None and pocket_model is not None:
        for i, feature in enumerate(vs_data):
            # 获取当前样本的预测标签
            predicted_labels = pocket_model.get_similar_class(feature.unsqueeze(0), top_k=3)

            # 处理难样本，增加额外标签
            if len(predicted_labels) > 1:  # 如果预测标签不止一个，说明是难样本
                for extra_label in predicted_labels:
                    # 遍历每个特征列，避免覆盖现有的加权
                    for col in range(n_feat):
                        mask = (vs_data[:, col] == feature[col])  # 获取与当前样本相同特征值的所有样本
                        prob_matrix[mask, col] = torch.max(prob_matrix[mask, col], torch.tensor(0.2).to(device))  # 通过`max`避免覆盖已有的加权值

            # 对每个特征列计算加权概率
            for col in range(n_feat):
                unique_vals, counts = vs_data[:, col].unique(return_counts=True)
                probs = counts.float() / vs_data.shape[0]
                for val, prob in zip(unique_vals, probs):
                    mask = (vs_data[:, col] == val)
                    prob_matrix[mask, col] = prob

    return vs_data * prob_matrix





def extract_features(args, batch, modelVS, device, pocket_model):
    """提取并处理特征的函数"""
    VitalSign = batch['VS']
    # cc_cls = batch['CC_tokens'][:, 0, :].squeeze()  

    if args.SFD:
        if args.mode == 'train':
            pocket_model.update_pocket(VitalSign, batch['Level'])
            vs_feat_cw = CasualWeighted(VitalSign,  device=device, pocket_model=pocket_model)
        elif args.mode == 'valid':
            pocket_model.update_pocket(VitalSign, batch['Level'])
            vs_feat_cw = CasualWeighted(VitalSign, labels=None, device=device, pocket_model=pocket_model)
        else:
            loaded_pocket_model = load_pocket_model('./try/pocket_model.pth', device)
            vs_feat_cw = CasualWeighted(VitalSign, labels=None, device=device, pocket_model=loaded_pocket_model)
        
    else:
        vs_feat_cw = VitalSign
    
    vs_feat = modelVS(vs_feat_cw) 
    return vs_feat



def train(args, train_loader, model, optimizer, scheduler, modelVS):
    pocket_model = PocketModel()  # 训练开始时实例化口袋模型
    if args.grade:
        head_threshold, head_subset, tail_subset = preprocess_loader(train_loader.dataset, quantile=args.quantile)
        print(f"Head threshold: {head_threshold}, Head samples: {len(head_subset)}, Tail samples: {len(tail_subset)}")
        batch_size = train_loader.batch_size
        num_workers = train_loader.num_workers if hasattr(train_loader, 'num_workers') else 4  # 默认使用4个工作线程
        head_loader = create_dataloader(head_subset, batch_size=batch_size, shuffle=True, 
                                        num_workers=num_workers, collate_fn=collate_fn)
        tail_loader = create_dataloader(tail_subset, batch_size=batch_size, shuffle=True, 
                                        num_workers=num_workers, collate_fn=collate_fn)
        meters_train = train_single_loader(args, head_loader, model, optimizer, scheduler, modelVS, device, pocket_model)
        meters_train = train_single_loader(args, tail_loader, model, optimizer, scheduler, modelVS, device, pocket_model)
    else:
        meters_train = train_single_loader(args, train_loader, model, optimizer, scheduler, modelVS, device, pocket_model)
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
def train_single_loader(args, train_loader, model, optimizer, scheduler, modelVS, device, pocket_model):
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

        vs_feat = extract_features(args, batch, modelVS, device, pocket_model)  # 传递 pocket_model
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
    loaded_pocket_model = load_pocket_model('./try/pocket_model.pth', device)  # 加载口袋模型
    with torch.no_grad():
        for batch in valid_loader:
            start_time = time.time()  
            
            batch = {k: v.to(device) for k, v in batch.items()}
            vs_feat = extract_features(args, batch, modelVS, device, loaded_pocket_model)  # 传递加载的口袋模型
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

def calc_metrics(true_labels, pred_labels, task_name):#加权
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
