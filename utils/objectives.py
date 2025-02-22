import torch
import torch.nn as nn
import torch.nn.functional as F

# ********************************* sdm loss ***************************************
def sdm_loss(feats, label, temperature=1.0, epsilon=1e-8):
    batch_size = feats.shape[0]
    label = label.reshape((batch_size, 1))
    label_dist = label - label.t()
    labels = (label_dist == 0).float()
    feats_norm = feats / feats.norm(dim=0, keepdim=True)
    cosine_metrix = feats_norm @ feats_norm.t()
    cosine_metrix = cosine_metrix / temperature
    
    labels_distribute = labels / (labels.sum(dim=1, keepdim=True) + epsilon)
    pred = F.softmax(cosine_metrix, dim=1)
    unimodal_loss = pred * (F.log_softmax(cosine_metrix, dim=1) - 
                            torch.log(labels_distribute + epsilon))
    loss = torch.mean(torch.sum(unimodal_loss, dim=1))
    return loss


# ********************************* mask loss ***************************************

def contrastive_loss(features_a, features_b, labels, temperature=0.1, eps=1e-8):
    features_a = F.normalize(features_a, dim=1)
    features_b = F.normalize(features_b, dim=1)
    similarity_ab = torch.mm(features_a, features_b.T)  # 跨模态相似性 (N, N)

    # 从标签生成正样本掩码和负样本掩码
    labels = labels.view(-1, 1)  # 调整形状为 (N, 1)
    positive_mask = (labels == labels.T).float()  # 正样本掩码 (N, N)
    negative_mask = 1.0 - positive_mask           # 负样本掩码 (N, N)
    positive_mask.fill_diagonal_(0)
    negative_mask.fill_diagonal_(0)

    exp_ab = torch.exp(similarity_ab / temperature)  # 跨模态
    denominator_ab = torch.sum(exp_ab, dim=1).clamp(min=eps)  # 跨模态
    positive_ab_sum = torch.sum(exp_ab * positive_mask, dim=1).clamp(min=eps)
    loss_ab = -torch.log(positive_ab_sum / denominator_ab)  # 跨模态
    return torch.mean(loss_ab)

def kl_loss(cc_feat, labels2_cn, temperature=1.0, epsilon=1e-8):
    # Normalize features
    cc_norm = cc_feat / cc_feat.norm(dim=1, keepdim=True)
    label_norm = labels2_cn / labels2_cn.norm(dim=1, keepdim=True)

    # Compute similarity matrices
    cc_cosine = cc_norm @ cc_norm.t() / temperature  # Similarity for cc_feat
    lb_cosine = label_norm @ label_norm.t() / temperature  # Similarity for labels2_cn

    # Convert to probabilities
    p_cc = F.softmax(cc_cosine, dim=1)  # Marginal probabilities P(X)
    p_lb = F.softmax(lb_cosine, dim=1)  # Marginal probabilities P(Y)

    # Joint probability P(X, Y)
    p_joint = (p_cc.unsqueeze(2) * p_lb.unsqueeze(1)).mean(dim=0)  # Combine probabilities

    # Marginal probabilities derived from the joint
    p_x = p_joint.sum(dim=1, keepdim=True)  # Marginal P(X)
    p_y = p_joint.sum(dim=0, keepdim=True)  # Marginal P(Y)

    # Calculate mutual information
    h_x = -torch.sum(p_x * torch.log(p_x + epsilon))  # H(X)
    h_y = -torch.sum(p_y * torch.log(p_y + epsilon))  # H(Y)
    h_xy = -torch.sum(p_joint * torch.log(p_joint + epsilon))  # H(X, Y)

    mutual_info = h_x + h_y - h_xy
    return mutual_info 

import torch
import torch.nn.functional as F

def infonce_loss(fusion2_vs, fusion1_cc, batch, args):

    p = args.prob  
    vs_feat, cc_feat = fusion2_vs, fusion1_cc
    ind1, ind2 = batch['VS_UniqueLevel'], batch['VS_UniqueLevel']
    pair_type = ind1 + ind2  
    
    weights = torch.where(
        pair_type == 0, p*0.2, # 反例对
        torch.where(pair_type == 1, p, 1 - p)  # 弱正例对和强正例对的权重分别为 p 和 1-p
    )
    weig_vs = weights[:, None] * vs_feat
    weig_cc = weights[:, None] * cc_feat

    sim_vs = weig_vs @ weig_vs.t() / args.temp 
    sim_cc = weig_cc @ weig_cc.t() / args.temp 

    loss = F.cross_entropy(sim_vs, batch['Level']) + F.cross_entropy(sim_cc, batch['Dept_digit'])
    return loss/2

# ********************************* pcl loss ***************************************
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        focal_loss = (1 - torch.exp(-ce_loss)) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def class_weights(labels):
    num_classes = torch.max(labels) + 1 
    class_counts = torch.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    alpha = class_weights / class_weights.sum()
    return alpha

def prob_loss(cc_feat, labels2, temperature=1.0):
    alpha_cc = class_weights(labels2)
    focal_loss_cc = FocalLoss(alpha=alpha_cc, gamma=2.0, reduction='mean')
    loss_d = focal_loss_cc(cc_feat / temperature, labels2)
    return loss_d


def cls_token_loss(fusion1_cc, fusion1_dpt, temperature=1.0): 
    batch_size = fusion1_cc.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(fusion1_cc.device)

    # normalized features
    image_norm = fusion1_cc / fusion1_cc.norm(dim=-1, keepdim=True)
    text_norm = fusion1_dpt / fusion1_dpt.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = image_norm @ text_norm.t() / temperature
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2
    return loss



class FeatureSimilarityLoss(nn.Module):
    def __init__(self, n_feat1, n_feat2, temperature=0.1, device=None):
        super(FeatureSimilarityLoss, self).__init__()
        self.project_layer = nn.Linear(n_feat1, n_feat2, bias=False).to(device)  # 投影层
        self.temperature = temperature

    def forward(self, A, B_local):
        batch_size, m, n_feat2 = B_local.size()
        # 1. 特征对齐：A 投影到 B 的特征空间
        A_proj = self.project_layer(A)  # (batch_size, n_feat2)

        # 2. 动态权重：计算注意力权重
        attention_scores = torch.matmul(B_local, A_proj.unsqueeze(-1))  # (batch_size, m, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, m, 1)
        B_weighted = (B_local * attention_weights).sum(dim=1)  # 加权后的局部特征 (batch_size, n_feat2)

        # 3. 正负样本对比：InfoNCE 损失
        A_proj = F.normalize(A_proj, dim=-1)  # 归一化
        B_weighted = F.normalize(B_weighted, dim=-1)

        # 正样本相似性
        positive_sim = torch.sum(A_proj * B_weighted, dim=-1)  # (batch_size,)

        # 负样本相似性：从其他样本中采样
        negative_sim = torch.matmul(A_proj, B_weighted.T)  # (batch_size, batch_size)
        mask = torch.eye(batch_size, device=A.device).bool()  # 避免自身成为负样本
        negative_sim = negative_sim[~mask].view(batch_size, batch_size - 1)  # 去掉对角线

        # InfoNCE 损失
        logits = torch.cat([positive_sim.unsqueeze(-1), negative_sim], dim=-1)  # (batch_size, batch_size)
        labels = torch.zeros(batch_size, dtype=torch.long, device=A.device)  # 正样本为第0列
        loss = F.cross_entropy(logits / self.temperature, labels)

        # 4. 正交约束
        proj_weights = self.project_layer.weight  # (n_feat2, n_feat1)
        orthogonal_loss = torch.norm(proj_weights @ proj_weights.T - torch.eye(n_feat2, device=A.device))

        # 总损失
        total_loss = loss + 0.01 * orthogonal_loss  # 正交约束权重为 0.01
        return total_loss

def sim_loss(fusion1_cc, fusion1_dpt, temperature=1.0):
    n_feat1, n_feat2 = fusion1_cc.size(-1), fusion1_dpt.size(-1)
    loss_fn = FeatureSimilarityLoss(n_feat1, n_feat2, temperature = temperature, device=fusion1_cc.device)
    loss = loss_fn(fusion1_cc, fusion1_dpt)
    return loss




















import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

class CentroidManager:
    def __init__(self, momentum=0.9, logit_scale=50, epsilon=1e-8):
        self.img_centroids = {}
        self.txt_centroids = {}
        self.momentum = momentum
        self.epsilon = epsilon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logit_scale = logit_scale

    def update_centroids(self, features, labels, feature_type='img'):
        centroids = self.img_centroids if feature_type == 'img' else self.txt_centroids
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            label_mask = labels == label
            label_features = features[label_mask]
            centroid = label_features.mean(dim=0)
            if label in centroids:
                centroids[label] = self.momentum * centroids[label] + (1 - self.momentum) * centroid
            else:
                centroids[label] = centroid

        if feature_type == 'img':
            self.img_centroids = centroids
        else:
            self.txt_centroids = centroids

    def cross_modal_prob(self, img_centroids, txt_centroids):
        img_centroid_list = torch.stack(list(img_centroids.values()))
        txt_centroid_list = torch.stack(list(txt_centroids.values()))
        dist = torch.cdist(img_centroid_list, txt_centroid_list, p=2).pow(2)
        dist = dist - dist.max(dim=1, keepdim=True).values
        probabilities = F.softmax((-dist * self.logit_scale) + self.epsilon, dim=1)
        return probabilities

    def centroid_contrastive_loss(self, img_features, txt_features, labels):
        img_features, txt_features, labels = img_features.to(self.device), txt_features.to(self.device), labels.to(self.device)
        
        self.update_centroids(img_features, labels, feature_type='img')
        self.update_centroids(txt_features, labels, feature_type='txt')
        
        cross_modal_probs = self.cross_modal_prob(self.img_centroids, self.txt_centroids).clamp(min=self.epsilon)

        img_labels = torch.tensor(list(self.img_centroids.keys()), device=self.device)
        txt_labels = torch.tensor(list(self.txt_centroids.keys()), device=self.device)
        
        cross_modal_mask = (img_labels.unsqueeze(1) == txt_labels.unsqueeze(0)).float().clamp(min=self.epsilon)
        target_probs = cross_modal_mask / (cross_modal_mask.sum(dim=1, keepdim=True))

        cross_modal_loss = F.kl_div(cross_modal_probs.log(), target_probs, reduction='batchmean')

        return cross_modal_loss

def MCU_loss(features_img, features_txt, y_labels, temperature = 0.1):
    features_img = features_img / features_img.norm(dim=1, keepdim=True)
    features_txt = features_txt / features_txt.norm(dim=1, keepdim=True)   
    centroid_manager = CentroidManager(logit_scale = 1/temperature)

    unique_y = y_labels.unique(sorted=True)
    class_to_idx = {cls.item(): idx for idx, cls in enumerate(unique_y)}
    labels = torch.tensor([class_to_idx[y.item()] for y in y_labels], dtype=torch.long).to(y_labels.device)

    loss = centroid_manager.centroid_contrastive_loss(features_img, features_txt, labels) 
    return loss 

