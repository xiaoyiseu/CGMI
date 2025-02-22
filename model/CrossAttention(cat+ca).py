import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import objectives

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(FeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_dim, 
                            hidden_dim // 2, 
                            num_layers=num_layers, 
                            batch_first=True,
                            bidirectional=True) 
        self.fc = nn.Linear(hidden_dim, output_dim) 

    def forward(self, x):
        if len(x.shape) !=3:
            x = x.unsqueeze(1) 
        lstm_out, _ = self.lstm(x) 
        output = self.fc(lstm_out[:, -1, :]) 
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, 
                                               num_heads=num_heads, 
                                               dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x, y=None):
        if y is None:
            y = x
        attn_output, _ = self.attention(x, y, y)
        # Add & Norm
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):# 前馈网络层
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        # Add & Norm
        x = residual + x
        x = self.norm(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, hidden_dim, dropout)
        
    def forward(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_att = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, hidden_dim, dropout)
        
    def forward(self, x, encoder_output):
        x = self.self_attention(x)
        x = self.cross_att(x, encoder_output)
        x = self.ffn(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, encoder_output):
        for layer in self.layers:
            x = layer(x, encoder_output)
        return x

class Transformer(nn.Module):

    def __init__(self, args, input_dim, embed_dim, num_heads, 
                 hidden_dim, num_encoder_layers, num_decoder_layers, 
                 output_dim_s, output_dim_d, 
                 dropout=0.1):
        super(Transformer, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.linear_layer = nn.Linear(self.input_dim, embed_dim)
        self.encoder = TransformerEncoder(num_encoder_layers, embed_dim, 
                                          num_heads, hidden_dim, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, embed_dim, 
                                          num_heads, hidden_dim, dropout)
        self.cross_att = MultiHeadAttention(256, num_heads, dropout)
        self.CEL = nn.CrossEntropyLoss()

        in_dim = 768
        embed_dim1 = 256
        embed_dim2 = 128*2 if self.args.CMF else 128       
        self.RNN   = nn.LSTM(input_size=in_dim, hidden_size=embed_dim1, batch_first=True)
        self.fc_severity   = nn.Linear(embed_dim1, output_dim_s)
        self.fc_department = nn.Linear(embed_dim1, output_dim_d)
        self.resnet = ResNet(input_dim=in_dim, output_dim=128)
        self.textcnn = TextCNN(input_dim=in_dim, output_dim=128)
        self.textresnet = TextResNet(input_dim=in_dim, output_dim=128)
        self.task_set()
        self.norm = nn.LayerNorm(embed_dim1)

    def task_set(self): 
        loss_names = self.args.loss
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} loss(es)')

    def forward(self, vs_feat, batch):
        ret = dict()
        label1, label2 = batch['Level'],  batch['Dept_digit']
        # ChiefComplaint: [CLS]+tokens
        cc_cls = batch['CC_tokens'][:, 0, :].squeeze()  
        cc_tokens = batch['CC_tokens'][:, 1:-1, :].squeeze()  
        cc_cls0 = self.linear_layer(cc_cls)

        
        enc_cc_cls = self.encoder(cc_cls0) if self.args.SA else cc_cls0
        # enc_cc_cls = self.decoder(cc_cls0, enc_cc_cls0)
        encoder_vs = self.encoder(vs_feat) if self.args.SA else vs_feat
        # ************************************************************************************
        #                            backbone  
        # ************************************************************************************
        if self.args.backbone == 'NoLocalCC': 
            # fusion1_cc =self.cross_att(enc_cc_cls, encoder_vs)
            # fusion2_vs =self.cross_att(encoder_vs, enc_cc_cls)
            logit_cc, logit_vs = enc_cc_cls, encoder_vs

        if self.args.backbone == 'Concat':  
            _, (h_n_rst, _) = self.RNN(cc_tokens)
            stats = statistics(h_n_rst[-1]) 
            cc_tks_stats = stats.view(stats.size(0), -1) 
            linear_layer = nn.Linear(cc_tks_stats.size(-1), 128).to(cc_tokens.device) 
            cc_tks_stats = linear_layer(cc_tks_stats) 

            # cc_tks = self.encoder(cc_tks_stats)
            LSTM   = FeatureExtractor(input_dim=cc_tks_stats.size(-1), hidden_dim=64, output_dim=128).to(cc_tks_stats.device)     
            cc_tks = LSTM(cc_tks_stats)

            # fusion1_cc = torch.cat((cc_tks, enc_cc_cls), dim = -1)  
            logit_cc = self.cross_att(enc_cc_cls, cc_tks)  
            logit_vs = self.cross_att(encoder_vs, enc_cc_cls)  

        if self.args.backbone == 'CrossAtt':
            cc_tokens0 = self.linear_layer(cc_tokens) 
            if self.args.filt:
                padding_mask = (cc_tokens.abs().sum(dim=-1) > 0).float()
                final_mask_cc = filter_tokens(cc_cls0.unsqueeze(1), cc_tokens0, 
                                              padding_mask=padding_mask, 
                                              threshold=self.args.threshold)            
                expanded_mask = final_mask_cc.unsqueeze(-1)
                tokens = cc_tokens * expanded_mask.float() #[B, N, 786]
            else :
                tokens = cc_tokens
            cc_toks = self.textcnn(tokens)            
            logit_cc = enc_cc_cls
            logit_vs = encoder_vs

        if self.args.backbone == 'NoCrossAtt':
            logit_cc = enc_cc_cls
            logit_vs = encoder_vs
        
        if self.args.backbone == 'Transformer':
            logit_cc = self.decoder(cc_cls0, enc_cc_cls)
            logit_vs = encoder_vs

        if self.args.backbone == 'ResNet':
            cc_all_tokens = batch['CC_tokens'].squeeze() 
            mask = (cc_all_tokens.abs().sum(dim=-1) > 0).float()
            logit_cc = self.resnet(cc_all_tokens, mask=mask)       
            logit_vs = encoder_vs

        if self.args.backbone == 'TextCNN' or self.args.backbone == 'TextResNet':
            cc_all_tokens = batch['CC_tokens'].squeeze() 
            mask = (cc_all_tokens.abs().sum(dim=-1) > 0).float()
            if self.args.backbone == 'TextCNN':
                logit_cc = self.textcnn(cc_all_tokens, mask=mask) 
            elif self.args.backbone == 'TextResNet':
                logit_cc = self.textresnet(cc_all_tokens, mask=mask)                              
            logit_vs = encoder_vs

        vs_fusion = torch.cat((logit_vs, logit_cc), dim=-1) 
        cc_fusion = torch.cat((logit_cc, logit_vs), dim=-1) 
        # print(vs_fusion.shape, cc_fusion.shape)

        if self.args.CA:
            fusion1_cc = self.cross_att(cc_fusion, vs_fusion)   
            fusion2_vs =  self.cross_att(vs_fusion, cc_fusion)
        else:
            fusion1_cc = logit_cc
            fusion2_vs = logit_vs
        
        if self.args.CMF:
            if self.args.backbone == 'CrossAtt':
                fusion_data = torch.cat((fusion1_cc, cc_toks), dim=-1) 
            else:
                fusion_data = torch.cat((fusion1_cc, fusion2_vs), dim=-1)
        else:
            fusion_data = fusion1_cc
            
        severity_out = self.fc_severity(fusion2_vs) # or fusion_data(略有提升)
        department_out = self.fc_department(fusion_data) 
        # ************************************************************************************
        #                            loss  
        # ************************************************************************************
        if 'pdc' in self.current_task:
            dpt_cls = batch['Dept_tokens'].squeeze()  
            mask = (dpt_cls.abs().sum(dim=-1) > 0).float()
            cc_labels_token= self.textresnet(dpt_cls, mask=mask) 
            m_loss = objectives.contrastive_loss2(logit_cc, cc_labels_token, label2, temperature= self.args.temp)
            ret.update({'kl_loss': m_loss}) 

        if 'pcl' in self.current_task:# Focal loss
            dpt_cls = batch['Dept_tokens'].squeeze()  
            mask = (dpt_cls.abs().sum(dim=-1) > 0).float()
            cc_labels_token= self.textresnet(dpt_cls, mask=mask)
            focal_loss  = objectives.sim_loss(logit_cc, cc_labels_token, temperature= self.args.temp)         
            # focal_loss = objectives.contrastive_loss(fusion1_cc, label2, temperature= self.args.temp)         
            ret.update({'pcl_loss': focal_loss})

        if 'ctl' in self.current_task:  
            dpt_cls = batch['Dept_tokens'].squeeze()  
            mask = (dpt_cls.abs().sum(dim=-1) > 0).float()
            cc_labels_token= self.textresnet(dpt_cls, mask=mask) 

            # loss1  = objectives.cls_token_loss(logit_cc, cc_labels_token,temperature= self.args.temp)
            loss1  = objectives.kl_matching_loss(logit_cc, cc_labels_token,temperature= self.args.temp)
            ret.update({'cont_loss': loss1})

        if 'cmc' in self.current_task: #cross-moadal contrastive loss
            mcu_loss = objectives.contrastive_loss(cc_fusion, label2,temperature = self.args.temp)           
            ret.update({'cmc_loss': mcu_loss}) 
        
        loss_s = self.CEL(severity_out, label1)
        loss_d = self.CEL(department_out, label2)          
        ret.update({'cel_loss': loss_s + loss_d})

        correct_s = (severity_out.argmax(dim=1) == label1).sum().item() 
        correct_d = (department_out.argmax(dim=1) == label2).sum().item() 
        ret.update({'severity_out': severity_out, 
                    'department_out': department_out, 
                    'correct_s': correct_s, 
                    'correct_d': correct_d})
        return ret, department_out, severity_out

def Resample(data, labels, n_dim = 128):
    unique_labels = torch.unique(labels)
    all_means = []
    all_vars = []
    for label in unique_labels:
        category_data = data[labels == label]  # 直接获取该类别所有的样本
        mean = category_data.mean(dim=1)  # (num_samples_in_category, n_feat1)
        var = category_data.var(dim=1, unbiased=False)  # (num_samples_in_category, n_feat1)
        all_means.append(mean)
        all_vars.append(var)
    all_means = torch.cat(all_means, dim=0)  # (B, n_feat1)
    all_vars = torch.cat(all_vars, dim=0)    # (B, n_feat1)
    distributions = torch.distributions.Normal(all_means, torch.sqrt(all_vars))
    
    sampled_features = distributions.sample((n_dim,))  # 生成形状为 (n_feat2, B, n_feat1)
    sampled_features = sampled_features.transpose(0, 1)  # (B, n_feat2, n_feat1)
    new_features = sampled_features.mean(dim=-1)  # (B, n_feat2)
    return new_features

def enhance_tokens(cls_embeddings, token_embeddings):
    cls_norm = F.normalize(cls_embeddings, dim=-1)  # [batch_size, 1, dim]
    tokens_norm = F.normalize(token_embeddings, dim=-1)  # [batch_size, seq_len, dim]
    cosine_sim = torch.bmm(tokens_norm, cls_norm.transpose(1, 2))
    attention_weights = F.softmax(cosine_sim, dim=1)  # [batch_size, seq_len, 1]
    enhanced_tokens = token_embeddings * attention_weights  # [batch_size, seq_len, dim]
    return enhanced_tokens

def filter_tokens(cls_embeddings, token_embeddings, padding_mask=None, threshold=0.5):
    enhanced_tokens = enhance_tokens(cls_embeddings, token_embeddings)
    cls_norm = F.normalize(cls_embeddings, dim=-1)  # [batch_size, 1, dim]
    enhanced_tokens_norm = F.normalize(enhanced_tokens, dim=-1)  # [batch_size, seq_len, dim]
 
    cosine_sim = torch.bmm(enhanced_tokens_norm, cls_norm.transpose(1, 2)).squeeze(-1)  # [batch_size, seq_len]
    similarity_mask = cosine_sim >= threshold  # [batch_size, seq_len]
    
    if padding_mask is not None:
        final_mask = similarity_mask & padding_mask.bool()  # 考虑 padding_mask
    else:
        final_mask = similarity_mask
    return final_mask


def statistics(data):
    mask = (data != 0).float()
    sum_mask = mask.sum(dim=-1)
    sum_data = (data * mask).sum(dim=-1)
    mean = sum_data / (sum_mask + 1e-8) 
    squared_diff = (data - mean.unsqueeze(-1)) ** 2 
    sum_squared_diff = (squared_diff * mask).sum(dim=-1) 
    std = torch.sqrt(sum_squared_diff / sum_mask.clamp(min=1e-8)) 
    stats = torch.stack((mean, std), dim=-1)
    return stats

class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=1) 
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1) 

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        if mask is not None:
            x = x * mask.unsqueeze(1) 
        x = self.conv1(x)
        residual = x
        x = self.res_block(x)
        x += residual
        x = self.global_pool(x)
        x = x.squeeze(-1) 
        return x

class TextCNN(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_sizes=(2, 3, 4), num_filters=256, dilation=2):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, num_filters, kernel_size=ks, dilation=dilation)
            for ks in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_dim)
        self.activation = nn.ReLU()

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
            x = x * mask
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x)
            conv_out = nn.ReLU()(conv_out)
            conv_out = nn.functional.max_pool1d(conv_out, kernel_size=conv_out.shape[2])
            conv_outputs.append(conv_out.squeeze(2))
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.activation(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, num_filters, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size=kernel_size, dilation=dilation, padding=dilation * (kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, dilation=dilation, padding=dilation * (kernel_size - 1) // 2)
        self.activation = nn.ReLU()
        self.downsample = nn.Conv1d(input_dim, num_filters, kernel_size=1) if input_dim != num_filters else None

    def forward(self, x):
        residual = x
        out = self.activation(self.conv1(x))
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.activation(out + residual)

class TextResNet(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_sizes=(2, 3, 4), num_filters=256, dilation=2):
        super(TextResNet, self).__init__()
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(input_dim, num_filters, kernel_size=ks, dilation=dilation)
            for ks in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_dim)
        self.activation = nn.ReLU()

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
            x = x * mask
        block_outputs = []
        for block in self.residual_blocks:
            block_out = block(x)
            block_out = F.max_pool1d(block_out, kernel_size=block_out.shape[2])
            block_outputs.append(block_out.squeeze(2))
        x = torch.cat(block_outputs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.activation(x)
        return x

