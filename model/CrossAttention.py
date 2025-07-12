import torch.nn as nn
import torch.nn.functional as F
import torch
import os

import torch
import os

def CasualWeighted(vs_data, labels, n_iter=None, save_dir="prob_matrices", mode="train"):
    save_dir = os.path.join(save_dir, mode)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"prob_matrix_batch_{n_iter}.pt")
    
    if os.path.exists(save_path):
        prob_matrix = torch.load(save_path)
    else:
        num_classes = labels.unique().numel()
        n_feat = vs_data.shape[1]
        prob_matrix = torch.zeros_like(vs_data, dtype=torch.float)
        for label in range(num_classes):
            vs_subset = vs_data[labels == label]
            for col in range(n_feat):
                unique_vals, counts = vs_subset[:, col].unique(return_counts=True)
                probs = counts.float() / vs_subset.shape[0]
                for val, prob in zip(unique_vals, probs):
                    mask = (vs_data[:, col] == val) & (labels == label)
                    prob_matrix[mask, col] = prob
        torch.save(prob_matrix, save_path)
    return vs_data * prob_matrix

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
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)
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
        
    def forward(self, x):
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
        in_dim = 768
        embed_dim1 = 64
        embed_dim2 = 128*2 if self.args.CMF else 128 

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.linear_layer = nn.Linear(self.input_dim, embed_dim)
        self.encoder = TransformerEncoder(num_encoder_layers, embed_dim, 
                                          num_heads, hidden_dim, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, embed_dim, 
                                          num_heads, hidden_dim, dropout)
        self.cross_att = MultiHeadAttention(embed_dim2, num_heads, dropout)
        self.CEL = nn.CrossEntropyLoss()
      

        self.resnet = ResNet(input_dim=in_dim, output_dim=128)
        # self.resnet2 = ResNet(input_dim=256, output_dim=embed_dim1)
        self.textcnn = TextCNN(input_dim=in_dim, output_dim=128)
        self.textresnet = TextResNet(input_dim=in_dim, output_dim=128)
        self.task_set()
        self.fc_severity   = nn.Linear(embed_dim2, output_dim_s)
        self.fc_department = nn.Linear(embed_dim2, output_dim_d)

    def task_set(self): 
        loss_names = self.args.loss
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} loss(es)')

    def forward(self, vs_feat, batch):
        ret = dict()
        label1, label2 = batch['Level'],  batch['Dept_digit']
        # ChiefComplaint: [CLS]+tokens
        cc_cls = batch['CC_tokens'][:, 0, :].squeeze()    
        cc_cls0 = self.linear_layer(cc_cls)

        
        enc_cc_cls = self.encoder(cc_cls0) if self.args.SA else cc_cls0
        # enc_cc_cls = self.decoder(cc_cls0, enc_cc_cls0)
        encoder_vs = self.encoder(vs_feat) if self.args.SA else vs_feat
        # ************************************************************************************
        #                            backbone  
        # ************************************************************************************
        if self.args.backbone == 'NomSEN': 
            logit_cc, logit_vs = enc_cc_cls, encoder_vs
        
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

        # ************************************************************************************
        #                              
        # ************************************************************************************
        if self.args.SPNPairs and self.args.mode== 'train':
            p = self.args.prob  
            vs_feat, cc_feat = logit_vs, logit_cc
            ind1, ind2 = batch['VS_UniqueLevel'], batch['VS_UniqueLevel']
            pair_type = ind1 + ind2  

            weights = torch.where(
                pair_type == 0, p*0.5, # 反例对
                torch.where(pair_type == 1, p, 1-p)  # 弱正例对和强正例对的权重分别为 p 和 1-p
            )
            logit_vs = weights[:, None] * vs_feat
            logit_cc = weights[:, None] * cc_feat

        if self.args.CMF:
            concat_vs = torch.cat((logit_vs, torch.flip(logit_cc, dims=[1])), dim=-1)
            concat_cc = torch.cat((logit_cc, torch.flip(logit_vs, dims=[1])), dim=-1) 
            fusion_vs = self.cross_att(concat_vs, concat_vs)
            fusion_cc = self.cross_att(concat_cc, concat_cc)
        else:
            fusion_cc = logit_cc
            fusion_vs = logit_vs
            
        severity_out = self.fc_severity(fusion_vs) 
        department_out = self.fc_department(fusion_cc) 
        
        loss_s = self.CEL(severity_out, label1)
        loss_d = self.CEL(department_out, label2)          
        ret.update({'cel_loss': loss_s + loss_d})

        correct_s = (severity_out.argmax(dim=1) == label1).sum().item() 
        correct_d = (department_out.argmax(dim=1) == label2).sum().item() 
        ret.update({'severity_out': severity_out, 
                    'department_out': department_out, 
                    'correct_s': correct_s, 
                    'correct_d': correct_d})
        return ret, severity_out, department_out

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

