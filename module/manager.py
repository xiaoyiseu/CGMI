import torch
from torch.utils.data.dataloader import default_collate
from module.MisData import DataImputer

def collate_fn(batch):
    max_len_cc = max(sample['CC_tokens'].size(0) for sample in batch)
    max_len_dept = max(sample['Dept_tokens'].size(0) for sample in batch)
    for sample in batch:
        cc_padding_len = max_len_cc - sample['CC_tokens'].size(0)# CC_tokens填充 
        if cc_padding_len > 0:
            cc_padding = torch.zeros((cc_padding_len, sample['CC_tokens'].size(1)), device=sample['CC_tokens'].device)
            sample['CC_tokens'] = torch.cat([sample['CC_tokens'], cc_padding], dim=0)
        
        dept_padding_len = max_len_dept - sample['Dept_tokens'].size(0)# Dept_tokens填充 
        if dept_padding_len > 0:
            dept_padding = torch.zeros((dept_padding_len, sample['Dept_tokens'].size(1)), device=sample['Dept_tokens'].device)
            sample['Dept_tokens'] = torch.cat([sample['Dept_tokens'], dept_padding], dim=0)
    return default_collate(batch)

def Match(indices, JointFeature):
    return [(JointFeature['VS'][idx], 
             JointFeature['CC'][idx],
             JointFeature['Level'][idx],
             JointFeature['Depart'][idx],
             JointFeature['Depart_cn'][idx]) for idx in indices]

import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import pkuseg
from module.DataEncoder import VitalSigDataset, ChiefCompDataset
vsEncoder = VitalSigDataset()

def Data_Indices(args):
    segcut = pkuseg.pkuseg(model_name = "medicine", user_dict = "default", postag = False)
    
    RawData = pd.read_csv(args.data_path, engine='python')
    stopwords = pd.read_csv(args.stopword_path, quoting=3, sep="\t", encoding='utf-8')
    
    data = pd.DataFrame(RawData, columns=['性别','出生日期','分诊时间','到院方式',
                                          '分诊印象','T℃', 'P(次/分)', 'R(次/分)', 'BP(mmHg)', 'SpO2', 
                                          '级别','去向'])
    data['去向'].replace('动物致伤','外科',inplace = True)
    data['去向'].replace('创伤中心','创伤救治中心',inplace = True)
    data['去向'].replace('神外','神经外科',inplace = True)
    data['去向'].replace('内','内科',inplace = True)
    data['去向'].replace('外','外科',inplace = True)
    data['去向'].replace('妇产科','产科',inplace = True)

    # data['去向'].replace('产科','妇产科',inplace = True)
    # data['去向'].replace('妇科','妇产科',inplace = True)
    # data['去向'].replace('神经外科','神经科',inplace = True)
    # data['去向'].replace('神经内科','神经科',inplace = True)
    data['分诊印象'].replace('\x7f骨科转入 手指痛','骨科转入 手指痛',inplace = True)
    ChiefComp = pd.Series(data['分诊印象'][:args.length])

    unique_levels = data['级别'][:args.length].dropna().unique()
    unique_departments = data['去向'][:args.length].dropna().unique()
    dic1 = {level: idx for idx, level in enumerate(unique_levels)}
    dic2 = {dept:  idx for idx, dept in enumerate(unique_departments)}
    Y1 = torch.tensor(data[:args.length]['级别'].map(dic1).fillna(-1).astype(int).values).long()
    Y2 = torch.tensor(data[:args.length]['去向'].map(dic2).fillna(-1).astype(int).values).long()
    labels_sety, labels_dept, labels_dept_cn = Y1, Y2, data[:args.length]['去向'].str.replace('科', '', regex=False)    


    exclude = []
    for i in np.asarray(stopwords):
        exclude.append(i[0])
    exclude.extend(' ')

    def langseg(text, exclude):
        words = segcut.cut(text)
        # words = jieba.lcut(text)
        return " ".join([word for word in words if word not in exclude])
    
    im = ChiefComp.apply(lambda x: langseg(x, exclude)).fillna("").tolist()

    if args.ImputMode in ['RF', 'MICE', 'GAN', 'VAE', 'CGMM']:
        imputer = DataImputer(args, latent_dim=16, learning_rate=1e-3, epochs=2000)
        data = imputer.impute(data)
        vs = torch.cat(vsEncoder.Structure(data), dim=1)
        if args.SFD:
            vs = vsEncoder.SFD_encoder(vs)
    else:
        vs = torch.cat(vsEncoder.Structure(data), dim=1)
        if args.SFD:
            vs = vsEncoder.SFD_encoder(vs)        

    train_indices, valid_test_indices = train_test_split(np.arange(Y1.size(0)), test_size=0.2, random_state=42)
    valid_indices, test_indices = train_test_split(valid_test_indices, test_size=0.5, random_state=42)
    
    JointFeature = {'VS': vs[:args.length],
                    'CC': im[:args.length],
                    'Level': labels_sety[:args.length], 
                    'Depart':labels_dept[:args.length],
                    'Depart_cn':labels_dept_cn[:args.length]}    
    
    if args.mode =='train':
        num_severity = len(torch.unique(Y1))
        num_department = len(torch.unique(Y2))

        train_data = Match(train_indices, JointFeature)
        valid_data = Match(valid_indices, JointFeature)
        train_set = ChiefCompDataset(args, train_data, args.ImputMode + '_' + args.mode)
        valid_set = ChiefCompDataset(args, valid_data, args.ImputMode + '_' + 'valid')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,collate_fn=collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,collate_fn=collate_fn)
        return train_loader, valid_loader, num_severity, num_department
    else:
        inverse_dic1 = {v: SeverityLib[k] for k, v in dic1.items()}
        inverse_dic2 = {v: DepartmentLib[k] for k, v in dic2.items()}

        test_data = Match(test_indices, JointFeature)
        test_set = ChiefCompDataset(args, test_data, args.ImputMode + '_' + args.mode)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,collate_fn=collate_fn)
        return test_loader, inverse_dic1, inverse_dic2

SeverityLib={'一级':'Level 1', 
             '二级':'Level 2', 
             '三级':'Level 3', 
             '四级':'Level 4'}
DepartmentLib={'内科':'Internal Medicine', 
               '产科':'Obstetrics', 
               '外科':'Surgery', 
               '眼科':'Ophthalmology', 
               '妇科':'Gynecology', 
               '耳鼻喉':'Otolaryngology',  
               '神经外科':'Neurosurgery', 
               '创伤救治中心':'Trauma Center',
               '骨科':'Orthopedics',
               '神经内科':'Neurology'
              }   
# DepartmentLib={'内科':'Internal Medicine', 
#                '妇产科':'ob-gyn', 
#                '外科':'Surgery', 
#                '眼科':'Ophthalmology', 
#                '耳鼻喉':'Otolaryngology',  
#                '神经科':'Neurology', 
#                '创伤救治中心':'Trauma Center',
#                '骨科':'Orthopedics',
#               }   