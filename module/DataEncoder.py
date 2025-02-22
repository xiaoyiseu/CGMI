from tqdm import tqdm
import torch, os
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset
import jieba.posseg as pseg
from module.StructureEncoder import StructureDataEncoder
import numpy as np

class VitalSigDataset:
    def __init__(self):
        self.digit = StructureDataEncoder()
        self.num_classes = {
            '到院方式': 4,  
            '性别': 3,
            '出生日期': 5,
            'T℃': 3,
            'P(次/分)': 3,
            'R(次/分)': 3,
            'BP(mmHg)': 5,
            'SpO2': 3
        }        
    def one_hot(self, y, num_classes=None):
        """Convert to one-hot encoding."""
        y_tensor = torch.tensor(y)
        if num_classes is None:
            num_classes = y_tensor.max() + 1
        return torch.nn.functional.one_hot(y_tensor, num_classes=num_classes).float()

    def Structure(self, data):
        ar = self.one_hot(data['到院方式'].apply(lambda x: self.digit.Arr_way(x)).values, self.num_classes['到院方式'])
        g  = self.one_hot(data['性别'].apply(lambda x: self.digit.Gender(x)).values, self.num_classes['性别'])
        a  = self.one_hot(data['出生日期'].apply(lambda x: self.digit.Age(x)).values, self.num_classes['出生日期'])
        t  = self.one_hot(data['T℃'].apply(lambda x: self.digit.Temperature(x)).values, self.num_classes['T℃'])
        p  = self.one_hot(data['P(次/分)'].apply(lambda x: self.digit.Pulse(x)).values, self.num_classes['P(次/分)'])
        r  = self.one_hot(data['R(次/分)'].apply(lambda x: self.digit.Respiration(x)).values, self.num_classes['R(次/分)'])
        bp = self.one_hot(data['BP(mmHg)'].apply(lambda x: self.digit.BloodPressure(x)).values, self.num_classes['BP(mmHg)'])
        s  = self.one_hot(data['SpO2'].apply(lambda x: self.digit.SpO2(x)).values, self.num_classes['SpO2'])
        return ar, g, a, t, p, r, bp, s

    def StructureNoOneHot(self, data):
        # 对每一列应用相应的处理函数，并保持每个变量为独立的数组
        ar = np.array(data['到院方式'].apply(lambda x: self.digit.Arr_way(x)).values)
        g  = np.array(data['性别'].apply(lambda x: self.digit.Gender(x)).values)
        a  = np.array(data['出生日期'].apply(lambda x: self.digit.Age(x)).values)
        t  = np.array(data['T℃'].apply(lambda x: self.digit.Temperature(x)).values)
        p  = np.array(data['P(次/分)'].apply(lambda x: self.digit.Pulse(x)).values)
        r  = np.array(data['R(次/分)'].apply(lambda x: self.digit.Respiration(x)).values)
        bp = np.array(data['BP(mmHg)'].apply(lambda x: self.digit.BloodPressure(x)).values)
        s  = np.array(data['SpO2'].apply(lambda x: self.digit.SpO2(x)).values)

        # 使用hstack将这些数组横向拼接
        result = np.hstack([ar[:, np.newaxis], g[:, np.newaxis], a[:, np.newaxis], 
                            t[:, np.newaxis], p[:, np.newaxis], r[:, np.newaxis], 
                            bp[:, np.newaxis], s[:, np.newaxis]])

        return result


    def SFD_encoder(self, vs):
        batch_size, _ = vs.shape
        indices = vs.nonzero(as_tuple=True)[1].view(batch_size, -1)
        num_indices = indices.shape[1]
        distance_matrix = torch.zeros((batch_size, num_indices, num_indices), dtype=torch.float32)
        for idx in range(batch_size):
            feature_indices = indices[idx].float().view(-1, 1)
            dist_matrix = torch.cdist(feature_indices, feature_indices, p=1)
            distance_matrix[idx] = dist_matrix
        tri_indices = torch.triu_indices(distance_matrix.size(1), distance_matrix.size(2), offset=1)
        return distance_matrix[:, tri_indices[0], tri_indices[1]]

#************************************    包含加权-频次
class ChiefCompDataset(Dataset):
    def __init__(self, args, dataset, dataset_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = args.cache_dir
        self.cache_file = os.path.join(
            self.cache_dir, 
            f'cached_{dataset_name}_wmd.pt' if args.SFD 
            else f'cached_{dataset_name}_onehot.pt'
        )
        
        if os.path.exists(self.cache_file):
            print(f"Loading cached data for {dataset_name}...")
            self.data = torch.load(self.cache_file)
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            self.model = BertModel.from_pretrained('bert-base-chinese')
            self.model.to(self.device)
            self.model.eval()
            print(f"Encoding data for {dataset_name}...")
            self.data = self.encode_and_cache_data(dataset)
            torch.save(self.data, self.cache_file)
            print(f"Data encoded and cached for {dataset_name}.")

    # def filter_nouns_verbs(self, text):
    #     words = pseg.cut(text)  # 对文本进行词性标注
    #     filtered_words = [word for word, flag in words if flag.startswith('n') or flag.startswith('v')]
    #     return ' '.join(filtered_words)

    def BertEncoder(self, data, max_len=77):
        with torch.no_grad():
            encoded = self.tokenizer(data, 
                                     padding=True, 
                                     truncation=True, 
                                     return_tensors="pt", 
                                     max_length=max_len)
            input_ids = encoded['input_ids'].to(self.device)
            att_mask = encoded['attention_mask'].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=att_mask)
            tokens = outputs.last_hidden_state
        return tokens

    def encode_and_cache_data(self, dataset, batch_size=64):
        all_data = []
        vs_level_map = {}  # 用于存储 VS 与 Level 的映射关系
        cc_dept_map = {}  # 用于存储 CC_tokens 的唯一性映射关系
        text_to_dept_map = {}  # 用于存储编码前的文本和对应部门的映射关系

        for i in tqdm(range(0, len(dataset), batch_size), desc="Batch encoding"):
            batch = dataset[i:i + batch_size]
            vitalsigns, chiefcpt, labels_sety, labels_dept, lb_dept_cn = zip(*batch)

            # 使用 BERT 编码 CC 和 Department Tokens
            cc_tokens = self.BertEncoder(chiefcpt, max_len=20)
            lb_tokens = self.BertEncoder(lb_dept_cn, max_len=8)

            for j in range(len(batch)):
                vs = vitalsigns[j]
                level = labels_sety[j]
                dept_digit = labels_dept[j]
                cc_token_cls = cc_tokens[j][0]  
                vs_key = tuple(map(lambda x: int(x.item()), vs.nonzero(as_tuple=True)[0]))
                is_unique_level = 1 if len(vs_level_map.get(vs_key, [])) == 1 else 0
                cc_text = chiefcpt[j]  
                is_unique_dept = 1

                if cc_text in text_to_dept_map:
                    if text_to_dept_map[cc_text] != dept_digit:
                        is_unique_dept = 0
                else:
                    text_to_dept_map[cc_text] = dept_digit
                if vs_key not in vs_level_map:
                    vs_level_map[vs_key] = []
                vs_level_map[vs_key].append(level)

                all_data.append({
                    'VS': vs,
                    'Level': level,
                    'VS_UniqueLevel': is_unique_level, 
                    'CC_tokens': cc_tokens[j],
                    'Dept_digit': dept_digit,
                    'Dept_tokens': lb_tokens[j],
                    'CC_UniqueDept': is_unique_dept 
                })
        return all_data

    # def encode_and_cache_data(self, dataset, batch_size=64):
    #     import torch
    #     from itertools import combinations  # 用于生成词组组合

    #     all_data = []
    #     vs_level_map = {}
    #     cc_phrase_combinations_map = {}

    #     for i in tqdm(range(0, len(dataset), batch_size), desc="Batch encoding"):
    #         batch = dataset[i:i + batch_size]
    #         vitalsigns, chiefcpt, labels_sety, labels_dept, lb_dept_cn = zip(*batch)

    #         # 统计 VS 和 Level 的关系
    #         for vs, level in zip(vitalsigns, labels_sety):
    #             vs_key = tuple(vs.tolist()) if isinstance(vs, torch.Tensor) else tuple(vs)
    #             if vs_key not in vs_level_map:
    #                 vs_level_map[vs_key] = set()
    #             vs_level_map[vs_key].add(level)

    #         # 统计主诉中所有组合与科室的关系
    #         for cc_text, dept in zip(chiefcpt, labels_dept):
    #             words = cc_text.split()  # 按空格拆分为单词
    #             max_comb_len = len(words)  # 最大组合长度，可调整为固定值
    #             for comb_len in range(1, max_comb_len + 1):  # 生成所有长度的组合
    #                 for comb in combinations(words, comb_len):
    #                     if comb not in cc_phrase_combinations_map:
    #                         cc_phrase_combinations_map[comb] = set()
    #                     cc_phrase_combinations_map[comb].add(dept)

    #         # BERT 编码
    #         cc_tokens = self.BertEncoder(chiefcpt, max_len=20)
    #         lb_tokens = self.BertEncoder(lb_dept_cn, max_len=8)

    #         # 遍历当前批次，生成标记
    #         for j in range(len(batch)):
    #             vs_key = tuple(vitalsigns[j].tolist()) if isinstance(vitalsigns[j], torch.Tensor) else tuple(vitalsigns[j])
    #             is_unique_level = 1 if len(vs_level_map.get(vs_key, set())) == 1 else 0

    #             # 检查主诉中所有组合是否唯一对应科室
    #             words = chiefcpt[j].split()
    #             max_comb_len = len(words)  # 最大组合长度
    #             unique_dept_flag = 1  # 假设所有组合都唯一对应科室
    #             for comb_len in range(1, max_comb_len + 1):
    #                 for comb in combinations(words, comb_len):
    #                     if len(cc_phrase_combinations_map.get(comb, set())) > 1:  # 非唯一对应
    #                         unique_dept_flag = 0
    #                         break
    #                 if unique_dept_flag == 0:
    #                     break

    #             # 整理数据
    #             all_data.append({
    #                 'VS': vitalsigns[j],
    #                 'Level': labels_sety[j],
    #                 'CC_tokens': cc_tokens[j],
    #                 'Dept_digit': labels_dept[j],
    #                 'Dept_tokens': lb_tokens[j],
    #                 'VS_UniqueLevel': is_unique_level,
    #                 'CC_UniqueDept': unique_dept_flag,
    #             })

    #     return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]