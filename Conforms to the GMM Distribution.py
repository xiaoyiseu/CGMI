import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture
from scipy import stats
import matplotlib.pyplot as plt
from module.StructureEncoder import StructureDataEncoder


def load_and_clean_data(path, max_rows=None):
    """
    Load raw data into a DataFrame, select relevant columns,
    replace placeholders, drop missing rows, and reset index.
    """
    raw = pd.read_csv(path, engine="python")
    cols = [
        '性别', '出生日期', '分诊时间', '到院方式', '分诊印象',
        'T℃', 'P(次/分)', 'R(次/分)', 'BP(mmHg)', 'SpO2',
        '级别', '去向'
    ]
    df = raw[cols].copy()
    df.replace('空值', np.nan, inplace=True)
    df.dropna(inplace=True)
    if max_rows is not None:
        df = df.iloc[:max_rows]
    df.reset_index(drop=True, inplace=True)
    return df


def encode_structure(df):
    """
    Apply StructureDataEncoder to each vital sign and returns stacked tensor and feature names.
    """
    encoder = StructureDataEncoder()
    specs = {
        'MOA': ('到院方式', encoder.Arr_way),
        'sex': ('性别', encoder.Gender),
        'age': ('出生日期', encoder.Age),
        'temp': ('T℃', encoder.Temperature),
        'pulse': ('P(次/分)', encoder.Pulse),
        'resp': ('R(次/分)', encoder.Respiration),
        'bp': ('BP(mmHg)', encoder.BloodPressure),
        'spo2': ('SpO2', encoder.SpO2)
    }
    features = []
    for name, (col, fn) in specs.items():
        values = df[col].apply(fn).values
        features.append(values)
    arr = np.stack(features).T  # shape (n_samples, n_features)
    return torch.tensor(arr), list(specs.keys())


def map_labels(df):
    """
    Map '级别' and '去向' to integer labels.
    """
    level_map = {v: i for i, v in enumerate(df['级别'].unique())}
    dept_map = {v: i for i, v in enumerate(df['去向'].unique())}
    y1 = df['级别'].map(level_map).astype(int)
    y2 = df['去向'].map(dept_map).astype(int)
    labels = pd.DataFrame({'Severity': y1, 'Department': y2})
    labels.reset_index(drop=True, inplace=True)
    return labels


def fit_gmm(data_arr, max_components=6, random_state=0):
    """
    Fit 1D GMM on numpy array data_arr, choose best by BIC.
    """
    bic_scores, models = [], []
    n = data_arr.shape[0]
    ks = range(1, min(max_components, n) + 1)
    for k in ks:
        gm = GaussianMixture(n_components=k, random_state=random_state)
        gm.fit(data_arr.reshape(-1, 1))
        bic_scores.append(gm.bic(data_arr.reshape(-1, 1)))
        models.append(gm)
    best = models[int(np.argmin(bic_scores))]
    return best


def analyze_vitals(df, feature_names, tasks, output_csv):
    """
    Loop over tasks and features: compute skewness, GMM, plot, and save summary.
    """
    records = []
    for col_task, task_name in tasks.items():
        classes = sorted(df[col_task].dropna().unique())
        for cls in classes:
            subset = df[df[col_task] == cls]
            print(f"Task={task_name}, Class={cls}, Samples={len(subset)}")
            for feat in feature_names:
                vals = subset[feat].dropna()
                vals = vals[np.isfinite(vals)]
                if len(vals) < 2:
                    continue
                skew = vals.skew()
                data_arr = vals.values
                gm = fit_gmm(data_arr)
                k_opt = gm.n_components

                # Plot
                plt.figure(figsize=(6, 3))
                plt.hist(data_arr, bins=25, density=True, alpha=0.5, label='Hist')
                xs = np.linspace(data_arr.min(), data_arr.max(), 200)
                if np.unique(data_arr).size > 1:
                    try:
                        kde = stats.gaussian_kde(data_arr)
                        plt.plot(xs, kde(xs), lw=2, label='KDE')
                    except:
                        pass
                logprob = gm.score_samples(xs.reshape(-1, 1))
                plt.plot(xs, np.exp(logprob), lw=2, label=f'GMM(K={k_opt})')
                plt.title(f"{task_name}={cls}, {feat}, skew={skew:.2f}, K={k_opt}")
                plt.legend()
                plt.tight_layout()
                plt.show()

                records.append({
                    'task': task_name,
                    'class': cls,
                    'feature': feat,
                    'skewness': skew,
                    'optimal_gmm_components': k_opt
                })
    result_df = pd.DataFrame(records)
    result_df.to_csv(output_csv, index=False)
    print(f"Saved analysis to {output_csv}")


def main():
    df = load_and_clean_data(DATA_PATH, max_rows=MAX_ROWS)
    env_tensor, feature_names = encode_structure(df)
    env_df = pd.DataFrame(env_tensor.numpy(), columns=feature_names)
    label_df = map_labels(df)
    combined = pd.concat([env_df, label_df], axis=1)

    tasks = {'Severity': 'Severity', 'Department': 'Department'}
    analyze_vitals(combined, feature_names, tasks, OUTPUT_CSV)


if __name__ == "__main__":
    DATA_PATH = "./data/AllRightData.txt"  
    OUTPUT_CSV = "vital_signs_gmm_analysis.csv"
    MAX_ROWS = None  
    main()
    