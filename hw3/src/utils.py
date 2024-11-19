import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

# 移除了未使用的導入：typing as t 和 LabelEncoder


def preprocess(df: pd.DataFrame):
    """
    預處理函數，確保所有輸出都是數值型
    """
    df_processed = df.copy()

    # 1. 處理類別特徵
    # 1.1 二元類別編碼
    binary_features = ['person_gender', 'previous_loan_defaults_on_file']
    for feature in binary_features:
        if feature in df_processed.columns:
            # 將字串轉換為數值
            df_processed[feature] = df_processed[feature].astype('category').cat.codes

    # 1.2 多類別 one-hot 編碼
    categorical_features = ['person_education', 'person_home_ownership', 'loan_intent']
    df_processed = pd.get_dummies(df_processed, columns=categorical_features, dummy_na=True)

    # 2. 處理數值特徵
    numeric_features = [
        'person_age', 'person_income', 'person_emp_exp',
        'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_cred_hist_length', 'credit_score'
    ]

    # 2.1 處理缺失值
    for feature in numeric_features:
        if feature in df_processed.columns:
            mean_value = df_processed[feature].mean()
            df_processed[feature] = df_processed[feature].fillna(mean_value)

    # 2.2 標準化
    scaler = StandardScaler()
    df_processed[numeric_features] = scaler.fit_transform(df_processed[numeric_features])

    # 確保所有列都是 float32 類型
    df_processed = df_processed.astype('float32')

    return df_processed


class WeakClassifier(nn.Module):
    """
    Use pyTorch to implement a 1 ~ 2 layers model.
    Here, for example:
        - Linear(input_dim, 1) is a single-layer model.
        - Linear(input_dim, k) -> Linear(k, 1) is a two-layer model.

    No non-linear activation allowed.
    """

    def __init__(self, input_dim, hidden_dim=16):
        super(WeakClassifier, self).__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def get_weights(self):
        """
        獲取模型權重用於特徵重要性計算
        修正矩陣乘法維度問題
        """
        # w1 shape: (hidden_dim, input_dim)
        # w2 shape: (1, hidden_dim)
        w1 = self.layer1.weight  # shape: (hidden_dim, input_dim)
        w2 = self.layer2.weight  # shape: (1, hidden_dim)

        # 計算組合權重
        combined_weights = torch.zeros(w1.shape[1])  # shape: (input_dim,)

        # 計算每個輸入特徵的重要性
        for i in range(w1.shape[1]):  # 對每個輸入特徵
            feature_importance = torch.abs(w2 @ w1[:, i].unsqueeze(1)).item()
            combined_weights[i] = feature_importance

        return combined_weights


def accuracy_score(y_trues, y_preds) -> float:
    return np.mean(y_trues == y_preds)


def entropy_loss(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)


def plot_learners_roc(y_preds, y_trues, fpath='./learners_roc_plot.png'):
    """
    繪製每個分類器的 ROC 曲線並顯示 AUC。

    Args:
        y_preds (list of sequences): 每個 learner 的預測值列表。
        y_trues (sequence): 實際標籤。
        fpath (str): 圖片保存路徑。
    """
    plt.figure(figsize=(10, 8))

    for i, y_pred in enumerate(y_preds):
        # 確認 y_pred 是一維數組，若是多維則取其中一列
        if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
            y_pred = y_pred[:, 1]  # 假設取第2列作為陽性類別的機率，根據需要調整

        fpr, tpr, _ = roc_curve(y_trues, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            lw=2,
            label=f'Learner {i + 1} (AUC = {roc_auc:.4f})'
        )

    # 繪製對角線（隨機猜測的基準線）
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Random Guess')

    # 設置圖表屬性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) for Multiple Learners')
    plt.legend(loc="lower right")
    plt.grid(True)

    # 保存圖片
    plt.savefig(fpath)
    plt.close()


def __main__():
    # 讀取數據
    df = pd.read_csv('/Users/angus/Desktop/college/intro_ML/hw3/train.csv')

    # 顯示基本資訊
    print("\n基本資訊：")
    print(df.info())

    print("\n前5筆數據：")
    print(df.head())

    print("\n數據統計：")
    print(df.describe())

    print("\n類別特徵的唯一值：")
    categorical_features = ['person_gender', 'person_education', 'person_home_ownership',
                            'loan_intent', 'previous_loan_defaults_on_file']
    for feature in categorical_features:
        print(f"\n{feature} 的唯一值：")
        print(df[feature].value_counts())

    # 檢查缺失值
    print("\n缺失值統計：")
    print(df.isnull().sum())

    # 使用預處理函數
    processed_df = preprocess(df)

    print("\n處理後的數據形狀：")
    print(processed_df.shape)

    print("\n處理後的特徵名稱：")
    print(processed_df.columns.tolist())

    # 檢查處理後的數值特徵統計
    numeric_features = ['person_age', 'person_income', 'person_emp_exp',
                        'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                        'cb_person_cred_hist_length', 'credit_score']
    print("\n處理後的數值特徵統計：")
    print(processed_df[numeric_features].describe())


if __name__ == '__main__':
    __main__()
