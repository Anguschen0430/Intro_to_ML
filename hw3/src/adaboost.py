import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier
import pandas as pd


class AdaBoostClassifier:
    def __init__(
        self, input_dim: int, num_learners: int = 10, hidden_dim: int = 16
    ) -> None:
        self.sample_weights = None
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim, hidden_dim=hidden_dim)
            for _ in range(10)
        ]
        self.alphas = []

    def fit(
        self, X_train, y_train, num_epochs: int = 500, learning_rate: float = 0.001
    ):
        """
        訓練 AdaBoost 分類器
        """
        n_samples = len(y_train)
        self.sample_weights = np.ones(n_samples) / n_samples
        losses_of_models = []

        # 確保數據類型正確
        X_train_arr = X_train.values.astype("float32")
        X_train_tensor = torch.FloatTensor(X_train_arr)
        y_train_tensor = torch.FloatTensor(y_train)
        # 訓練每個弱分類器
        for model in self.learners:
            model_losses = []
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

            # 訓練單個模型
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                outputs = model(X_train_tensor).squeeze()

                # 計算加權損失
                weighted_loss = torch.mean(
                    torch.tensor(self.sample_weights, dtype=torch.float32)
                    * nn.BCEWithLogitsLoss(reduction="none")(outputs, y_train_tensor)
                )

                weighted_loss.backward()
                optimizer.step()
                model_losses.append(weighted_loss.item())

            losses_of_models.append(model_losses)

            # 計算預測結果
            with torch.no_grad():
                predictions = (
                    (torch.sigmoid(model(X_train_tensor)).squeeze() > 0.5)
                    .float()
                    .numpy()
                )

            # 計算錯誤率和模型權重
            weighted_error = np.sum(self.sample_weights * (predictions != y_train))
            epsilon = max(1e-10, min(weighted_error, 1 - 1e-10))
            alpha = 0.5 * np.log((1 - epsilon) / epsilon)
            self.alphas.append(alpha)

            # 更新樣本權重
            y_train_mod = np.where(y_train == 0, -1, 1)
            predictions_mod = np.where(predictions == 0, -1, 1)
            self.sample_weights *= np.exp(-alpha * y_train_mod * predictions_mod)
            self.sample_weights /= np.sum(self.sample_weights)

        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """Implement your code here"""
        if isinstance(X, pd.DataFrame):
            X = X.values.astype("float32")
        X_tensor = torch.FloatTensor(X)
        weighted_preds = np.zeros(len(X))
        probas = np.zeros(len(X))

        # 結合所有弱分類器的預測
        for model, alpha in zip(self.learners, self.alphas):
            with torch.no_grad():
                pred_probs = torch.sigmoid(model(X_tensor)).squeeze().numpy()
                predictions = (pred_probs > 0.5).astype(int)
                predictions_mod = np.where(predictions == 0, -1, 1)
                weighted_preds += alpha * predictions_mod
                probas += alpha * pred_probs

        # 計算最終預測
        final_predictions = (weighted_preds > 0).astype(int)
        final_probas = probas / np.sum(self.alphas)  # 正規化機率

        # 收集每個弱學習器的預測概率
        learner_probs = []
        for model in self.learners:
            with torch.no_grad():
                pred_probs = torch.sigmoid(model(X_tensor)).squeeze().numpy()
                learner_probs.append(pred_probs)

        return final_predictions, final_probas, learner_probs

    def compute_feature_importance(self) -> t.Sequence[float]:
        """
        計算特徵重要性 - 修改為處理兩層模型

        Returns:
            importance: 每個特徵的重要性分數
        """
        # 使用 get_weights 方法獲取組合後的權重
        importance = np.zeros(self.learners[0].layer1.weight.shape[1])

        for model, alpha in zip(self.learners, self.alphas):
            with torch.no_grad():
                # 使用 get_weights 方法
                weights = model.get_weights().numpy()
                importance += alpha * np.abs(weights)

        # 正規化特徵重要性
        return importance / np.sum(importance)
