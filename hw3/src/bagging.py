import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier
import pandas as pd


class BaggingClassifier:
    def __init__(self, input_dim: int) -> None:
        # create 10 learners, dont change.
        self.learners = [WeakClassifier(input_dim=input_dim) for _ in range(10)]

    def fit(self, X_train, y_train, num_epochs: int = 200, learning_rate: float = 0.01):
        """
        使用 Bootstrap 訓練每個基礎學習器
        """
        n_samples = len(y_train)
        losses_of_models = []

        # 將輸入轉換為 numpy array（如果是 DataFrame）
        X_train_arr = (
            X_train.values.astype("float32")
            if isinstance(X_train, pd.DataFrame)
            else X_train
        )

        # 訓練每個基礎學習器
        for model in self.learners:
            # Bootstrap 採樣
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_batch = torch.FloatTensor(X_train_arr[indices])
            y_batch = torch.FloatTensor(y_train[indices])

            # 初始化優化器
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            model_losses = []

            # 訓練單個模型
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()

                # 計算損失
                loss = nn.BCEWithLogitsLoss()(outputs, y_batch)
                loss.backward()
                optimizer.step()

                model_losses.append(loss.item())

            losses_of_models.append(model_losses)

        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """
        使用投票機制進行預測
        """
        # 轉換輸入
        if isinstance(X, pd.DataFrame):
            X = X.values.astype("float32")
        X_tensor = torch.FloatTensor(X)

        predictions = []
        probas = []

        # 收集所有模型的預測
        with torch.no_grad():
            for model in self.learners:
                outputs = model(X_tensor).squeeze()
                probs = torch.sigmoid(outputs).numpy()
                preds = (probs > 0.5).astype(int)
                predictions.append(preds)
                probas.append(probs)

        # 計算最終預測（多數投票）
        predictions = np.array(predictions)
        final_predictions = (np.mean(predictions, axis=0) > 0.5).astype(int)

        # 計算平均概率
        final_probas = np.mean(probas, axis=0)
        learner_probs = []
        with torch.no_grad():
            for model in self.learners:
                outputs = model(X_tensor).squeeze()
                probs = torch.sigmoid(outputs).numpy()
                learner_probs.append(probs)

        return final_predictions, final_probas, learner_probs

    def compute_feature_importance(self) -> t.Sequence[float]:
        """
        計算特徵重要性（所有模型的平均）
        """
        importance = np.zeros(self.learners[0].layer1.weight.shape[1])

        for model in self.learners:
            with torch.no_grad():
                # 獲取每個模型的特徵重要性
                weights = model.get_weights().numpy()
                importance += np.abs(weights)

        # 正規化
        return importance / (len(self.learners) * np.sum(importance))
