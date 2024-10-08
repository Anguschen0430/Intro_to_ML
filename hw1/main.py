import numpy as np
import pandas as pd
from loguru import logger
import random
import matplotlib.pyplot as plt


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        # bias
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        # add bias
        y = y.ravel()
        X_b = np.c_[np.ones((len(X), 1)), X]
        X_T = X_b.T
        w = np.linalg.inv(X_T @ X_b) @ X_T @ y
        # print(w)
        self.intercept = w[0]
        self.weights = w[1:]
        # print(f"{self.weights}, {self.intercept}")

    def predict(self, X):
        # print(X)
        return X @ self.weights + self.intercept


class LinearRegressionGradientdescent(LinearRegressionBase):
    def __init__(self):
        super().__init__()
        self.losses = []

    def fit(self, X, y, learning_rate: float = 0.001, epochs: int = 1000):
        m, n = X.shape
        # y = y.ravel()
        # print(f'{y.shape=}')
        self.weights = np.random.rand(n)
        self.intercept = random.random()

        X, mean, std = self.normalization(X)

        # Remove unused variable 'best'
        for epoch in range(epochs):
            y_predict = X @ self.weights + self.intercept
            # print(y_predict)
            loss = compute_mse(y_predict, y)
            self.losses.append(loss)

            y_predict = y_predict.reshape(-1, 1)
            y = y.reshape(-1, 1)

            dw = (1 / m) * (X.T @ (y_predict - y))  # Add spaces around division
            db = (1 / m) * np.sum(y_predict - y)    # Add spaces around division
            self.weights = self.weights.reshape(-1, 1)
            # print(f'{dw.shape=}, {self.weights.shape=}')
            self.weights -= learning_rate * dw
            self.intercept -= learning_rate * db

            if epoch % 10000 == 0:
                logger.info(f'Epoch {epoch}, Loss: {loss:.4f}')

        self.weights = self.weights / std.reshape(-1, 1)
        self.intercept -= np.sum((self.weights * mean.reshape(-1, 1)), axis=0)

        self.intercept = self.intercept[0]
        return self.losses

    def predict(self, X):
        return X @ self.weights + self.intercept

    def plot_learning_curve(self, losses):
        plt.plot(range(len(losses)), losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Learning Curve')
        plt.grid(True)
        plt.show()

    def normalization(self, X):
        # Normalize the data, also return the mean and std of each column.
        # Return the normalized data, mean and std of each column.
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X = (X - mean) / std
        return X, mean, std


def compute_mse(prediction, ground_truth):
    return np.mean(((prediction - ground_truth) ** 2))


def main():
    train_df = pd.read_csv('./train.csv')
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy().reshape(-1, 1)

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    LR_GD = LinearRegressionGradientdescent()
    losses = LR_GD.fit(train_x, train_y, learning_rate=2e-4, epochs=50000)
    LR_GD.plot_learning_curve(losses)
    LR_GD.weights = LR_GD.weights.ravel()
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    test_df = pd.read_csv('./test.csv')
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy().reshape(-1, 1)

    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_cf = y_preds_cf.reshape(-1, 1)
    y_preds_gd = y_preds_gd.reshape(-1, 1)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).mean()
    logger.info(f'Mean prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = (np.abs(mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()
