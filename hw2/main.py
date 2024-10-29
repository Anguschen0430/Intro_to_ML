import typing as t
import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate: float = 5e-3, num_iterations: int = 100, seed=65):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None
        self.seed = seed

    def fit(
        self,
        inputs: npt.NDArray[float],  # (n_samples, n_features)
        targets: t.Sequence[int],  # (n_samples, )
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        np.random.seed(self.seed)
        # Initialization with random values
        n_samples, n_features = inputs.shape
        self.weights = np.random.uniform(-0.5, 0.5, n_features)  # Initialize weights within -0.5 to 0.5
        self.intercept = np.random.uniform(-0.5, 0.5)  # Initialize intercept within -0.5 to 0.5

        # Gradient descent
        for _ in range(self.num_iterations):
            y = np.dot(inputs, self.weights) + self.intercept
            y_pred = self.sigmoid(y)
            dw = np.dot(inputs.T, (y_pred - targets)) / n_samples
            db = np.sum(y_pred - targets) / n_samples
            self.weights -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db  # Consistent learning rate for weights and intercept

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Tuple[t.Sequence[np.float64], t.Sequence[int]]:
        """
        Predict class labels for samples in inputs.
        Returns:
        1. sample probability of being class_1
        2. sample predicted class (0 or 1)
        """
        y = np.dot(inputs, self.weights) + self.intercept
        y_pred = self.sigmoid(y)
        y_pred_classes = (y_pred > 0.5).astype(int)
        return y_pred, y_pred_classes

    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))


class FLD:
    """Implement FLD
    You can add arguments as you need,
    but don't modify those already exist variables.
    """
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        # Separate inputs by class
        X0 = inputs[targets == 0]
        X1 = inputs[targets == 1]

        # Compute mean vectors
        self.m0 = np.mean(X0, axis=0)
        self.m1 = np.mean(X1, axis=0)

        # Compute within-class scatter matrix Sw
        S0 = np.dot((X0 - self.m0).T, (X0 - self.m0))
        S1 = np.dot((X1 - self.m1).T, (X1 - self.m1))
        self.sw = S0 + S1

        # Compute between-class scatter matrix Sb
        mean_diff = (self.m1 - self.m0).reshape(-1, 1)
        self.sb = np.dot(mean_diff, mean_diff.T)

        # Compute the optimal projection vector w
        self.w = np.linalg.pinv(self.sw).dot(self.m1 - self.m0)
        self.w = self.w / np.linalg.norm(self.w)  # Normalize w

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Sequence[int]:
        # Project inputs onto w
        projections = inputs.dot(self.w)

        # Compute threshold
        threshold = (self.m0.dot(self.w) + self.m1.dot(self.w)) / 2

        # Classify based on threshold
        predictions = (projections >= threshold).astype(int)
        return predictions

    def plot_projection(self, inputs: npt.NDArray[float], targets: t.Sequence[int], dataset_name="Testing"):
        y_pred = self.predict(inputs)

        # Calculate unit vector along w
        w_unit = self.w / np.linalg.norm(self.w)

        # Choose a point on the line (mean of inputs)
        x0 = np.mean(inputs, axis=0)

        # Calculate projections onto the discriminant vector
        projections = ((inputs - x0) @ w_unit)  # scalar projection
        projected_points = x0 + np.outer(projections, w_unit)  # vector projections

        # Plot original data points with color based on true class
        plt.scatter(inputs[targets == 0][:, 0], inputs[targets == 0][:, 1], color="red", label="Class 0")
        plt.scatter(inputs[targets == 1][:, 0], inputs[targets == 1][:, 1], color="blue", label="Class 1")

        # Plot projected points and lines
        for i in range(len(inputs)):
            plt.plot([inputs[i, 0], projected_points[i, 0]], [inputs[i, 1], projected_points[i, 1]],
                     'gray', linestyle="--", alpha=0.5)
            if y_pred[i] == 0:
                plt.scatter(projected_points[i, 0], projected_points[i, 1], color='red', marker='x')
            else:
                plt.scatter(projected_points[i, 0], projected_points[i, 1], color='blue', marker='x')

        # Plot the projection line
        line_extent = np.max(np.linalg.norm(inputs - x0, axis=1)) * 1.5
        line_points = np.array([x0 - w_unit * line_extent, x0 + w_unit * line_extent])

        # Calculate slope and intercept
        self.slope = w_unit[1] / w_unit[0] if w_unit[0] != 0 else float('inf')
        self.intercept = x0[1] - self.slope * x0[0]

        plt.plot(line_points[:, 0], line_points[:, 1], 'k-', label="Projection Line")
        plt.title(f"{dataset_name} Set Projection on FLD\nAccuracy: {accuracy_score(targets, y_pred):.4f}")

        # Display slope and intercept on the plot
        plt.text(-2, 1.3, f"Slope: {self.slope:.5f}", fontsize=10)
        plt.text(-2, 1.1, f"Intercept: {self.intercept:.5f}", fontsize=10)

        plt.legend()
        plt.axis('equal')
        plt.show()


def compute_auc(y_trues, y_preds):
    return roc_auc_score(y_trues, y_preds)


def accuracy_score(y_trues, y_preds):
    correct_predictions = sum(1 for y_true, y_pred in zip(y_trues, y_preds) if y_true == y_pred)
    accuracy = correct_predictions / len(y_trues)
    return accuracy


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=1e-2,  # You can modify the parameters as you want
        num_iterations=1000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['10', '20']  # Don't modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    # 1) Fit the FLD model
    FLD_.fit(x_train, y_train)

    # 2) Make prediction
    y_pred = FLD_.predict(x_test)

    # 3) Compute the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1} of {cols=}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')

    # Plot the projection
    FLD_.plot_projection(x_test, y_test)


if __name__ == '__main__':
    main()
