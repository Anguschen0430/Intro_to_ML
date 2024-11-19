import pandas as pd
from loguru import logger
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from src import AdaBoostClassifier, BaggingClassifier, DecisionTree
from src.utils import preprocess, plot_learners_roc


def plot_loss_curves(losses, fpath='loss_curves.png'):
    """
    繪製損失曲線
    Args:
        losses: 每個分類器的損失歷史
        fpath: 圖片保存路徑
    """
    plt.figure(figsize=(12, 6))
    for i, loss in enumerate(losses):
        plt.plot(loss, label=f'Learner {i + 1}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss of Weak Learners')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()


def plot_feature_importance(importance, feature_names, title, fpath):
    plt.figure(figsize=(12, 6))
    # Sort features by importance in descending order
    sorted_idx = importance.argsort()[::-1]
    sorted_importance = importance[sorted_idx]
    sorted_features = [feature_names[i] for i in sorted_idx]
    # Plot horizontal bars
    plt.barh(range(len(sorted_importance)), sorted_importance)
    # Customize plot
    plt.yticks(range(len(sorted_importance)), sorted_features)
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()


def plot_feature_importance2(clf, feature_names, title, fpath):
    """
    繪製特徵重要性圖
    """
    importances = clf.compute_feature_importances(feature_names)
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.barh(range(len(importances)), importances[indices])
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()


def main():
    """
    Note:
    1) Part of line should not be modified.
    2) You should implement the algorithm by yourself.
    3) You can change the I/O data type as you need.
    4) You can change the hyperparameters as you want.
    5) You can add/modify/remove args in the function, but you need to fit the requirements.
    6) When plot the feature importance, the tick labels of one of the axis should be feature names.
    """
    random.seed(777)  # DON'T CHANGE THIS LINE
    torch.manual_seed(777)  # DON'T CHANGE THIS LINE
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    X_train = train_df.drop(['target'], axis=1)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    X_test = test_df.drop(['target'], axis=1)
    y_test = test_df['target'].to_numpy()

    # (TODO): Implement you preprocessing function.
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    """
    (TODO): Implement your ensemble methods.
    1. You can modify the hyperparameters as you need.
    2. You must print out logs (e.g., accuracy) with loguru.
    """
    feature_names = X_train.columns.tolist()

    # AdaBoost
    clf_adaboost = AdaBoostClassifier(
        input_dim=X_train.shape[1]
    )
    losses = clf_adaboost.fit(
        X_train,
        y_train,
        num_epochs=30,
        learning_rate=1e-4
    )
    plot_loss_curves(losses, 'adaboost_loss.png')
    y_pred_classes, y_pred_probs, learner_probs = clf_adaboost.predict_learners(X_test)
    accuracy_ = np.mean(y_pred_classes == y_test)
    logger.info(f'AdaBoost - Accuracy: {accuracy_:.4f}')

    # print(type(y_pred_probs), y_pred_probs)
    plot_learners_roc(
        y_preds=learner_probs,
        y_trues=y_test,
        fpath='adaboost_roc.png'
    )

    # Draw the feature importance
    feature_importance = clf_adaboost.compute_feature_importance()
    plot_feature_importance(
        feature_importance,
        feature_names,
        'AdaBoost Feature Importance',
        'adaboost_feature_importance.png'
    )

    # Bagging
    clf_bagging = BaggingClassifier(
        input_dim=X_train.shape[1],
    )
    losses = clf_bagging.fit(
        X_train,
        y_train,
        num_epochs=20,
        learning_rate=0.01
    )
    plot_loss_curves(losses, 'bagging_loss.png')
    y_pred_classes, y_pred_probs, learner_probs = clf_bagging.predict_learners(X_test)
    accuracy_ = np.mean(y_pred_classes == y_test)
    logger.info(f'Bagging - Accuracy: {accuracy_:.4f}')

    plot_learners_roc(
        y_preds=learner_probs,
        y_trues=y_test,
        fpath='bagging_roc.png'
    )
    feature_importance = clf_bagging.compute_feature_importance()
    # Draw the feature importance
    plot_feature_importance(
        feature_importance,
        feature_names,
        'Bagging Feature Importance',
        'bagging_feature_importance.png'
    )

    # Decision Tree
    clf_tree = DecisionTree(
        max_depth=7  # You can set the value as desired
    )
    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)
    accuracy_ = np.mean(y_pred_classes == y_test)
    logger.info(f'DecisionTree - Accuracy: {accuracy_:.4f}')
    plot_feature_importance2(
        clf_tree,
        feature_names,
        'Decision Tree Feature Importance',
        'decision_tree_feature_importance.png'
    )


if __name__ == '__main__':
    main()
