import numpy as np
import pandas as pd

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = {}

    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def find_best_split(self, X, y):
        num_samples, num_features = X.shape
        best_info_gain = -1
        best_feature_idx = -1
        best_threshold = None

        parent_entropy = self.entropy(y)

        for feature_idx in range(num_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            for threshold in unique_values:
                left_indices = np.where(feature_values <= threshold)[0]
                right_indices = np.where(feature_values > threshold)[0]

                left_entropy = self.entropy(y[left_indices])
                right_entropy = self.entropy(y[right_indices])

                info_gain = parent_entropy - (len(left_indices) / num_samples) * left_entropy - (len(right_indices) / num_samples) * right_entropy

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    def build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        classes, counts = np.unique(y, return_counts=True)

        # 노드에 속한 클래스가 하나일 경우
        if len(classes) == 1:
            return classes[0]

        # 모든 특성을 사용했을 때 노드에 속한 샘플이 하나일 경우
        if num_samples == 1:
            return classes[np.argmax(counts)]

        # 트리의 최대 깊이에 도달한 경우
        if self.max_depth is not None and depth >= self.max_depth:
            return classes[np.argmax(counts)]

        best_feature_idx, best_threshold = self.find_best_split(X, y)

        # 분할 조건을 만족하지 못하는 경우
        if best_feature_idx == -1:
            return classes[np.argmax(counts)]

        left_indices = np.where(X[:, best_feature_idx] <= best_threshold)[0]
        right_indices = np.where(X[:, best_feature_idx] > best_threshold)[0]

        subtree = {}
        subtree['feature_idx'] = best_feature_idx
        subtree['threshold'] = best_threshold
        subtree['left'] = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        subtree['right'] = self.build_tree(X[right_indices], y[right_indices], depth + 1)

        return subtree

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_sample(self, x, tree):
        if not isinstance(tree, dict):
            return tree

        feature_idx = tree['feature_idx']
        threshold = tree['threshold']

        if x[feature_idx] <= threshold:
            return self.predict_sample(x, tree['left'])
        else:
            return self.predict_sample(x, tree['right'])

    def predict(self, X):
        num_samples = X.shape[0]
        y_pred = []

        for i in range(num_samples):
            y_pred.append(self.predict_sample(X[i], self.tree))

        return np.array(y_pred)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터프레임으로 변환
df_train = pd.DataFrame(X_train, columns=iris.feature_names)
df_train['target'] = y_train

df_test = pd.DataFrame(X_test, columns=iris.feature_names)

# 의사 결정 트리 모델 학습 및 예측
model = DecisionTree(max_depth=3)
model.fit(df_train.iloc[:, :-1].values, df_train.iloc[:, -1].values)
predictions = model.predict(df_test.values)

print(predictions)  # 예측 결과 출력
