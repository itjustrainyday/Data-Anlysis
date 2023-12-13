import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self):
        self.class_probabilities = {}
        self.feature_probabilities = {}

    def fit(self, X, y):
        num_samples, num_features = X.shape
        classes = np.unique(y)

        # 클래스별 사전 확률 계산
        for class_label in classes:
            self.class_probabilities[class_label] = np.sum(y == class_label) / num_samples

        # 특성별 조건부 확률 계산
        for feature_idx in range(num_features):
            feature_values = np.unique(X[:, feature_idx])
            self.feature_probabilities[feature_idx] = {}

            for class_label in classes:
                self.feature_probabilities[feature_idx][class_label] = {}

                for feature_value in feature_values:
                    # 특성값이 feature_value일 때의 조건부 확률 계산
                    numerator = np.sum((X[:, feature_idx] == feature_value) & (y == class_label))
                    denominator = np.sum(y == class_label)
                    self.feature_probabilities[feature_idx][class_label][feature_value] = numerator / denominator

    def predict(self, X):
        num_samples, num_features = X.shape
        y_pred = []

        for i in range(num_samples):
            probabilities = {}

            for class_label, class_probability in self.class_probabilities.items():
                probability = class_probability

                for feature_idx in range(num_features):
                    feature_value = X[i, feature_idx]
                    conditional_probability = self.feature_probabilities[feature_idx][class_label][feature_value]
                    probability *= conditional_probability

                probabilities[class_label] = probability

            # 가장 높은 확률을 가진 클래스 선택
            predicted_class = max(probabilities, key=probabilities.get)
            y_pred.append(predicted_class)

        return y_pred
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

# 나이브 베이즈 모델 학습 및 예측
model = NaiveBayes()
model.fit(df_train.iloc[:, :-1].values, df_train.iloc[:, -1].values)
predictions = model.predict(df_test.values)

print(predictions)  # 예측 결과 출력
