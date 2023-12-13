import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # 가중치 초기화
        self.weights = np.zeros(num_features)
        self.bias = 0

        # 경사 하강법을 통한 학습
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # 손실 함수의 그래디언트 계산
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # 가중치 업데이트
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

X_train = np.array([[2, 4], [1, 5], [4, 2], [5, 1]])
y_train = np.array([0, 0, 1, 1])

model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
model.fit(X_train, y_train)

X_test = np.array([[3, 3], [1, 2]])
predictions = model.predict(X_test)

print(predictions)  # 예측 결과 출력
