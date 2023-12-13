import numpy as np

class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # 가중치와 편향 초기화
        self.weights = np.zeros(num_features)
        self.bias = 0

        # 경사 하강법을 통한 학습
        for _ in range(self.num_iterations):
            # 가중치와 편향에 대한 그래디언트 계산
            dw = np.zeros(num_features)
            db = 0

            for i in range(num_samples):
                if y[i] * (np.dot(X[i], self.weights) + self.bias) >= 1:
                    dw += 0
                    db += 0
                else:
                    dw += self.lambda_param * y[i] * X[i]
                    db += self.lambda_param * y[i]

            # 가중치와 편향 업데이트
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = np.sign(linear_model)
        return y_predicted
X_train = np.array([[2, 4], [1, 5], [4, 2], [5, 1]])
y_train = np.array([1, 1, -1, -1])

model = SVM(learning_rate=0.01, lambda_param=0.01, num_iterations=1000)
model.fit(X_train, y_train)

X_test = np.array([[3, 3], [1, 2]])
predictions = model.predict(X_test)

print(predictions)  # 예측 결과 출력
