import numpy as np
import pandas as pd

def kmeans_clustering(data, k, max_iterations=100):
    # 데이터프레임에서 데이터 추출
    X = data.values

    # 랜덤하게 초기 중심점 선택
    np.random.seed(0)
    center_indices = np.random.choice(range(len(X)), k, replace=False)
    centers = X[center_indices]

    for _ in range(max_iterations):
        # 각 데이터 포인트에 대해 가장 가까운 중심점 찾기
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=-1), axis=-1)

        # 중심점 업데이트
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # 수렴 여부 확인
        if np.all(centers == new_centers):
            break

        centers = new_centers

    return labels, centers

# 예시 데이터프레임 생성
data = pd.DataFrame({
    'x': [1, 1.5, 3, 5, 3.5, 4.5, 3.5],
    'y': [1, 2, 4, 7, 5, 5, 4.5]
})

# k-means 클러스터링 수행
k = 2
labels, centers = kmeans_clustering(data, k)

# 결과 출력
print("클러스터링 결과:")
for i in range(k):
    cluster_points = data.loc[labels == i]
    print(f"클러스터 {i+1}:")
    print(cluster_points)
    print()

print("중심점:")
print(centers)
