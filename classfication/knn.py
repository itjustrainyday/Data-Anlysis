import numpy as np

def kmeans(X, k, max_iters=100):
    # 데이터 포인트들 중에서 랜덤하게 k개를 선택하여 초기 중심점으로 설정
    centers = X[np.random.choice(range(X.shape[0]), size=k, replace=False)]

    for _ in range(max_iters):
        # 각 데이터 포인트를 가장 가까운 중심점의 클러스터에 할당
        cluster_labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)

        # 각 클러스터의 평균을 계산하여 새로운 중심점으로 설정
        new_centers = np.array([X[cluster_labels == i].mean(axis=0) for i in range(k)])

        # 중심점이 더 이상 변하지 않으면 종료
        if np.all(centers == new_centers):
            break

        centers = new_centers

    return cluster_labels, centers

import numpy as np

def kmeans(X, K):
    # 중심점 초기화
    centers = X[np.random.choice(range(X.shape[0]), size = K, replace = False)]

    while True:
        # 각 데이터 포인트에 가장 가까운 중심점을 할당
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis = 2), axis = 1)

        # 새로운 중심점을 계산
        new_centers = np.array([X[labels == i].mean(axis = 0) for i in range(K)])

        # 중심점이 변하지 않으면 종료
        if np.all(centers == new_centers):
            break
            
        centers = new_centers

    return labels, centers
