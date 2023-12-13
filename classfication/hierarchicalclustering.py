import numpy as np
import pandas as pd

def hierarchical_clustering(data):
    # 데이터프레임에서 데이터 추출
    X = data.values

    # 각 데이터 포인트 간의 거리 계산
    distances = np.linalg.norm(X[:, np.newaxis] - X, axis=-1)

    # 초기 클러스터 설정: 각 데이터 포인트를 하나의 클러스터로 간주
    clusters = [[i] for i in range(len(X))]

    # 클러스터 병합
    while len(clusters) > 1:
        # 가장 가까운 두 클러스터 찾기
        min_distance = np.inf
        merge_indices = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = np.min(distances[clusters[i], :][:, clusters[j]])
                if distance < min_distance:
                    min_distance = distance
                    merge_indices = (i, j)

        # 클러스터 병합
        merged_cluster = clusters[merge_indices[0]] + clusters[merge_indices[1]]
        clusters = [clusters[i] for i in range(len(clusters)) if i not in merge_indices]
        clusters.append(merged_cluster)

    return clusters[0]

# 예시 데이터프레임 생성
data = pd.DataFrame({
    'x': [1, 1.5, 3, 5, 3.5, 4.5, 3.5],
    'y': [1, 2, 4, 7, 5, 5, 4.5]
})

# hierarchical clustering 수행
result = hierarchical_clustering(data)

# 결과 출력
print("클러스터링 결과:")
for i, cluster in enumerate(result):
    cluster_points = data.loc[cluster]
    print(f"클러스터 {i+1}:")
    print(cluster_points)
    print()
