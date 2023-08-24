import numpy as np

def calc_entropy(y):
    # 计算数据集的熵
    unique_labels, label_counts = np.unique(y, return_counts=True)
    label_probs = label_counts / len(y)
    entropy = -np.sum(label_probs * np.log2(label_probs))
    return entropy

def calc_info_gain(X, y, feature_idx):
    # 计算特征(feature_idx)的信息增益
    total_entropy = calc_entropy(y)
    feature_values = np.unique(X[:, feature_idx])
    weighted_entropy = 0
    for value in feature_values:
        subset_indices = np.where(X[:, feature_idx] == value)
        subset_y = y[subset_indices]
        subset_entropy = calc_entropy(subset_y)
        weighted_entropy += len(subset_y) / len(y) * subset_entropy
    info_gain = total_entropy - weighted_entropy
    return info_gain

def find_best_feature(X, y):
    # 找到最佳划分特征
    num_features = X.shape[1]
    best_feature_idx = -1
    best_info_gain = -np.inf
    for feature_idx in range(num_features):
        info_gain = calc_info_gain(X, y, feature_idx)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature_idx = feature_idx
    return best_feature_idx

def create_tree(X, y, features):
    # 创建决策树
    unique_labels, label_counts = np.unique(y, return_counts=True)
    if len(unique_labels) == 1:
        # 如果所有样本都属于同一类别，则直接返回该类别
        return unique_labels[0]
    if len(features) == 0:
        # 如果没有可用特征，则返回样本数最多的类别
        return unique_labels[np.argmax(label_counts)]
    best_feature_idx = find_best_feature(X, y)
    best_feature = features[best_feature_idx]
    tree = {best_feature: {}}
    feature_values = np.unique(X[:, best_feature_idx])
    for value in feature_values:
        value_indices = np.where(X[:, best_feature_idx] == value)
        subset_X = X[value_indices]
        subset_y = y[value_indices]
        subset_features = np.delete(features, best_feature_idx)
        tree[best_feature][value] = create_tree(subset_X, subset_y, subset_features)
    return tree

def predict(tree, x):
    # 使用决策树进行预测
    for feature, subtree in tree.items():
        value = x[feature]
        if value in subtree:
            if isinstance(subtree[value], dict):
                return predict(subtree[value], x)
            else:
                return subtree[value]

# 示例数据
X = np.array([[1, 1, 0],
              [1, 1, 1],
              [0, 1, 0],
              [0, 0, 0],
              [1, 0, 0]])
y = np.array([0, 1, 1, 0, 0])
features = np.array([0, 1, 2])  # 特征索引

# 创建决策树
tree = create_tree(X, y, features)

# 进行预测
x_test = np.array([1, 0, 1])
prediction = predict(tree, x_test)
print('预测结果:', prediction)
