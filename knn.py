"""
KNN算法的步骤如下：

输入：训练集和待预测样本。
计算待预测样本与训练集中每个样本的距离。常用的距离度量方法有欧氏距离、曼哈顿距离等。
根据距离排序，找到与待预测样本最近的K个邻居。
对于分类任务，根据K个邻居的标签进行投票，选择出现次数最多的标签作为预测结果。
对于回归任务，根据K个邻居的标签进行平均或加权平均，作为预测结果。
返回预测结果。

"""


import numpy as np

def euclidean_distance(x1, x2):
    # 计算欧氏距离
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(X_train, y_train, X_test, k):
    y_pred = []

    for test_sample in X_test:
        distances = [euclidean_distance(test_sample, train_sample) for train_sample in X_train]
        # 计算待预测样本与训练集中每个样本的距离
        # np.argsort() ---> 将数组中的元素按升序进行排序并返回对应的索引值
        nearest_indices = np.argsort(distances)[:k]
        print(nearest_indices)
        # 根据距离排序，找到与待预测样本最近的K个邻居
        nearest_labels = y_train[nearest_indices]
        print(nearest_labels)
        # np.unique() ---> 去重，并按照升序返回数组中的元素，返回唯一值的索引
        unique_labels, label_counts = np.unique(nearest_labels, return_counts=True)
        # 对K个邻居的标签进行统计
        predicted_label = unique_labels[np.argmax(label_counts)]
        # 选择出现次数最多的标签作为预测结果
        y_pred.append(predicted_label)

    return np.array(y_pred)


if __name__ == '__main__':
    x1 = np.array([3,3,3,3])
    x2 = np.array([5,5,5,5])
    x3 = np.array([6,6,6,6])
    print(knn_predict(x1, x2, x3, 3))


