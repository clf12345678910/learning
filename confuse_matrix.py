import numpy as np
from sklearn.metrics import confusion_matrix

# def cal_confuse_matrix(y_pred,y_true,n_classes):
#     matrix = np.zeros(shape=(n_classes,n_classes))
#     for p,t in zip(y_pred[0],y_true[0]):
#         matrix[t-1][p-1] += 1
#     return matrix
#
# if __name__ == '__main__':
#     y_pred = np.array([[1,1,1,2,2,2,3,3,3,1]])
#     y_true = np.array([[1,2,3,1,2,3,1,2,3,2]])
#     matrix = cal_confuse_matrix(y_pred,y_true,n_classes=3)
#     print(matrix)


y_pred = np.random.randint(0, 2, size=100)
y_true = np.random.randint(0, 2, size=100)
# 混淆矩阵
confuse_mtrics = confusion_matrix(y_true, y_pred)

if __name__ == '__main__':
    tp = confuse_mtrics[1, 1]
    fn = confuse_mtrics[1, 0]
    tn = confuse_mtrics[0, 0]
    fp = confuse_mtrics[0, 1]
    # 准确率
    accuracy = (tp + fp) / (tp + fp + tn + fn)
    # 召回率
    summon = tp / (tp + fn)
    # 精确率
    precision = tp / (tp + fp)
    # f1_score通过结合召回率与精确率来对模型分类结果进行评价，后两者越大前者越大，评价越高
    # f1_score = 2 / ((1 / summon) + (1 / precision))
    f1_score = 2 * summon * precision / (summon + precision)
    print(accuracy, summon, precision, f1_score)
