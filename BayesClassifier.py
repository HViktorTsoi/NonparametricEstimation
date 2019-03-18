import sklearn.datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Estimator import KNNEstimator, ParzenWindowEstimator
import numpy as np

# 样本数据集
# dataset = sklearn.datasets.load_iris()
# dataset = sklearn.datasets.load_breast_cancer()
dataset = sklearn.datasets.load_digits()

X = dataset['data']
y = dataset['target']
# 总类别数
num_classes = len(set(dataset['target']))
# 分割训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练样本转换为列向量
assert len(X_train.shape) > 1 and len(X_test.shape) > 1, '训练样本不是列向量！'
X_train = X_train.T
X_test = X_test.T

# 每个类别的先验概率 设置为一样大
prior_prob = [1 / num_classes for _ in range(num_classes)]

# 初始化每个类别对应的类条件概率的非参数估计器
class_prob = [
    ParzenWindowEstimator()
        .fit_data(
        np.squeeze(X_train[:, np.where(y_train == c)]),  # 对应类别的训练样本 删除多余维度
        window_size=150  # 窗体长度
    )
    for c in range(num_classes)
]

pred = []
# 使用最小风险贝叶斯决策来对测试集进行分类
for idx, X in enumerate(X_test.T):
    # 当前样本在各个类别的后验概率
    posterior_prob_list = []
    for c in range(num_classes):
        # 计算后验概率=估计的类概率密度*先验概率 忽略归一化的分母(因为只需要比较大小)
        posterior_prob = class_prob[c].p(
            X.reshape(-1, 1),  # 样本转列向量
            kernel=class_prob[c].rect_kernel
        ) * prior_prob[c]  # 先验概率

        posterior_prob_list.append(posterior_prob)
    print('=' * 30)
    print('样本：', X, '各类别概率：', posterior_prob_list)
    # 找后验概率最大的类别 即最小错误率类别
    predict = int(np.argmax(posterior_prob_list))
    print('Prob: ', posterior_prob_list[predict], '预测：', predict, 'GroundTruth：', y_test[idx])
    pred.append([predict, posterior_prob_list[predict]])
# 计算分类器性能
pred = np.array(pred)
print('准确率：', np.sum(pred[:, 0] == y_test) / len(y_test))
