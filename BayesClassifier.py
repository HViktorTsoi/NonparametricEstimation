import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from Estimator import KNNEstimator, ParzenWindowEstimator
import numpy as np
import matplotlib

matplotlib.rcParams['font.size'] = 14
plt.rcParams['figure.dpi'] = 300

# 估计器类型
ESTIMATE_TYEP = 'PARZEN'
# 样本数据集
# dataset = sklearn.datasets.load_iris()
dataset = sklearn.datasets.load_breast_cancer()
# dataset = sklearn.datasets.load_digits()

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
# Parzen窗估计
if ESTIMATE_TYEP == 'PARZEN':
    class_prob = [
        ParzenWindowEstimator()
            .fit_data(
            np.squeeze(X_train[:, np.where(y_train == c)]),  # 对应类别的训练样本 删除多余维度
            kernel_type=ParzenWindowEstimator.KERNEL_TYPE_GAU,  # 窗体类型
            window_size=40  # 窗体长度
        )
        for c in range(num_classes)
    ]
else:
    # # KNN估计
    class_prob = [
        KNNEstimator()
            .fit_data(
            np.squeeze(X_train[:, np.where(y_train == c)]),  # 对应类别的训练样本 删除多余维度
            Kn=8
        )
        for c in range(num_classes)
    ]

pred = []
# 使用最小错误率贝叶斯决策来对测试集进行分类
for idx, X in enumerate(X_test.T):
    # 当前样本在各个类别的后验概率
    posterior_prob_list = []
    for c in range(num_classes):
        # 计算后验概率=估计的类概率密度*先验概率 忽略归一化的分母(因为只需要比较大小)
        posterior_prob = class_prob[c].p(
            X.reshape(-1, 1),  # 样本转列向量
        ) * prior_prob[c]  # 先验概率

        posterior_prob_list.append(posterior_prob)

    # 找后验概率最大的类别 即最小错误率类别
    predict = int(np.argmax(posterior_prob_list))
    pred.append([predict, posterior_prob_list[predict]])

    print(idx, '  ', '=' * 30)
    print('各类别概率：', posterior_prob_list)
    print('Prob: ', posterior_prob_list[predict], '预测：', predict, 'GroundTruth：', y_test[idx])

# 计算分类器性能
pred = np.array(pred)
# 准确率
print('准确率：', metrics.accuracy_score(y_test, pred[:, 0]))
# 召回率
print('召回率：', metrics.recall_score(y_test, pred[:, 0], average='macro'))
# PR曲线
if num_classes == 2:
    # 两类只需要画一个PR曲线
    tpr, fpr, th = metrics.precision_recall_curve(y_test, pred[:, 1], pos_label=1)
    plt.plot(tpr, fpr)
    plt.title('ROC Curve')
    plt.xlabel('TPR')
    plt.ylabel('FPR')
    plt.savefig('./output/BR_{}_PR.png'.format(ESTIMATE_TYEP))
    plt.show()
else:
    plt.gcf().set_size_inches(13, 5)
    # 多类要画多个PR曲线
    for class_id in range(num_classes):
        plt.subplot(2, 5, class_id + 1)
        # 两类只需要画一个PR曲线
        tpr, fpr, th = metrics.roc_curve(y_test, pred[:, 1], pos_label=class_id)
        plt.plot(tpr, fpr)
        plt.title('CLASS:{}'.format(class_id))
        plt.xlabel('TPR')
        plt.ylabel('FPR')
    plt.tight_layout()
    plt.savefig('./output/DG_{}_PR.png'.format(ESTIMATE_TYEP))
    plt.show()
