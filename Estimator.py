import numpy as np
import queue


class ParzenWindowEstimator:
    """
    Parzen窗核密度估计方法
    """

    def __init__(self):
        # 样本和标签
        self.X_train = None
        self.y_train = None
        # 协方差矩阵
        self.Q = None
        self.dim_features = -1
        self.num_samples = - 1
        self.window_size = -1

    def fit_data(self, X_train, y_train=None):
        """
        初始化估计器
        :param X_train: 训练数据
        :param y_train: 标签
        :return: None
        """
        assert len(X_train.shape) == 2, '样本必须为列向量构成的矩阵！！！'
        self.X_train = X_train
        self.y_train = y_train
        # 特征维度和样本数
        self.dim_features, self.num_samples = X_train.shape
        # 计算协方差
        self.Q = np.cov(X_train).reshape([self.dim_features, self.dim_features])

    def ball_kernel(self, X, Xi, Hn):
        """
        超球体窗
        :param X: 新样本
        :param Xi: 训练集样本
        :param Hn: 窗宽
        :return: 核函数值
        """
        if np.linalg.norm(X - Xi) <= Hn:
            # 如果X在以Xi为中心的超球体内
            return (Hn ** self.dim_features) ** -1
        else:
            return 0

    def rect_kernel(self, X, Xi, Hn):
        """
        矩形窗
        :param X: 新样本
        :param Xi: 训练集样本
        :param Hn: 窗宽
        :return: 核函数值
        """
        if (np.abs(X - Xi) < Hn / 2).all():
            # 所有的特征距中心点都小于一半的棱长
            return 1 / np.power(Hn, self.dim_features)
        else:
            return 0

    def gaussian_kernel(self, X, Xi, Hn):
        """
        高斯窗
        :param X: 新样本
        :param Xi: 训练集样本
        :param Hn: 窗宽
        :return: 核函数值
        """
        # 计算核函数值
        K = 1 / np.sqrt(
            (2 * np.pi) ** self.dim_features
            * Hn ** (2 * self.dim_features)
            * np.linalg.norm(self.Q)
        ) * np.exp(
            -0.5 * np.linalg.multi_dot([
                (X - Xi).T, np.linalg.inv(self.Q), (X - Xi)
            ]) / Hn ** 2
        )
        # 保证返回一个浮点数
        return np.squeeze(K)

    def p(self, X, kernel):
        """
        计算样本X的类条件概率
        :param X:样本X
        :return:x的概率密度
        """
        # 窗宽
        Hn = self.window_size / np.sqrt(self.num_samples)
        # 超立方体体积
        Vn = np.power(Hn, self.dim_features)
        # 转换为列向量 求核函数累积
        Px = 1 / self.num_samples * sum(
            kernel(X.reshape(-1, 1), self.X_train[:, idx].reshape(-1, 1), Hn)
            # 1 / Vn * kernel(X.reshape(-1, 1), self.X_train[:, idx].reshape(-1, 1), Hn)
            for idx in range(self.num_samples)
        )
        return Px


class KNNEstimator:
    """
    KNN 概率密度估计
    """

    class CompareSample():
        def __init__(self, distance):
            pass

    def __init__(self):
        # 样本和标签
        self.X_train = None
        self.y_train = None
        self.dim_features = -1
        self.num_samples = - 1
        self.k = -1

    def fit_data(self, X_train, y_train=None):
        """
        初始化估计器
        :param X_train: 训练数据
        :param y_train: 标签
        :return: None
        """
        assert len(X_train.shape) == 2, '样本必须为列向量构成的矩阵！！！'
        self.X_train = X_train
        self.y_train = y_train
        # 特征维度和样本数
        self.dim_features, self.num_samples = X_train.shape

    def p(self, X):
        q = queue.PriorityQueue()
        # for
        pass
