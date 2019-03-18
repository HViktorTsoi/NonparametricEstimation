import numpy as np
import queue


class ParzenWindowEstimator:
    """
    Parzen窗核密度估计方法
    """
    # 窗类型
    KERNEL_TYPE_GAU = 0  # 高斯窗
    KERNEL_TYPE_RECT = 1  # 矩形窗
    KERNEL_TYPE_BALL = 2  # 超球窗

    def __init__(self):
        # 样本和标签
        self.X_train = None
        self.y_train = None
        # 协方差矩阵
        self.Q = None
        self.dim_features = -1
        self.num_samples = - 1
        self.window_size = -1
        self.kernel = None

    def fit_data(self, X_train, y_train=None, window_size=-1, kernel_type=None):
        """
        初始化估计器
        :param X_train: 训练数据
        :param y_train: 标签
        :return: self
        """
        assert len(X_train.shape) == 2, '样本必须为列向量构成的矩阵！！！'
        self.X_train = X_train
        self.y_train = y_train
        # 特征维度和样本数
        self.dim_features, self.num_samples = X_train.shape
        # 计算协方差和其逆
        self.Q = np.cov(X_train).reshape([self.dim_features, self.dim_features])
        try:
            self.Q_inv = np.linalg.inv(self.Q)
        except Exception as e:
            # 如果协方差矩阵为奇异矩阵 则使用矩阵的伪逆代替
            self.Q_inv = np.linalg.pinv(self.Q)
        # 计算窗宽
        self.window_size = window_size
        # 设置kernel
        if kernel_type == self.KERNEL_TYPE_GAU:
            self.kernel = self.gaussian_kernel
        elif kernel_type == self.KERNEL_TYPE_BALL:
            self.kernel = self.ball_kernel
        elif kernel_type == self.KERNEL_TYPE_RECT:
            self.kernel = self.rect_kernel
        # 返回本身
        return self

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
                (X - Xi).T,
                self.Q_inv,
                (X - Xi)
            ]) / Hn ** 2
        )
        # 保证返回一个浮点数
        return np.squeeze(K)

    def p(self, X):
        """
        计算样本X的类条件概率
        :param X:样本X
        :return:x的概率密度
        """
        # 窗宽
        Hn = self.window_size / np.sqrt(self.num_samples)
        # 转换为列向量 求核函数累积
        Px = 1 / self.num_samples * sum(
            self.kernel(X.reshape(-1, 1), self.X_train[:, idx].reshape(-1, 1), Hn)
            # 1 / Vn * kernel(X.reshape(-1, 1), self.X_train[:, idx].reshape(-1, 1), Hn)
            for idx in range(self.num_samples)
        )
        print('X:', X, '概率密度:', Px)
        return Px


class KNNEstimator:
    """
    KNN 概率密度估计
    """

    def __init__(self):
        # 样本和标签
        self.X_train = None
        self.y_train = None
        self.dim_features = -1
        self.num_samples = - 1
        self.k = -1

    def fit_data(self, X_train, y_train=None, Kn=-1):
        """
        初始化估计器
        :param X_train: 训练数据
        :param y_train: 标签
        :param k: K值
        :return: self
        """
        assert len(X_train.shape) == 2, '样本必须为列向量构成的矩阵！！！'
        assert Kn >= 2, 'Kn不能设置的过小 至少大于等于2'
        self.X_train = X_train
        self.y_train = y_train
        # 特征维度和样本数
        self.dim_features, self.num_samples = X_train.shape
        # 设置k值
        self.k = Kn
        return self

    def p(self, X):
        # 转换为列向量
        X = X.reshape(-1, 1)
        # 以下基本都为numpy的广播运算
        # 求样本X到全部训练集合的欧式距离
        all_distance = np.linalg.norm(X - self.X_train, axis=0)
        # 递增排序之后对应的下标
        sorted_distance_indices = np.argsort(all_distance)
        # 获取前k个下标及其对应的样本
        nearest_k_indices = sorted_distance_indices[:self.k]
        nearest_k_samples = self.X_train[:, nearest_k_indices]
        # 找到这k个样本d维特征的最大最小值 相减就得到了超立方体的d个边长
        max_x = np.max(nearest_k_samples, axis=1)
        min_x = np.min(nearest_k_samples, axis=1)
        # 求超棱长
        edge_lengths = max_x - min_x
        # 求超体积
        V = np.prod(edge_lengths)
        # 防止除0 若出现0则用整个数据集的平均体积代替
        if V == 0:
            V = np.prod(
                np.max(self.X_train, axis=1) - np.min(self.X_train, axis=1)
            ) / self.num_samples
        # 求Px
        Px = (self.k / self.num_samples) / V
        return Px
