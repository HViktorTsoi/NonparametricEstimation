import numpy as np
import matplotlib.pylab as plt
import sklearn.datasets
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


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
            1 / Vn * kernel(X.reshape(-1, 1), self.X_train[:, idx].reshape(-1, 1), Hn)
            for idx in range(self.num_samples)
        )
        return Px


def make_dataset(dim):
    """
    生成数据集
    :dim: 特征维度
    :return: X Y
    """
    # 载入数据集
    dataset = sklearn.datasets.load_iris()
    # 降到需要的维度
    # dataset['data'] = PCA(n_components=dim).fit_transform(dataset['data'])
    X_train = dataset['data'].T[2:2 + dim, :]
    # 如果只有1维 强制转换为列向量 使计算过程统一
    if len(X_train.shape) == 1:
        X_train.reshape(1, X_train.shape[0])
    y_train = dataset['target']
    return X_train, y_train


def experiment_1d_data():
    # 生成数据集
    X_train, y_train = make_dataset(dim=1)
    # 初始化估计器
    pw_estimator = ParzenWindowEstimator()
    # 超参数窗体宽度
    pw_estimator.window_size = 1

    pw_estimator.fit_data(X_train)

    # 生成均匀数据空间
    X = np.arange(0, 8, 0.01)
    # 计算概率密度
    Y = [pw_estimator.p(np.array(x), kernel=pw_estimator.gaussian_kernel) for x in X]
    # 可视化
    plt.gcf().set_size_inches(8, 4)
    plt.yticks([])
    plt.plot(X, Y, label='Density')

    # 绘制原始数据点
    plt.scatter(X_train, np.zeros(X_train.shape), c='r', label='Train Samples')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def experiment_2d_data():
    # 生成数据集
    X_train, y_train = make_dataset(dim=2)
    # 初始化估计器
    pw_estimator = ParzenWindowEstimator()
    # 超参数窗体宽度
    pw_estimator.window_size = 5

    pw_estimator.fit_data(X_train)

    # 求概率密度分布
    X, Y = np.meshgrid(np.arange(0, 9, 0.1), np.arange(0, 5, 0.1))
    Z = [pw_estimator.p(np.array(x), kernel=pw_estimator.ball_kernel) for x in zip(X.flat, Y.flat)]

    ax = Axes3D(plt.figure())
    ax.plot_surface(X, Y, np.array(Z).reshape(X.shape), rstride=1, cstride=1, cmap='hot_r')
    ax.scatter(X_train[0, :], X_train[1, :], 1.1 * max(Z) * np.ones([X_train.shape[1]]), label='Train Samples')
    ax.set_zlabel('Density')
    ax.set_zticks([])
    ax.legend()
    plt.show()


if __name__ == '__main__':
    # 二维数据集测试
    experiment_2d_data()
    # experiment_1d_data()
