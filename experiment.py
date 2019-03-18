import numpy as np
import matplotlib.pylab as plt
import sklearn.datasets
from mpl_toolkits.mplot3d import Axes3D
from Estimator import ParzenWindowEstimator, KNNEstimator


def make_dataset(dim):
    """
    生成数据集
    :dim: 特征维度
    :return: X Y
    """
    if dim == 1:
        SIZE = 40000
        X = np.hstack([
            np.random.normal(loc=2, scale=0.6, size=SIZE),
            np.random.normal(loc=8, scale=1.5, size=SIZE),
            np.random.normal(loc=12, scale=1.1, size=SIZE),
        ])
        # 绘制GrounTruth分布
        dist, edges = np.histogram(X, bins=60)
        plt.plot(edges[:-1], dist / len(X))
        plt.gcf().set_size_inches(7, 4)
        plt.show()
        # 返回X
        return X.reshape(1, -1), None
    elif dim == 2:
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


def experiment_1d_parzen():
    # 生成数据集
    X_train, y_train = make_dataset(dim=1)
    # 随机抽取
    indices = np.random.choice(X_train.shape[1], 500, replace=False)
    X_train = X_train[:, indices]
    # 初始化估计器
    pw_estimator = ParzenWindowEstimator() \
        .fit_data(X_train, window_size=5, kernel_type=ParzenWindowEstimator.KERNEL_TYPE_GAU)

    # 生成均匀数据空间
    X = np.arange(np.min(X_train), np.max(X_train), 0.05)
    # 计算概率密度
    Y = [pw_estimator.p(np.array(x)) for x in X]
    # 可视化
    plt.gcf().set_size_inches(7, 4)
    plt.yticks([])
    plt.plot(X, Y, label='Density')

    # 绘制原始数据点
    plt.scatter(X_train, np.zeros(X_train.shape), c='r', label='Train Samples')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def experiment_2d_parzen():
    # 生成数据集
    X_train, y_train = make_dataset(dim=2)

    # 初始化估计器
    pw_estimator = ParzenWindowEstimator() \
        .fit_data(X_train, window_size=1, kernel_type=ParzenWindowEstimator.KERNEL_TYPE_GAU)

    # 求概率密度分布
    X, Y = np.meshgrid(np.arange(0, 9, 0.1), np.arange(0, 5, 0.1))
    Z = [pw_estimator.p(np.array(x)) for x in zip(X.flat, Y.flat)]

    ax = Axes3D(plt.figure())
    ax.plot_surface(X, Y, np.array(Z).reshape(X.shape), rstride=1, cstride=1, cmap='hot_r')
    ax.scatter(X_train[0, :], X_train[1, :], 1.1 * max(Z) * np.ones([X_train.shape[1]]), label='Train Samples')
    ax.set_zlabel('Density')
    ax.set_zticks([])
    ax.legend()
    plt.show()


def experiment_1d_knn():
    # 生成数据集
    X_train, y_train = make_dataset(dim=1)
    # 随机抽取
    indices = np.random.choice(X_train.shape[1], 400, replace=False)
    X_train = X_train[:, indices]

    # 初始化估计器
    knn_estimator = KNNEstimator()
    # 超参数K值
    knn_estimator.fit_data(X_train, Kn=50)

    # 生成均匀数据空间
    X = np.arange(np.min(X_train), np.max(X_train), 0.01)
    # 计算概率密度
    Y = [knn_estimator.p(np.array(x)) for x in X]
    # 可视化
    plt.gcf().set_size_inches(7, 4)
    plt.yticks([])
    plt.plot(X, Y, label='Density')

    # 绘制原始数据点
    plt.scatter(X_train, np.zeros(X_train.shape), c='r', label='Train Samples')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def experiment_2d_knn():
    # 生成数据集
    X_train, y_train = make_dataset(dim=2)
    # 初始化估计器
    knn_estimator = KNNEstimator()
    knn_estimator.fit_data(X_train, Kn=15)

    # 求概率密度分布
    X, Y = np.meshgrid(np.arange(0, 9, 0.1), np.arange(0, 5, 0.1))
    Z = [knn_estimator.p(np.array(x)) for x in zip(X.flat, Y.flat)]

    ax = Axes3D(plt.figure())
    ax.plot_surface(X, Y, np.array(Z).reshape(X.shape), rstride=1, cstride=1, cmap='hot_r')
    ax.scatter(X_train[0, :], X_train[1, :], 1.1 * max(Z) * np.ones([X_train.shape[1]]), label='Train Samples')
    ax.set_zlabel('Density')
    ax.set_zticks([])
    ax.legend()
    plt.show()


if __name__ == '__main__':
    # 二维数据集测试
    experiment_1d_parzen()
    # experiment_2d_parzen()
    # experiment_1d_knn()
    # experiment_2d_knn()
