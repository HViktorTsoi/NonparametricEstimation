import numpy as np
import matplotlib.pylab as plt
import sklearn.datasets
from mpl_toolkits.mplot3d import Axes3D
from Estimator import ParzenWindowEstimator, KNNEstimator
import matplotlib

matplotlib.rcParams['font.size'] = 14
plt.rcParams['figure.dpi'] = 300


def make_dataset(dim, size=None):
    """
    生成数据集
    :dim: 特征维度
    :return: X Y
    """
    if dim == 1:
        SIZE = size
        X = np.hstack([
            np.random.normal(loc=2, scale=0.6, size=SIZE),
            np.random.normal(loc=8, scale=1.5, size=SIZE),
            np.random.normal(loc=12, scale=1.1, size=SIZE),
        ])
        # plt.gcf().set_size_inches(7, 4)
        # plt.show()
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


def experiment_1d_parzen(window_size, kernel_type, N):
    # 随机抽取
    indices = np.random.choice(X_train_origin.shape[1], N, replace=False)
    X_train = X_train_origin[:, indices]
    # 初始化估计器
    pw_estimator = ParzenWindowEstimator() \
        .fit_data(X_train, window_size=window_size, kernel_type=kernel_type)

    # 生成均匀数据空间
    X = np.arange(np.min(X_train), np.max(X_train), SAMPLE_SCALE)
    # 计算概率密度
    Y = [pw_estimator.p(np.array(x)) for x in X]
    # 可视化
    plt.yticks([])
    plt.plot(X, Y, 'r', linewidth=2, label='Esti')

    # 绘制GrounTruth分布
    dist, edges = np.histogram(X_train_origin, bins=60)
    plt.plot(edges[:-1], dist * (max(Y) / max(dist)), 'g--', label='GT')
    plt.legend(loc='upper right', prop={'size': 10})
    plt.ylabel('Density')
    plt.xlabel('X({})'.format(kernel_type))
    plt.title('N={}  Hn={}'.format(N if N < 1e3 else '∞', window_size))


def experiment_2d_parzen(ax, window_size):
    # 生成数据集
    X_train, y_train = make_dataset(dim=2)

    # 初始化估计器
    pw_estimator = ParzenWindowEstimator() \
        .fit_data(X_train, window_size=window_size, kernel_type=ParzenWindowEstimator.KERNEL_TYPE_GAU)

    # 求概率密度分布
    X, Y = np.meshgrid(np.arange(0, 9, SAMPLE_SCALE), np.arange(0, 5, SAMPLE_SCALE))
    Z = [pw_estimator.p(np.array(x)) for x in zip(X.flat, Y.flat)]

    ax.plot_surface(X, Y, np.array(Z).reshape(X.shape), rstride=1, cstride=1, cmap='hot_r')
    ax.scatter(X_train[0, :], X_train[1, :], 1.1 * max(Z) * np.ones([X_train.shape[1]]), label='GT')
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    ax.set_zlabel('Density')
    ax.set_zticks([])
    ax.set_title('Parzen, Hn={}'.format(window_size))
    ax.legend()


def experiment_1d_knn(K, kernel_type, N):
    # 随机抽取
    indices = np.random.choice(X_train_origin.shape[1], N, replace=False)
    X_train = X_train_origin[:, indices]
    # 初始化估计器
    # 初始化估计器
    knn_estimator = KNNEstimator().fit_data(X_train, Kn=K)

    # 生成均匀数据空间
    X = np.arange(np.min(X_train), np.max(X_train), SAMPLE_SCALE)
    # 计算概率密度
    Y = [knn_estimator.p(np.array(x)) for x in X]
    # 可视化
    plt.yticks([])
    plt.plot(X, Y, 'r', linewidth=2, label='Esti')

    # 绘制GrounTruth分布
    dist, edges = np.histogram(X_train_origin, bins=60)
    plt.plot(edges[:-1], dist * (max(Y) / max(dist)), 'g--', label='GT')
    plt.legend(loc='upper right', prop={'size': 10})
    plt.ylabel('Density')
    plt.xlabel('X({})'.format(kernel_type))
    plt.title('N={}  K={}'.format(N if N < 1e4 else '∞', K))


def experiment_2d_knn(ax, K):
    # 生成数据集
    X_train, y_train = make_dataset(dim=2)

    # 初始化估计器
    knn_estimator = KNNEstimator() \
        .fit_data(X_train, Kn=K)

    # 求概率密度分布
    X, Y = np.meshgrid(np.arange(0, 9, SAMPLE_SCALE), np.arange(0, 5, SAMPLE_SCALE))
    Z = [knn_estimator.p(np.array(x)) for x in zip(X.flat, Y.flat)]

    ax.plot_surface(X, Y, np.array(Z).reshape(X.shape), rstride=1, cstride=1, cmap='hot_r')
    ax.scatter(X_train[0, :], X_train[1, :], 1.1 * max(Z) * np.ones([X_train.shape[1]]), label='GT')
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    ax.set_zlabel('Density')
    ax.set_zticks([])
    ax.set_title('KNN, K={}'.format(K))
    ax.legend()


if __name__ == '__main__':
    SAMPLE_SCALE = 0.06
    # parzen窗测试
    # 生成数据集
    X_train_origin, y_train = make_dataset(dim=1, size=5000)

    # 1d Parzen窗
    # 每一类窗宽参数
    parameters = [
        [1, 10, 50],
        [1, 10, 50],
        [0.1, 2, 10],
    ]
    for k_id, kernel_type in enumerate([
        ParzenWindowEstimator.KERNEL_TYPE_RECT,
        ParzenWindowEstimator.KERNEL_TYPE_BALL,
        ParzenWindowEstimator.KERNEL_TYPE_GAU,
    ]):
        for N in [5, 50, 500, 3000]:
            plt.gcf().set_size_inches(13, 3.4)
            for idx, window_size in enumerate(parameters[k_id]):
                plt.subplot(1, 3, idx + 1)
                experiment_1d_parzen(window_size=window_size, kernel_type=kernel_type, N=N)
            plt.tight_layout()
            plt.savefig('./output/{}_{}.png'.format(kernel_type, N))
            plt.show()

    # 1d KNN
    for N in [100, 1000, 5000, 10000]:
        plt.gcf().set_size_inches(13, 3.4)
        for idx, K in enumerate([10, 200, 900]):
            plt.subplot(1, 3, idx + 1)
            experiment_1d_knn(K=K, kernel_type='KNN', N=N)
        plt.tight_layout()
        plt.savefig('./output/{}_{}.png'.format('KNN', N))
        plt.show()

    SAMPLE_SCALE = 0.1

    # 2d PARZEN
    fig = plt.figure()
    fig.set_size_inches(13, 3.4)
    for idx, window_size in enumerate([0.1, 2, 10]):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        experiment_2d_parzen(ax, window_size)
    plt.savefig('./output/{}_{}.png'.format('PZ', '3d'))
    plt.tight_layout()
    plt.show()

    # 2d KNN
    fig = plt.figure()
    fig.set_size_inches(13, 3.4)
    for idx, K in enumerate([5, 10, 50]):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        experiment_2d_knn(ax, K)
    plt.savefig('./output/{}_{}.png'.format('KNN', '3d'))
    plt.tight_layout()
    plt.show()
