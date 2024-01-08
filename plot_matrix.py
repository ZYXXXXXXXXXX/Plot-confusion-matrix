import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np


def plot_matrix(y_true, y_pred, labels_name, title=None,  axis_labels=None):
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例

    # 图像标题
    if title is not None:
        plt.title(title)
    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=20)  # 将标签印在x轴坐标
    plt.yticks(num_local, axis_labels)
    plt.ylabel('Real Direction')  # ontdict={'family': 'Times New Roman'}
    plt.xlabel('Perceived Direction')

    # calculate the threshold decide black or white color
    thresh = int(np.max(cm)/2)

    # 大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if cm[i][j] == cm[i][j]:
                # 打印每格的字符
                plt.text(j, i, format(int(cm[i][j]), 'd'),
                         ha="center", va="center",
                         color="white" if cm[i][j] > thresh else "black")

    plt.savefig("res_matrix.png",dpi=600)

    plt.show()



if __name__ == '__main__':
    # 上 0 下 1
    # 左 2 右 3
    y_true = [2, 2, 1, 0, 1, 1, 3, 3, 1, 3, 0, 3, 0, 0, 3, 3, 2, 0, 0, 2, 1, 0, 1, 0, 1, 1, 1, 2, 0, 1, 0, 3, 2, 3, 1,
              2, 2, 3, 0, 0]
    y_predict = [2, 3, 1, 0, 1, 1, 3, 3, 2, 3, 0, 3, 0, 0, 3, 3, 2, 0, 0, 2, 3, 0, 1, 0, 1, 1, 1, 0, 3, 1, 0, 3, 1, 3,
                 1, 3, 2, 3, 0, 0]
    plot_matrix(y_true, y_predict, [0, 1, 2, 3], title='Confusion Matrix of Directions',
                axis_labels=['Up', 'Down', 'Left', 'Right'])
