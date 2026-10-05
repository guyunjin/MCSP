import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE














import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os










































































































































































































import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os



DEEP_COLORS = [

    (0.1, 0.3, 0.8), (0.9, 0.1, 0.1), (0.1, 0.7, 0.3), (0.8, 0.5, 0.0),
    (0.6, 0.0, 0.8), (0.0, 0.6, 0.6), (0.8, 0.1, 0.6), (0.5, 0.4, 0.0),
    (0.0, 0.3, 0.8), (0.8, 0.0, 0.0), (0.0, 0.5, 0.0), (0.7, 0.3, 0.0),
    (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.7, 0.0, 0.5), (0.4, 0.3, 0.0),
    (0.0, 0.2, 0.7), (0.7, 0.0, 0.0), (0.0, 0.4, 0.0), (0.6, 0.2, 0.0),

    (0.1, 0.1, 0.1),
    (0.5, 0.5, 0.9),
    (1.0, 0.4, 0.4),
    (0.4, 0.9, 0.4),
    (0.9, 0.8, 0.1),
    (0.1, 0.9, 0.9),
    (0.5, 0.2, 0.2),
    (0.2, 0.5, 0.2),
    (0.2, 0.2, 0.5),
    (0.9, 0.6, 0.9)
]























def visualize_epoch_clustering_withid(args, features, true_labels, pred_labels, epoch=None, output_dir=".",
                               metrics=None, cmap_true=None, cmap_pred=None, id2label=None):
    """
    可视化聚类结果，使用 t-SNE 降维到二维空间，并用不同颜色和边框表示真实标签与预测标签。

    参数：
        args: 包含方法名、数据集名、随机种子等信息的对象
        features: 特征向量（可以是 torch.Tensor 或 numpy.ndarray）
        true_labels: 真实标签列表
        pred_labels: 预测标签列表
        epoch: 当前训练轮次（用于标题显示）
        output_dir: 图像保存路径
        metrics: 可选的评估指标字典，如 {'ACC': 0.95, 'NMI': 0.88}
        cmap_true: 真实标签的颜色映射
        cmap_pred: 预测标签的颜色映射
        id2label: 类别 ID 到标签名称的映射字典（可选）



    """










    if 'torch' in str(type(features)):
        features = features.detach().cpu().numpy()
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)


    X = features
    if X.shape[1] > 50:
        pca = PCA(n_components=50)
        X = pca.fit_transform(X)
    tsne = TSNE(n_components=2, init='pca', random_state=0,
                perplexity=30,
                early_exaggeration=25,
                learning_rate=200,
                n_iter=1000)
    X_2d = tsne.fit_transform(X)


    unique_true = sorted(np.unique(true_labels))
    unique_pred = sorted(np.unique(pred_labels))

    num_true_classes = len(unique_true)
    num_pred_classes = len(unique_pred)


    if num_true_classes > len(DEEP_COLORS) or num_pred_classes > len(DEEP_COLORS):
        raise ValueError(f"最多支持 {len(DEEP_COLORS)} 类，当前类别数为：{max(num_true_classes, num_pred_classes)}")


    cmap_true = plt.cm.colors.ListedColormap([DEEP_COLORS[i] for i in range(num_true_classes)])
    cmap_pred = plt.cm.colors.ListedColormap([DEEP_COLORS[i] for i in range(num_pred_classes)])


    true_color_map = {label: cmap_true(i) for i, label in enumerate(unique_true)}
    pred_color_map = {label: cmap_pred(i) for i, label in enumerate(unique_pred)}

    facecolors = [true_color_map[label] for label in true_labels]
    edgecolors = [pred_color_map[label] for label in pred_labels]


    plt.figure(figsize=(14, 11), dpi=800)


    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=facecolors, edgecolors=edgecolors,
                marker='o', s=20, linewidths=1, alpha=1.0)


    mask = (true_labels != pred_labels)
    if np.any(mask):
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], marker='x', c='k', s=20, linewidths=0.2, alpha=1.0)


    title = f"Epoch {epoch}" if epoch is not None else "Clustering Visualization"
    if metrics:
        met_str = ", ".join(f"{k}={v:.4f}" for k,v in metrics.items())
        title += " - " + met_str
    plt.title(title)


    plt.xticks([]); plt.yticks([])


    if id2label is not None:
        legend_elements = []
        for lab in unique_true:
            rgba = (*true_color_map[lab][:3], 1.0)
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                              label=id2label.get(lab, str(lab)),
                                              markerfacecolor=rgba, markersize=10))

        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=len(unique_true), frameon=False)

        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True,  ncol=(len(unique_true) + 1) // 2, frameon=False,prop={'size': 14})

    plt.tight_layout()


    os.makedirs(output_dir, exist_ok=True)


    fname = f"{args.method}_{args.dataset}_{args.seed}.png"
    save_path = os.path.join(output_dir, fname)


    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


    return save_path
