import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc,roc_curve, auc,accuracy_score, confusion_matrix,f1_score,precision_score, recall_score,classification_report
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot


# calucate F1-S precision accuracy recall
def calucate(y_label, fake_x_pre4):
    target_names = ['class 0', 'class 1', 'class 2', 'class 3']

    accuracy = accuracy_score(y_label, fake_x_pre4)
    pre1 = precision_score(y_label, fake_x_pre4, average='micro')
    pre2 = c(y_label, fake_x_pre4, average='macro')
    re1 = recall_score(y_label, fake_x_pre4, average='micro')
    re2 = recall_score(y_label, fake_x_pre4, average='macro')
    f1_s1 = f1_score(y_label, fake_x_pre4, average='micro')
    f1_s2 = f1_score(y_label, fake_x_pre4, average='macro')
    f1_s3 = f1_score(y_label, fake_x_pre4, average='weighted')
    f1_s4 = f1_score(y_label, fake_x_pre4, average=None)


    re1 = recall_score(y_label, fake_x_pre4, average='micro')
    re2 = recall_score(y_label, fake_x_pre4, average='macro')
    re3 = recall_score(y_label, fake_x_pre4, average='weighted')
    re4 = recall_score(y_label, fake_x_pre4, average=None)


    pre3 = precision_score(y_label, fake_x_pre4, average='micro')
    pre4 = precision_score(y_label, fake_x_pre4, average='macro')
    pre5 = precision_score(y_label, fake_x_pre4, average='weighted')
    pre6 = precision_score(y_label, fake_x_pre4, average=None)

    print('f1_sore_micro', f1_s1)
    print('recall_micro', re1)
    print('precision_micro', pre3)
    print('\n')


    print('f1_sore_weighted', f1_s3)
    print('recall_weighted', re3)
    print('precision_weighted', pre5)
    print('\n')

    print('f1_sore_None', f1_s4)
    print('recall_None_', re4)
    print('precision_None_', pre6)
    print('\n')
    print('\n')
    matrix = confusion_matrix(y_label, fake_x_pre4)
    print('\n')

    matrix = confusion_matrix(y_label, fake_x_pre4)
    print('matrix', matrix)
    print('classification', classification_report(y_label, fake_x_pre4, target_names=target_names))
    print('accuracy',accuracy)
    print('f1_sore macro ', f1_s2)
    print('recall macro ', re2)
    print('precision macro ', pre4)

def roc(sum_label,f_yroc,classes,roc_dir):
    num_classes = classes
    sum_label = np.array(sum_label)
    f_yroc = np.array(f_yroc)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    accuracy_i = dict()
    precision = dict()
    recall = dict()
    f1 = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(sum_label[:, i], f_yroc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])



    fpr["micro"], tpr["micro"], _ = roc_curve(sum_label.ravel(),f_yroc.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    lw = 2

    np.save(roc_dir + '/vgg_fpr.npy', fpr)
    np.save(roc_dir + '/vgg_tpr.npy', tpr)
    np.save(roc_dir + '/vgg_roc_auc.npy', roc_auc)
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','yellow','block'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('our_best')
    plt.legend(loc="lower right")
    plt.show()
    print('roc_auc["macro"] ',roc_auc["macro"] )


