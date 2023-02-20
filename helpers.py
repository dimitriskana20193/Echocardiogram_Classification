import numpy as np
def RMSE(x_pred,x_true):
   return np.sqrt(np.mean((x_true - x_pred)**2))
def confusion_matrix(y_pred, y_true):
    classes = np.unique(y_true)
    n_classes = len(classes)
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(len(y_true)):
        true_class = y_true[i]
        pred_class = y_pred[i]
        conf_matrix[true_class][pred_class] += 1
        
    return conf_matrix
def accuracy(y_pred, y_true):
    return np.mean(y_true == y_pred)