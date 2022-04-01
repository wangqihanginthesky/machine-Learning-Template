import numpy as np
from sklearn.metrics import mean_squared_error
# Function to early stop with root mean squared percentage error
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def feval_rmspe(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'RMSPE', rmspe(y_true, y_pred), False

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred,squared=True)