from sklearn.metrics import accuracy_score, mean_squared_error

def evaluate_classification(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse

