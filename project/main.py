import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from utils.preprocessing import preprocess_data
from utils.evaluation import evaluate_classification, evaluate_regression

# Set the paths
data_path = 'data/train.csv'
model_path = 'models/'

# Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(data_path)

# Train and save the logistic regression model for classification
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)
accuracy = evaluate_classification(y_test, logistic_predictions)
print("Logistic Regression Accuracy:", accuracy)
joblib.dump(logistic_model, os.path.join(model_path, 'logistic_regression_model.joblib'))

# Train and save the random forest model for classification
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
accuracy = evaluate_classification(y_test, rf_predictions)
print("Random Forest Classifier Accuracy:", accuracy)
joblib.dump(rf_classifier, os.path.join(model_path, 'random_forest_classifier_model.joblib'))

# Train and save the random forest model for regression
rf_regressor = RandomForestRegressor()
rf_regressor.fit(X_train, y_train)
rf_predictions = rf_regressor.predict(X_test)
mse = evaluate_regression(y_test, rf_predictions)
print("Random Forest Regressor MSE:", mse)
joblib.dump(rf_regressor, os.path.join(model_path, 'random_forest_regressor_model.joblib'))

