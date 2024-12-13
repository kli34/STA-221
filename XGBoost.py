from xgboost import XGBClassifier
from sklearn.model_selection import  train_test_split
from joblib import load
import joblib
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

data, label = load("image_dataset.joblib")
f = '/Users/lizc/Downloads/split_data_wo_gan.pkl'
X_train, X_test, Y_train, Y_test = joblib.load(f)

params = {
    "max_depth": [3, 6, 9],
    "learning_rate": [0.1, 0.2], 
    "n_estimators": [50, 100, 200], 
    "subsample": [0.6, 0.8, 1.0], 
    "scale_pos_weight": [1048182/393]
}

def CV_XGBoost(params, method):

    model = XGBClassifier()
    scorer = make_scorer(f1_score, average = "weighted")

    if method == "RandomSearch":
        CV = RandomizedSearchCV(estimator = model, param_distributions = params, scoring = scorer, cv = 5)
        CV.fit(X_train, Y_train)
    elif method == "GridSearch":
        CV = GridSearchCV(estimator= model, param_grid= params, scoring= scorer, cv= 5)
        CV.fit(X_train, Y_train)

    return CV

def model_fit():
    model = XGBClassifier()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)
    print("Confusion Matrix:\n", cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues")
    plt.show()


if __name__ == "__main__":
    
    model = CV_XGBoost(params= params, method= "RandomSearch")
    print(model.best_score_)
    print(model.best_params_)
    Y_predict = model.predict(X_test)
    ac = accuracy_score(Y_test, Y_predict)
    print("Accuracy: ", ac)
    cm = confusion_matrix(Y_test, Y_predict)
    print(cm)
    cm_plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    cm_plot.plot(cmap="Blues")
    plt.show()



