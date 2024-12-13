import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib
from lasso_selection import lasso_sel
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import os
from collections import Counter





def cross_validation(cv = 5, clf = None, x = None, y = None,
                     F1 = False, Accuracy = False,
                     Precision = False, Recall = False,
                     use_lasso = False):
    f1_val_scores = []
    f1_train_scores = []
    acc_val_scores = []
    acc_train_scores = []
    pre_val_scores = []
    pre_train_scores = []
    re_val_scores = []
    re_train_scores = []
    

    kf = StratifiedKFold(n_splits=cv, random_state=0, shuffle=True)
    clf = clf

    # Convert y to NumPy array
    # y = np.array(y)

    for train_index, val_index in kf.split(x,y):
        
        # Debug: Ensure no overlap between training and validation indices
        assert set(train_index).isdisjoint(set(val_index)), "Data leakage: Training and validation sets overlap!"
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        #use lasso for feature selection
        if use_lasso:
            try:
                opt_alpha, x_train = lasso_sel(x_train, y_train)
                coef_mask = x_train.shape[1]
                x_val = x_val[:, :coef_mask]
            except Exception as e:
                print(f"Error during Lasso feature selection: {e}")
                continue

        clf.fit(x_train, y_train)

        pred_val = clf.predict(x_val)

        pred_train = clf.predict(x_train)
        
        #metrics
        f1_val = metrics.f1_score(y_val, pred_val, average = 'weighted')
        f1_train = metrics.f1_score(y_train, pred_train, average = 'weighted')
        
        acc_val = metrics.accuracy_score(y_val, pred_val)
        acc_train = metrics.accuracy_score(y_train, pred_train)

        pre_val = metrics.precision_score(y_val, pred_val, average = 'weighted')
        pre_train = metrics.precision_score(y_train, pred_train, average = 'weighted')

        re_val = metrics.recall_score(y_val, pred_val, average = 'weighted')
        re_train = metrics.recall_score(y_train, pred_train, average = 'weighted')
        
        f1_val_scores.append(f1_val)
        f1_train_scores.append(f1_train)

        acc_val_scores.append(acc_val)   
        acc_train_scores.append(acc_train) 

        pre_val_scores.append(pre_val)   
        pre_train_scores.append(pre_train)

        re_val_scores.append(re_val)   
        re_train_scores.append(re_train)
    if F1 == True:
        return f1_val_scores, f1_train_scores
    if Accuracy == True:
        return acc_val_scores, acc_train_scores
    if Precision == True:
        return pre_val_scores, pre_train_scores
    if Recall == True:
        return re_val_scores, re_train_scores
    
def plot(CV = None, train_metrics = None, val_metrics = None):
    Fold = list(range(1, CV + 1))
    train_metric = pd.Series(train_metrics)
    val_metric = pd.Series(val_metrics)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Fold, train_metrics, c = 'b', marker = "^",ls='--',label='Greedy',fillstyle='none')
    ax.plot(Fold, val_metrics, c='g',marker=(8,2,0),ls='--',label='Greedy Heuristic')
    plt.title('Overall Situation of Fitting')
    plt.legend(['Train_F1','Test_F1'])
    plt.show()


def predict(clf = None,
            F1 = None,
            Accuracy = None,
            Recall = None,
            uselasso = False,
            pca = False,
            X_train = None, y_train = None, X_test = None, y_test = None):
    scaler = StandardScaler()
    # X_train, X_test, y_train, y_test = joblib.load(file_path)
    if uselasso == True:
        try:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            best_alpha, X_train = lasso_sel(X_train, y_train)
            coef_mask = X_train.shape[1]
            X_test = X_test[:, :coef_mask]
            print(f"Check Shape of X Matrix: {np.shape(X_train)}")
        except Exception as e:
            print(f"Error during Lasso feature selection: {e}")
            return None
    if pca == True:
        try: 
            normalized_train = scaler.fit_transform(X_train)
            normalized_test = scaler.transform(X_test)
            pca = PCA(n_components=0.9,
            random_state=112755)
            transformed_train = pca.fit_transform(normalized_train)
            transformed_test = pca.transform(normalized_test)
            explained_variance = pca.explained_variance_
            total_variance = explained_variance.sum()
            print(f"total variance retained: {total_variance:.2%}")
            print(f"Check Shape of X Matrix: {np.shape(transformed_train)}")
            X_train = transformed_train
            X_test = transformed_test
        except Exception as e:
            print(f"Error during PCA Transformation: {e}")
            


    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    # print out metrics
    if F1 == True:
        return metrics.f1_score(y_test, pred, average='weighted')
    elif Accuracy == True:
        return metrics.accuracy_score(y_test, pred)
    elif Recall == True:
        return metrics.recall_score(y_test, pred)
    else:
        return metrics.precision_score(y_test, pred)
    
def gird_search(model, param_grid, X_train, y_train, cv = 5, scoring='f1_weighted'):
    grid_search = GridSearchCV(estimator = model,
                               param_grid = param_grid,
                               cv = cv,
                               scoring = scoring,
                               verbose=0,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)
    # print(f"Best Parameters: {grid_search.best_params_}")
    # print(f"Best Score: {grid_search.best_score_}")

    best_model = grid_search.best_estimator_
    return best_model, grid_search

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def saveF1(score, filename):
    with open (filename, 'w') as file:
        file.write(f"weighted F1 score: {score}\n")
    print(f"Weighted F1 score saved to {filename}")


def logistic_gird(X_train, X_test, y_train, y_test,
                  dataset_name = None):
    param_grid_lr = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs'],
        'max_iter': [1000, 5000, 10000]
    }

    model, bestpara = gird_search(model = LogisticRegression(max_iter=10000,
                                                             random_state=112257), 
                                                             param_grid = param_grid_lr, 
                                                             scoring='f1_weighted',
                           X_train=X_train, y_train=y_train)
    save_model(model, r"C:\\Users\\leeyo\\OneDrive\\Desktop\\STA_221\\model\\lr_{dataset_name}.pkl")
    pred = predict(clf = model, uselasso=False, pca=True, F1=True, X_train=X_train, X_test=X_test,
                   y_train = y_train, y_test=y_test)
    saveF1(pred, r"C:\\Users\\leeyo\\OneDrive\\Desktop\\STA_221\\Result\\logistic_f1_score_{dataset_name}.txt")
    print(f"Completed Logistic Regression for {dataset_name}")
    print("Weighted F1 Score:", pred)
    print("Optimal Parameter:", bestpara)

def rf_gird(X_train, X_test, y_train, y_test,
            dataset_name = None):
    param_grid_rf = {
        'n_estimators': [50, 100, 200],  # Number of trees in the forest
        'max_depth': [10, 20, 50],      # Maximum depth of the trees
        'min_samples_split': [5, 10],      # Minimum number of samples to split a node
        'min_samples_leaf': [1, 2],        # Minimum number of samples at a leaf node
        'max_features': ['log2', 'sqrt']  # Number of features to consider at each split
    }

    model, bestpara = gird_search(model = RandomForestClassifier(random_state=42), 
                                  param_grid = param_grid_rf,
                                  scoring='f1_weighted',
                           X_train=X_train, y_train=y_train)
    save_model(model, r"C:\Users\leeyo\OneDrive\Desktop\STA_221\model\rf_{dataset_name}.pkl")
    pred = predict(clf = model, uselasso=False, pca=True, F1=True, X_train=X_train, X_test=X_test,
                   y_train = y_train, y_test=y_test)
    saveF1(pred, r"C:\Users\leeyo\OneDrive\Desktop\STA_221\Result\rf_f1_score_{dataset_name}.txt")
    print("Weighted F1 Score:", pred)
    print("Optimal Parameter:", bestpara)
    
def run(model = 'lr'):
    res = []
    file = [r"C:\\Users\\leeyo\\OneDrive\\Desktop\\STA_221\Data\\split_data_wo_gan.pkl",
            r"C:\\Users\\leeyo\\OneDrive\\Desktop\\STA_221\\Data\\split_data_gan.pkl"
            ]
    for f in file:
        try:
            X_train, X_test, y_train, y_test = joblib.load(f)
            print(f"Training set distribution: {Counter(y_train)}")
            print(f"Testing set distribution: {Counter(y_test)}")
            print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

            # dataset_name = f.split("\\")[-1].split('.')[0]  # Extract 'split_data_wo_gan' or 'split_data_gan'
            dataset_name = os.path.basename(f).split('.')[0]
            print(f"Processing dataset: {dataset_name}")
            # Debug print
            print(f"Processing file: {f}")
            print(f"X_train type: {type(X_train)}, shape: {np.shape(X_train)}")
            print(f"y_train type: {type(y_train)}, shape: {np.shape(y_train)}")

            # check and ensure data consistency
            assert len(X_train) == len(y_train)
            assert len(X_test) == len(y_test)

            if model == 'rf':
                pred = rf_gird(X_train, X_test, y_train, y_test, dataset_name=dataset_name)
            else:
                pred = logistic_gird(X_train, X_test, y_train, y_test, dataset_name=dataset_name)

                #Ensure pred is not None
            if pred is None:
                print(f"Warning: No result returned for {dataset_name}")
            else:
                res.append(pred)

        except Exception as e:
            print(f"Error processing file {file}: {e}")
    return res

        







    



if __name__ == '__main__':

    res = run(model='rf')
    print("Results:")
    for i, result in enumerate(res, 1):
        print(f"Dataset {i}:")
        print( f"  Model Path: {result['model_path']}")
        print(f"  F1 Score: {result['f1_score']}")
        print(f"  Best Parameters: {result['best_params']}")


  