import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.linear_model import LassoCV


def lasso_sel(x, y):
    best_score = 0
    alpha = np.logspace(-4, 1, 50)  # From 10^-4 to 10
    opt_alpha = None

    for a in alpha:
        lasso = Lasso(alpha=a, random_state=42)
        lasso.fit(x,y)
        coef_mask  = lasso.coef_ != 0
        xsel = x[:, coef_mask]

        if xsel.shape[1] == 0:
            continue

        #score from validation
        score = cross_val_score(LogisticRegression(max_iter=1000), xsel, y, cv = 5, scoring='f1')
        mean_score = np.mean(score)

        #select model

        if mean_score > best_score:
            opt_alpha = a
            Bestlasso = Lasso(alpha=opt_alpha).fit(x,y)
            Bestlasso_coef = Bestlasso.coef_ != 0 
            finalX = x[:, Bestlasso_coef]
    print("Best Alpha:", opt_alpha)
    return opt_alpha, finalX


if __name__ == '__main__':
    file_path = '/Users/kuangli/Desktop/UCD/Fall_2024/STA_221/Vision/split_data.pkl'
    X_train, X_test, y_train, y_test = joblib.load(file_path)

    bestaplha, _ = lasso_sel(X_train, y_train)
    print(bestaplha)
