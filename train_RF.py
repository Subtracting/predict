from matplotlib import pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, brier_score_loss
from sklearn import decomposition

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import numpy as np
import pickle
import pandas as pd

pd.options.mode.chained_assignment = None


def conf_mat(y_test, y_pred):

    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print("Confusion matrix:\n", conf_mat)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    plt.xlabel("Predicted")
    plt.ylabel("Expected")
    plt.show()


def split_data(df):

    X = df.drop("target", axis=1)
    y = df.target

    splits = int(len(df) * 0.7)

    X_train = X[:splits]
    y_train = y[:splits]

    X_test = X[splits:]
    y_test = y[splits:]

    return X_train, y_train, X_test, y_test


def train_model(clf, X_train, y_train, X_test, y_test, tscv):

    model_name = "RF_tuned.pkl"

    std_slc = StandardScaler()

    pipe = Pipeline(steps=[("std_slc", std_slc), ("random_Forest", clf)])

    random_grid = {
        "bootstrap": [True, False],
        "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10],
        "n_estimators": [200, 400, 600],
    }  # , 800, 1000, 1200, 1400, 1600, 1800, 2000

    # grid = GridSearchCV(pipe, param_grid=param_grid, scoring='f1',
    #                     cv=tscv, verbose=2, n_jobs=-1)

    grid = RandomizedSearchCV(
        estimator=clf,
        param_distributions=random_grid,
        n_iter=4,
        cv=tscv,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    best_grid = grid.best_estimator_
    best_param = grid.best_params_

    print("\n")
    print("====== :results: ======")

    predictions = best_grid.predict(X_test)

    print(classification_report(y_test, predictions))
    print(cross_val_score(best_grid, X_test, y_test, cv=tscv))

    plt.show()

    print("\n")
    print("====== :best_grid: ======")

    print(best_grid)
    print(best_param)

    # confusion matrix plot
    conf_mat(y_test, predictions)

    # # save model
    with open(model_name, "wb") as f:
        pickle.dump(best_grid, f)
