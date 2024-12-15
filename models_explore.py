import pickle
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit

from prepare import prepare_data
from train_LR import split_data, train_model
from models import general_models, lr_models, svc_models, forest_models
from feat_import import feat_importance
from plotting import conf_matrix

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import datetime

models, model_list = lr_models()


def pipeline(X_train, y_train, X_test, y_test, models):

    for model in models:
        print("\n", datetime.datetime.now(), model)

        clf = make_pipeline(StandardScaler(), model)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        n_splits = 4
        tscv = TimeSeriesSplit(n_splits)

        class_report = classification_report(y_test, y_pred, output_dict=True)
        cv_scores = cross_val_score(clf, X_test, y_test, cv=tscv)

        fpr, tpr, _thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
        acc_score = round(metrics.accuracy_score(y_test, y_pred), 4)
        auc_score = round(metrics.auc(fpr, tpr), 4)

        # acc_list.append(metrics.accuracy_score(y_test, y_pred))
        # fpr, tpr, _thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
        # auc_list.append(round(metrics.auc(fpr, tpr), 4))
        # cm_list.append(confusion_matrix(y_test, y_pred))

        model_name = f"model_{model}.pkl"

        with open(model_name, "wb") as f:
            pickle.dump(clf, f)

        print(acc_score, auc_score)


if __name__ == "__main__":
    print("started at:", datetime.datetime.now())

    df = prepare_data()

    corr = df.corr()
    sns.heatmap(corr)
    plt.show()

    X_train, y_train, X_test, y_test = split_data(df)

    pipeline(X_train, y_train, X_test, y_test, models)
