from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from xgboost import XGBRFClassifier


def general_models():
    general_model_pipeline = []
    general_model_list = ["Logistic Regression", "SGD", "KNN", "XGBRF", "Naive Bayes"]

    general_model_pipeline.append(LogisticRegression(solver="liblinear"))
    general_model_pipeline.append(SGDClassifier(max_iter=1000)),
    general_model_pipeline.append(KNeighborsClassifier())
    # general_model_pipeline.append(DecisionTreeClassifier())
    # general_model_pipeline.append(RandomForestClassifier())
    general_model_pipeline.append(XGBRFClassifier(n_estimators=100, subsample=0.9))
    general_model_pipeline.append(GaussianNB())

    return general_model_pipeline, general_model_list


def lr_models():
    lr_model_pipeline = []
    lr_model_list = [
        "lbfgs",
        "liblinear",
        "newton-cg",
        "newton-cholesky",
        "sag",
        "saga",
    ]

    lr_model_pipeline.append(LogisticRegression(solver="lbfgs", max_iter=1000))
    # lr_model_pipeline.append(LogisticRegression(
    #     solver='liblinear', max_iter=1000))
    # lr_model_pipeline.append(LogisticRegression(
    #     solver='newton-cg', max_iter=1000))
    # lr_model_pipeline.append(LogisticRegression(
    #     solver='newton-cholesky', max_iter=1000))
    # lr_model_pipeline.append(LogisticRegression(solver='sag', max_iter=1000))
    # lr_model_pipeline.append(LogisticRegression(solver='saga', max_iter=1000))

    return lr_model_pipeline, lr_model_list


def svc_models():
    svc_model_pipeline = []
    svc_model_list = ["linear", "poly", "rbf", "sigmoid"]

    svc_model_pipeline.append(SVC(kernel="linear"))
    svc_model_pipeline.append(SVC(kernel="poly"))
    svc_model_pipeline.append(SVC(kernel="rbf"))
    svc_model_pipeline.append(SVC(kernel="sigmoid"))

    return svc_model_pipeline, svc_model_list


def forest_models():
    forest_model_pipeline = []
    forest_model_list = ["RF_100", "XGB_100"]

    forest_model_pipeline.append(RandomForestClassifier(n_estimators=100))
    forest_model_pipeline.append(XGBRFClassifier(n_estimators=100, subsample=0.9))
    # forest_model_pipeline.append(RandomForestClassifier(n_estimators=1000))
    # forest_model_pipeline.append(RandomForestClassifier(n_estimators=1000))

    return forest_model_pipeline, forest_model_list
