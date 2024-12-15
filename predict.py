import pickle
from prepare import prepare_data


# load model
def load_model(model):
    with open(model, "rb") as f:
        clf = pickle.load(f)
    return clf


def predict(clf, df_act):

    # predict
    for i in range(len(df_act)):
        pred = clf.predict_proba(df_act.iloc[[i]])
        prediction = clf.predict(df_act.iloc[[i]])[0]


if __name__ == "__main__":

    clf = load_model("model_name.pkl")

    print(clf)

    df_act = prepare_data()

    print("data prepared")

    predict(clf, df_act)
