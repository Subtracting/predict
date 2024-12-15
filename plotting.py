import seaborn as sns
from matplotlib import pyplot as plt


def conf_matrix(cm_list, model_list):
    fig = plt.figure(figsize=(18, 10))

    for i in range(len(cm_list)):
        cm = cm_list[i]
        model = model_list[i]
        sub = fig.add_subplot(2, 3, i + 1).set_title(model)
        cm_plot = sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")
        cm_plot.set_xlabel("Predicted Values")
        cm_plot.set_ylabel("Actual Values")

    plt.show()
