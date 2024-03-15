from http.client import RemoteDisconnected
import time
from typing import Callable
from matplotlib import pyplot as plt
import numpy as np
import mlflow
from http.client import RemoteDisconnected
import time
from sklearn.metrics import confusion_matrix
import itertools
import tempfile
from pathlib import Path
import os


def robust_mlflow(
    f: Callable, *args, max_tries: int = 8, delay: int = 1, backoff: int = 2, **kwargs
):
    """
    This is a wrapper function for mlflow logging. It attemts to log using the ptovided
    function. If it fails it tries again.

    Args:
        f: logging function to be used.
        *args: positional arguments to pass to the fucntion.
        max_tries: max number of attempts to call the fucntion.
        delay: waiting time in between attempts.
        backoff: increase the delay between attempts by the product with the delay.
        **kwargs: key word arguments to pass to the fucntion.

    Returns:
        The return value of the corresponding logging function. Expect nothing.
    """
    while max_tries > 1:
        try:
            return f(*args, **kwargs)
        except RemoteDisconnected:
            print(f"MLFLOW remote disconnected. Trying again in {delay}s")
            time.sleep(delay)
            max_tries -= 1
            delay *= backoff
    return f(*args, **kwargs)


def make_confusion_matrix(prediction, ground_truth, classes):
    cm = confusion_matrix(
        y_true=ground_truth, y_pred=prediction, labels=np.arange(len(classes))
    )
    fig = plot_confusion_matrix(cm, classes, figname=None, normalize=False)
    return fig


def plot_confusion_matrix(
    cm, classes, figname=None, normalize=False, title=None, cmap=plt.cm.Oranges
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=18)
    plt.yticks(tick_marks, classes, fontsize=18)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(
                j,
                i,
                "%.2f" % cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16,
            )
        else:
            plt.text(
                j,
                i,
                cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16,
            )
    plt.tight_layout()
    plt.ylabel("True label", fontsize=20)
    plt.xlabel("Predicted label", fontsize=20)
    ax.imshow(cm, interpolation="nearest", cmap=cmap)
    if title is not None:
        ax.set_title(title)
    return fig


def save_fig_to_mlflow(fig, mlflow_dir, name):
    with tempfile.TemporaryDirectory() as temp_dir_name:
        file_name = Path(temp_dir_name) / f"{name}.png"
        fig.savefig(str(file_name), dpi=300, bbox_inches="tight")
        robust_mlflow(
            mlflow.log_artifact,
            str(file_name),
            artifact_path=mlflow_dir,
        )
        plt.close(fig=fig)


def start_mlflow_run(root, pred_target, out_dir):
    # Define mlflow experiment
    dataset_name = os.path.basename(root)
    mlflow.set_experiment(experiment_name=f"san_{dataset_name}_{pred_target}")

    # Make new mlflow run if none exists
    path_to_mlflow_run_id_file = os.path.join(out_dir, "mlflow_run_id.txt")

    if os.path.exists(path_to_mlflow_run_id_file):
        # Read in run_id
        with open(path_to_mlflow_run_id_file, "r") as f:
            run_id = f.read()

        # Start run with run id
        mlflow.start_run(run_id=run_id)
    else:
        mlflow.start_run()
        run = mlflow.active_run()
        run_id = run.info.run_id

        # write run_id to file
        with open(path_to_mlflow_run_id_file, "w") as f:
            run_id = f.write(run_id)
