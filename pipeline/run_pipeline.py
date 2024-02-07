import graph_cl
from graph_cl.datasets.RawDataLoader import RawDataLoader
from graph_cl.configuration import CONFIG
from graph_cl.preprocessing.filter import filter_samples
from graph_cl.preprocessing.split import create_folds
from graph_cl.preprocessing.normalize import normalize_folds
from graph_cl.datasets.generate_graphs import generate_graphs

if __name__ == "__main__":
    RawDataLoader(CONFIG)()
    filter_samples()
    create_folds()
    normalize_folds()
