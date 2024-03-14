from graph_cl.cli.dataset.cli import process_dataset, download_dataset


def test_process_dataset():
    assert process_dataset("jackson") == None
