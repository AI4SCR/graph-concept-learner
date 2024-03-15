from graph_cl.cli.dataset.cli import process, download


def test_process_dataset():
    assert process("jackson") == None
