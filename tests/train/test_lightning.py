from graph_cl.data_models.Train import TrainConfig
from graph_cl.data_models.Experiment import GCLExperiment
from graph_cl.train.lightning import LitGNN
from graph_cl.data_models.Model import ModelGNNConfig

import tempfile

factory = GCLExperiment(experiment_name="test")


def create_gnn_model():
    from graph_cl.data_models.Model import ModelGNNConfig
    from graph_cl.models.gnn import GNN_plus_MPL

    factory = GCLExperiment(experiment_name="test")
    model_config = ModelGNNConfig.model_validate_from_json(
        factory.model_gnn_config_path
    )
    model_config.num_classes = 2
    model_config.in_channels = 28

    model = GNN_plus_MPL(model_config.dict())
    return model


def test_LitGNN():
    factory = GCLExperiment(experiment_name="test")

    model_config = ModelGNNConfig.model_validate_from_json(
        factory.model_gnn_config_path, num_classes=2, in_channels=28
    )
    train_config = TrainConfig.model_validate_from_json(factory.pretrain_config_path)

    module = LitGNN(model_config=model_config, train_config=train_config)

    with tempfile.NamedTemporaryFile() as f:
        path = f.name

        # TODO: is this really the best way to save a pytorch lightning model without using fit?
        # https://lightning.ai/forums/t/saving-a-lightningmodule-without-a-trainer/2217/4
        from pytorch_lightning import Trainer

        trainer = Trainer()
        trainer.strategy.connect(module)

        trainer.save_checkpoint(path)
        model_loaded = LitGNN.load_from_checkpoint(path)
    assert True


def test_LitGCL():
    from graph_cl.train.lightning import LitGCL
    from graph_cl.models.graph_concept_learnerV2 import GraphConceptLearner
    from graph_cl.data_models.Model import ModelGCLConfig
    import torch.nn as nn

    gnn_models = {
        "concept1": create_gnn_model().get_submodule("gnn"),
        "concept2": create_gnn_model().get_submodule("gnn"),
    }

    model_config = ModelGCLConfig.model_validate_from_json(
        factory.model_gnn_config_path
    )
    train_config = TrainConfig.model_validate_from_json(factory.pretrain_config_path)

    graph_concept_learner = GraphConceptLearner(
        concept_learners=nn.ModuleDict(gnn_models),
        config=model_config.dict(),
    )
    gcl = LitGCL(model=graph_concept_learner, train_config=train_config)

    with tempfile.NamedTemporaryFile() as f:
        path = f.name

        # TODO: is this really the best way to save a pytorch lightning model without using fit?
        # https://lightning.ai/forums/t/saving-a-lightningmodule-without-a-trainer/2217/4
        from pytorch_lightning import Trainer

        trainer = Trainer()
        trainer.strategy.connect(gcl)

        trainer.save_checkpoint(path)
        model_loaded = LitGNN.load_from_checkpoint(path, model=graph_concept_learner)
    assert True
