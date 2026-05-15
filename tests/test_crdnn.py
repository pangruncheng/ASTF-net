from typing import Any, Dict

import pytest
import torch

from astfnet.models.base import ASTFModule
from astfnet.models.crdnn import CRDNN
from astfnet.models.rnn import RNNFactory

BATCH_SIZE = 3
SEQ_LEN = 256
OUTPUT_LENGTH = 128


@pytest.fixture
def crdnn_config() -> Dict[str, Any]:
    return {
        "model_name": "crdnn",
        "loss": "mse",
        "in_channels": 2,
        "output_length": OUTPUT_LENGTH,
        "cnn_channels": [16, 32],
        "cnn_kernel_size": 5,
        "cnn_pool_size": 2,
        "rnn_hidden_size": 24,
        "rnn_layers": 1,
        "rnn_bidirectional": True,
        "dnn_hidden_size": 64,
        "dnn_layers": 1,
        "dropout": 0.0,
    }


@pytest.mark.parametrize("rnn_name", ["GRU", "LSTM", "gru", "lstm"])
def test_rnn_factory_builds_supported_rnns(rnn_name: str) -> None:
    factory = RNNFactory(name=rnn_name, hidden_size=16, num_layers=1, bidirectional=True)
    rnn = factory.build(input_size=8)

    assert factory.name == rnn_name.upper()
    assert factory.output_size == 32
    assert isinstance(rnn, (torch.nn.GRU, torch.nn.LSTM))


def test_rnn_factory_rejects_unknown_rnn() -> None:
    with pytest.raises(ValueError, match="Unknown RNN"):
        RNNFactory(name="RNN")


@pytest.mark.parametrize("rnn_name", ["GRU", "LSTM"])
def test_crdnn_forward_shape(rnn_name: str, crdnn_config: Dict[str, Any]) -> None:
    kwargs = {k: v for k, v in crdnn_config.items() if k not in {"model_name", "loss"}}
    model = CRDNN(**{**kwargs, "rnn_name": rnn_name})
    target = torch.randn(BATCH_SIZE, SEQ_LEN)
    egf = torch.randn(BATCH_SIZE, SEQ_LEN)

    out = model(target, egf)

    assert out.shape == (BATCH_SIZE, OUTPUT_LENGTH)
    assert (out >= 0).all()


def test_crdnn_accepts_rnn_factory() -> None:
    factory = RNNFactory(name="LSTM", hidden_size=12, num_layers=1, bidirectional=False)
    model = CRDNN(
        output_length=OUTPUT_LENGTH,
        cnn_channels=[16],
        rnn_factory=factory,
        dnn_hidden_size=32,
        dropout=0.0,
    )

    out = model(torch.randn(2, SEQ_LEN), torch.randn(2, SEQ_LEN))

    assert out.shape == (2, OUTPUT_LENGTH)
    assert isinstance(model.rnn, torch.nn.LSTM)


@pytest.mark.parametrize("in_channels", [1, 3])
def test_crdnn_rejects_non_two_channel_configs(in_channels: int) -> None:
    with pytest.raises(ValueError, match="exactly 2 input channels"):
        CRDNN(in_channels=in_channels)


def test_astf_module_builds_crdnn_from_config(crdnn_config: Dict[str, Any]) -> None:
    model = ASTFModule({**crdnn_config, "rnn_name": "GRU"})
    batch = {
        "target": torch.randn(BATCH_SIZE, SEQ_LEN),
        "egf": torch.randn(BATCH_SIZE, SEQ_LEN),
        "astf": torch.abs(torch.randn(BATCH_SIZE, OUTPUT_LENGTH)),
    }

    out = model(batch["target"], batch["egf"])
    loss = model.training_step(batch, 0)

    assert out.shape == (BATCH_SIZE, OUTPUT_LENGTH)
    assert loss.requires_grad
