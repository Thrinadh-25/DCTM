from .mlp import MLPIDS
from .cnn import CNNIDS
from .rnn import RNNIDS
from .cnn_bilstm import CNNBiLSTMIDS

DEEP_MODELS = {
    "mlp": MLPIDS,
    "cnn": CNNIDS,
    "rnn": RNNIDS,
    "cnn_bilstm": CNNBiLSTMIDS,
}
