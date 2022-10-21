from dataclasses import dataclass


@dataclass
class Paths:
    log: str
    data: str
    model: str


@dataclass
class TrainingParams:
    batch_size: int
    epochs: int
    lr: float


@dataclass
class ModelParams:
    n_token: int
    n_token_layer: int
    n_hidden_layer: int


@dataclass
class CifarConf:
    paths: Paths
    training_params: TrainingParams
    model_params: ModelParams
