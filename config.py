from dataclasses import dataclass


@dataclass
class Paths:
    log: str
    data: str
    model: str


@dataclass
class Params:
    batch_size: int
    epochs: int
    lr: float


@dataclass
class CifarConf:
    paths: Paths
    params: Params
