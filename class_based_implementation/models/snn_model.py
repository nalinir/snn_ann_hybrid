import torch
from class_based_implementation.models.base_temporal_model import BaseTemporalModel

class SNN(BaseTemporalModel):
    """A pure Spiking Neural Network (100% SNN neurons)."""
    def __init__(self, **kwargs):
        kwargs.setdefault('percent_snn', 1.0)
        super().__init__(**kwargs)
