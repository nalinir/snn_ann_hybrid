import torch
from class_based_implementation.models.base_temporal_model import BaseTemporalModel

class ANN_with_LIF_output(BaseTemporalModel):
    """A pure Artificial Neural Network (0% SNN neurons, 'ann' type) with a LIF readout."""
    def __init__(self, **kwargs):
        kwargs.setdefault('percent_snn', 0.0)
        kwargs.setdefault('non_spiking_type', 'ann')
        super().__init__(**kwargs)

class NSN_with_LIF_output(BaseTemporalModel):
    """A pure Non-Spiking Neuron network (0% SNN neurons, 'nsn' type) with a LIF readout."""
    def __init__(self, **kwargs):
        kwargs.setdefault('percent_snn', 0.0)
        kwargs.setdefault('non_spiking_type', 'nsn')
        super().__init__(**kwargs)
