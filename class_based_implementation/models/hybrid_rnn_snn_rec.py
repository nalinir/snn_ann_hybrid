import torch
from class_based_implementation.models.base_temporal_model import BaseTemporalModel

class Hybrid_RNN_SNN_rec(BaseTemporalModel):
    """
    A hybrid SNN/ANN model with separate recurrent connections.
    """
    def __init__(self, **kwargs):
        # Default to a 50/50 split unless specified otherwise by the user
        kwargs.setdefault('percent_snn', 0.5)
        kwargs.setdefault('separate_recurrent_paths', True)
        kwargs.setdefault('non_spiking_type', 'ann')
        
        # Call the main base model with the configuration for this specific architecture
        super().__init__(
            # non_spiking_type='ann',           # Set the non-spiking part to be a standard ANN
            # separate_recurrent_paths=True,  # Enable separate recurrent paths
            **kwargs
        )

class Hybrid_NSN_SNN_rec(BaseTemporalModel):
    """
    A hybrid SNN/NSN model with separate recurrent connections.
    """
    def __init__(self, **kwargs):
        # Default to a 50/50 split unless specified otherwise by the user
        kwargs.setdefault('percent_snn', 0.5)
        kwargs.setdefault('separate_recurrent_paths', True)
        kwargs.setdefault('non_spiking_type', 'nsn')
        
        # Call the main base model with the configuration for this specific architecture
        super().__init__(
            # non_spiking_type='nsn',           # Set the non-spiking part to be a stateful NSN
            # separate_recurrent_paths=True,  # Enable separate recurrent paths
            **kwargs
        )
