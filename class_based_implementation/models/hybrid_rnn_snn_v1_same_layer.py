import torch
from class_based_implementation.models.base_temporal_model import BaseTemporalModel

class Hybrid_RNN_SNN_V1_same_layer(BaseTemporalModel):
    """
    A hybrid SNN/ANN model with shared recurrent connections.
    """
    def __init__(self, **kwargs):
        # Default to a 50/50 split unless specified otherwise by the user
        kwargs.setdefault('percent_snn', 0.5)
        kwargs.setdefault('non_spiking_type', 'ann')
        kwargs.setdefault('separate_recurrent_paths', False) # This is the default in base
        
        # Call the main base model with the configuration for this specific architecture
        super().__init__(
            # non_spiking_type='ann',            # Set the non-spiking part to be a standard ANN
            # separate_recurrent_paths=False,  # Disable separate recurrent paths (shared connections)
            **kwargs
        )

class Hybrid_NSN_SNN_V1_same_layer(BaseTemporalModel):
    """
    A hybrid SNN/NSN model with shared recurrent connections.
    """
    def __init__(self, **kwargs):
        # Default to a 50/50 split unless specified otherwise by the user
        kwargs.setdefault('percent_snn', 0.5)
        kwargs.setdefault('non_spiking_type', 'nsn')
        kwargs.setdefault('separate_recurrent_paths', False) # This is the default in base
        
        # Call the main base model with the configuration for this specific architecture
        super().__init__(
            # non_spiking_type='nsn',             # Set the non-spiking part to be a stateful NSN
            # separate_recurrent_paths=False,   # Disable separate recurrent paths (shared connections)
            **kwargs
        )


class Hybrid_NSN_SNN_V1_Flexible_Spiking(BaseTemporalModel):
    """
    A hybrid SNN/NSN model with shared recurrent connections.
    """
    def __init__(self, **kwargs):
        # Remove it completely to allow flexible spiking
        kwargs.setdefault('flexible_spiking', True)
        kwargs.setdefault('gamma_init_type', 'rand')
        kwargs.setdefault('non_spiking_type', 'nsn')
        kwargs.setdefault('separate_recurrent_paths', False) # This is the default in base, I should change that
        kwargs.pop('percent_snn', None)  # Remove percent_snn if present
        
        # Call the main base model with the configuration for this specific architecture
        super().__init__(
            # non_spiking_type='nsn',             # Set the non-spiking part to be a stateful NSN
            # separate_recurrent_paths=False, 
            # flexible_spiking=True,
            # gamma_init_type='rand',  # Enable flexible spiking
            **kwargs
        )
