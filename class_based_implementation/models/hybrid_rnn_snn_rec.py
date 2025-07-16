import torch
from class_based_implementation.models.base_hybrid import HybridBaseModel

class Hybrid_RNN_SNN_rec(HybridBaseModel):
    """
    Hybrid RNN-SNN model with separate recurrent connections for ANN and SNN parts.
    Inherits from HybridBaseModel.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)