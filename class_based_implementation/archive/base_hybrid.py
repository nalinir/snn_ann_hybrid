# NO LONGER USED
import torch
from class_based_implementation.models.base_temporal_model import BaseTemporalModel
import torch.nn as nn

class HybridBaseModel(BaseTemporalModel): # Inherit directly from BaseTemporalModel
    def __init__(self,
                 input_features: int,
                 hidden_features: int,
                 output_features: int,
                 data_config: dict,
                 model_type: str, # model_type will be passed by concrete hybrid classes
                 spike_fn = None, # SurrGradSpike class or similar
                 snn_mask = None, # Pass the existing mask or None to generate it
                 recurrent: bool = True,
                 learning_rate: float = 1e-3,
                 loss_fn: nn.Module = nn.CrossEntropyLoss(),
                 zenke_config: dict = None,
                 optimizer_name: str = "Adam",
                 spike_grad_scale: float = 100.0,
                 weight_scale: float = 1,
                 dtype: torch.dtype = torch.float32,
                 max_logit: bool = True,
                 shuffle_neurons: bool = False, # New property
                 gradient_clip_val: float = None, # Add this parameter
                 gradient_clip_algorithm: str = None, # Add this parameter
                 percent_snn: float = None, # This is only relevant for hybrid
                #  nonspikingneuron: bool = False, # New property
                 new_class_implementation: str = "Fixed units", # Tracking for wandb
                 flexible_spiking: bool = False, # New property
                ):

        # Common __init__ logic for hybrid models
        # if hidden_features % 2 != 0:
        #     raise ValueError("hidden_features must be an even number for an exact 50/50 SNN/ANN split.")
        if snn_mask is None:
            if percent_snn is None:
                multiplier = 0.5
            else:
                multiplier = percent_snn

            num_snn_neurons = int(hidden_features * multiplier)
            print(f"Number of SNN neurons: {num_snn_neurons}")
            snn_mask = torch.cat([
                torch.ones(num_snn_neurons),
                torch.zeros(hidden_features - num_snn_neurons)
            ])

            if shuffle_neurons:
                snn_mask = snn_mask[torch.randperm(hidden_features)]

        # Call BaseTemporalModel's __init__ with the generated mask and passed model_type
        super().__init__(
            input_features=input_features,
            hidden_features=hidden_features,
            output_features=output_features,
            data_config=data_config,
            model_type=model_type, # Pass the specific model_type
            spike_fn=spike_fn,
            recurrent=recurrent,
            learning_rate=learning_rate,
            snn_mask=snn_mask, # Pass the generated mask
            loss_fn=loss_fn,
            zenke_config=zenke_config,
            optimizer_name=optimizer_name,
            spike_grad_scale=spike_grad_scale,
            weight_scale=weight_scale,
            dtype=dtype,
            max_logit=max_logit,
            shuffle_neurons=shuffle_neurons, # Pass the new property
            # nonspikingneuron=nonspikingneuron, # Pass the new property
            gradient_clip_algorithm=gradient_clip_algorithm,
            gradient_clip_val=gradient_clip_val,
            percent_snn=percent_snn,
            new_class_implementation=new_class_implementation
        )
        self.save_hyperparameters() # Save hyperparameters after super().__init__

        # Common validation checks for hybrid models
        if self.snn_mask is None:
            raise ValueError("snn_mask must be provided (or generated) for HybridBaseModel subclasses.")
        if self.spike_fn is None:
            raise ValueError("spike_fn must be provided for HybridBaseModel subclasses.")
