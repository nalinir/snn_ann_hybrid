import os
import json
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

from class_based_implementation.surr_grad import SurrGradSpike
from class_based_implementation.zenke_regularizer import regularization_loss_zenke
from class_based_implementation.models.base_temporal_model import BaseTemporalModel

class TwoLayerTemporalModel(BaseTemporalModel):
    """
    A two-hidden-layer version of the BaseTemporalModel.
    Hidden Layer 1 -> Hidden Layer 2 -> Output.
    """
    def __init__(self, **kwargs):
        # Ensure the model type is correctly set for identification/logging
        kwargs['model_type'] = 'TwoLayerTemporalModel'
        super().__init__(**kwargs)
        # Re-save hyperparameters to capture the updated model_type
        self.save_hyperparameters(ignore=['spike_fn', 'loss_fn'])
        # Cleanup inherited single-layer properties after super().__init__
        if hasattr(self, 'gamma'): delattr(self, 'gamma')
        if hasattr(self, 'snn_mask'): delattr(self, 'snn_mask')

    # --- Overrides for Initialization ---

    def _init_weights(self, dtype, weight_scale):
        # Layer 1: Input to Hidden 1 (H1)
        w1 = torch.empty((self.input_features, self.hidden_features), dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(w1, mean=0.0, std=weight_scale / np.sqrt(self.input_features))
        self.w1 = nn.Parameter(w1)

        # Layer 2: Hidden 1 (H1) to Hidden 2 (H2)
        # Assuming H2 has the same feature size as H1 for simplicity
        w1_2 = torch.empty((self.hidden_features, self.hidden_features), dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(w1_2, mean=0.0, std=weight_scale / np.sqrt(self.hidden_features))
        self.w1_2 = nn.Parameter(w1_2)

        # Output Layer: Hidden 2 (H2) to Output
        w2 = torch.empty((self.hidden_features, self.output_features), dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(w2, mean=0.0, std=weight_scale / np.sqrt(self.hidden_features))
        self.w2 = nn.Parameter(w2)

    def _init_recurrent(self, dtype, weight_scale):
        if self.hparams.recurrent:
            # Layer 1 Recurrent (H1 -> H1)
            v1_1 = torch.empty((self.hidden_features, self.hidden_features), dtype=dtype, requires_grad=True)
            torch.nn.init.normal_(v1_1, mean=0.0, std=weight_scale / np.sqrt(self.hidden_features))
            self.v1_1 = nn.Parameter(v1_1)
            
            # Layer 2 Recurrent (H2 -> H2)
            v1_2 = torch.empty((self.hidden_features, self.hidden_features), dtype=dtype, requires_grad=True)
            torch.nn.init.normal_(v1_2, mean=0.0, std=weight_scale / np.sqrt(self.hidden_features))
            self.v1_2 = nn.Parameter(v1_2)
        else:
            self.v1_1 = None
            self.v1_2 = None
    
    def _init_gamma_and_mask(self, hidden_features, dtype, snn_mask, model_type):
        """Initializes masks/gammas for two hidden layers."""
        
        # We must re-run the initialization logic for the base class since the
        # logic for single self.gamma and self.snn_mask is tied to the base class
        # initialization flow. We will now use layer-specific names.

        if self.hparams.flexible_spiking:
            self.gamma1 = self._init_gamma(hidden_features, dtype)
            self.gamma2 = self._init_gamma(hidden_features, dtype)
            # Total number of neurons in the two hidden layers combined
            self.num_snn_neurons_in_layer = hidden_features * 2 
            self.gamma = None # Ensure base property is unset
        else:
            # Layer 1 Mask (use provided snn_mask if available)
            mask1, num_snn1 = self._init_snn_mask(hidden_features, dtype, snn_mask, model_type)
            self.register_buffer('snn_mask1', mask1.unsqueeze(0))
            
            # Layer 2 Mask (use percent_snn setting)
            mask2, num_snn2 = self._init_snn_mask(hidden_features, dtype, None, model_type) 
            self.register_buffer('snn_mask2', mask2.unsqueeze(0))

            self.num_snn_neurons_in_layer = num_snn1 + num_snn2
            self.gamma1 = None
            self.gamma2 = None
            self.snn_mask = None # Ensure base property is unset
    
    # --- New Core Processing Method ---

    def _process_layer_timestep(self, layer_input, mem, syn, prev_hidden_out, recurrent_weight, snn_mask_or_gamma, non_spiking_type):
        """
        Processes a single timestep for a generic hidden layer.
        """
        # 1. Determine the neuron behavior mixing factor
        snn_mix = snn_mask_or_gamma
        ann_mix = 1.0 - snn_mix

        # 2. Calculate recurrent input
        recurrent_input = 0.0
        if self.hparams.recurrent and recurrent_weight is not None:
            # Simplified recurrent path for two layers, using the non-separated approach by default
            recurrent_input = torch.einsum("ab,bc->ac", prev_hidden_out, recurrent_weight)
        
        total_input_t = layer_input + recurrent_input
        
        # 3. Calculate potential SNN updates
        # Check if any part of the layer uses SNN
        snn_path_active = self.hparams.flexible_spiking or (snn_mix > 0).any()
        if snn_path_active: 
            mem_snn_next, syn_snn_next, spikes_t = self._snn_neuron_update_step(total_input_t, mem, syn)
        else:
            mem_snn_next, syn_snn_next, spikes_t = torch.zeros_like(mem), torch.zeros_like(syn), torch.zeros_like(mem)
        
        # 4. Calculate potential non-spiking updates
        # Check if any part of the layer uses ANN/NSN
        ann_path_active = self.hparams.flexible_spiking or (ann_mix > 0).any()
        if ann_path_active:
            if non_spiking_type == 'ann':
                rate_activations_t = self._ann_neuron_update_step(total_input_t)
                mem_ann_next, syn_ann_next = torch.zeros_like(mem), torch.zeros_like(syn)
            else: # 'nsn'
                mem_ann_next, syn_ann_next, rate_activations_t = self._nsn_neuron_update_step(total_input_t, mem, syn)
        else:
            mem_ann_next, syn_ann_next, rate_activations_t = torch.zeros_like(mem), torch.zeros_like(syn), torch.zeros_like(mem)

        # 5. Combine outputs and state updates
        hidden_out_t = (snn_mix * spikes_t) + (ann_mix * rate_activations_t)
        final_mem = (snn_mix * mem_snn_next) + (ann_mix * mem_ann_next)
        final_syn = (snn_mix * syn_snn_next) + (ann_mix * syn_ann_next)
            
        return final_mem, final_syn, hidden_out_t, snn_mix*spikes_t, ann_path_active*rate_activations_t

    # --- Overrides for Simulation and Loss ---
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor]:
        batch_size, num_time_steps, _ = x.shape
        device = x.device
        
        # Layer 1 States
        mem1 = torch.zeros((batch_size, self.hidden_features), device=device, dtype=torch.float32)
        syn1 = torch.zeros((batch_size, self.hidden_features), device=device, dtype=torch.float32)
        prev_hidden_out1 = torch.zeros((batch_size, self.hidden_features), device=device, dtype=torch.float32)
        
        # Layer 2 States
        mem2 = torch.zeros((batch_size, self.hidden_features), device=device, dtype=torch.float32)
        syn2 = torch.zeros((batch_size, self.hidden_features), device=device, dtype=torch.float32)
        prev_hidden_out2 = torch.zeros((batch_size, self.hidden_features), device=device, dtype=torch.float32)
        
        # Recorders
        hidden_out_rec2, spikes_rec1, ann_activations_rec1 = [], [], []
        spikes_rec2, ann_activations_rec2 = [], []

        # Mix factors (must be defined outside loop)
        snn_mask_or_gamma1 = self.gamma1 if self.hparams.flexible_spiking else self.snn_mask1.to(device)
        snn_mask_or_gamma2 = self.gamma2 if self.hparams.flexible_spiking else self.snn_mask2.to(device)

        for t in range(num_time_steps):
            # --- Layer 1 Update (Input -> H1) ---
            h1_input_t = torch.einsum("ab,bc->ac", (x[:, t], self.w1))
            mem1, syn1, hidden_out_t1, spikes_t1, ann_activations_t1 = \
                self._process_layer_timestep(
                    h1_input_t, mem1, syn1, prev_hidden_out1, self.v1_1, snn_mask_or_gamma1, self.hparams.non_spiking_type
                )
            prev_hidden_out1 = hidden_out_t1
            
            # --- Layer 2 Update (H1 -> H2) ---
            h2_input_t = torch.einsum("ab,bc->ac", (hidden_out_t1, self.w1_2))
            mem2, syn2, hidden_out_t2, spikes_t2, ann_activations_t2 = \
                self._process_layer_timestep(
                    h2_input_t, mem2, syn2, prev_hidden_out2, self.v1_2, snn_mask_or_gamma2, self.hparams.non_spiking_type
                )
            prev_hidden_out2 = hidden_out_t2

            # Record H2 output (final hidden layer)
            hidden_out_rec2.append(hidden_out_t2)
            
            # Record Spikes/Activations for regularization/metrics (both layers)
            spikes_rec1.append(spikes_t1)
            ann_activations_rec1.append(ann_activations_t1)
            spikes_rec2.append(spikes_t2)
            ann_activations_rec2.append(ann_activations_t2)

        # Output: H2 output tensor
        hidden_out_tensor = torch.stack(hidden_out_rec2, dim=1) 
        
        # Auxiliary Outputs: Concatenate spikes and non-spiking activations from both layers
        # The shape is (B, T, H1+H2)
        all_spikes_tensor = torch.cat([torch.stack(spikes_rec1, dim=1), torch.stack(spikes_rec2, dim=1)], dim=2)
        all_ann_activations_tensor = torch.cat([torch.stack(ann_activations_rec1, dim=1), torch.stack(ann_activations_rec2, dim=1)], dim=2)
        
        # Readout from H2
        h3_input = torch.einsum("abc,cd->abd", (hidden_out_tensor, self.w2))
        predictions = self._apply_readout_layer(h3_input)
        
        auxiliary_outputs = {
            # Use combined spikes for zenke loss and metrics
            'spikes': all_spikes_tensor, 
            'ann_activations': all_ann_activations_tensor,
            'membrane_potentials': None, # Omitted for complexity of two layers
            'synaptic_currents': None     # Omitted for complexity of two layers
        }
        
        return predictions, auxiliary_outputs, x, hidden_out_tensor
    
    def _gamma_loss(self):
        """Regularization to encourage gamma values towards 0 or 1 for both layers."""
        if self.gamma1 is None:
            return 0.0
        
        # Concatenate gammas from both layers
        all_gammas = torch.cat([self.gamma1, self.gamma2])
        gamma_reg = ((all_gammas * (1.0 - all_gammas)) ** 2).mean()
        return self.hparams.gamma_penalty_strength * gamma_reg

    def _common_step(self, batch, step_type: str):
        """
        Overrides the common step to correctly handle two layers for Zenke loss and gamma metrics.
        """
        inputs, targets = batch
        predictions, auxiliary_outputs, _, _ = self(inputs)
        
        # Determine the logits for loss/accuracy based on max_logit setting
        output_logits = predictions[:, -1]
        if self.hparams.max_logit:
            output_logits = torch.max(predictions, dim=1).values 
        else:
            output_logits = torch.sum(predictions, dim=1)

        main_loss = self.loss_fn(output_logits, targets)
        total_loss = main_loss
        acc_metric = getattr(self, f'{step_type}_accuracy')
        acc_metric.update(output_logits, targets)
        self.log(f'{step_type}_accuracy', acc_metric, on_step=False, on_epoch=True, prog_bar=True)
        
        spikes = auxiliary_outputs['spikes'] # (B, T, H1+H2)
        
        if self.num_snn_neurons_in_layer > 0:
            
            # --- Zenke Loss & Spike Metrics Calculation ---
            if self.hparams.flexible_spiking:
                # Use concatenated gammas and all spikes for zenke loss
                gamma_to_pass = torch.cat([self.gamma1, self.gamma2])
                snn_spikes = spikes
                # Determine the number of SNN neurons based on the binarized gamma
                num_snn_neurons = (gamma_to_pass >= 0.5).sum().item()
            else:
                # Use concatenated fixed masks for slicing spikes
                full_mask = torch.cat([self.snn_mask1.squeeze(0), self.snn_mask2.squeeze(0)]).bool()
                snn_spikes = spikes[:, :, full_mask]
                gamma_to_pass = None 
                # Use the pre-calculated number of SNN neurons
                num_snn_neurons = self.num_snn_neurons_in_layer
                
            if self.zenke_enabled:
                zenke_loss, lower_l2, upper_l1, upper_l2 = regularization_loss_zenke(
                    snn_spikes, self.zenke_config, gamma=gamma_to_pass
                )
                total_loss += zenke_loss
                
                # Zenke Loss Logging (REQUIRED)
                self.log(f'{step_type}_zenke_loss', zenke_loss, on_step=False, on_epoch=True)
                self.log(f'{step_type}_zenke_lower_l2', lower_l2, on_step=False, on_epoch=True)
                self.log(f'{step_type}_zenke_upper_l1', upper_l1, on_step=False, on_epoch=True)
                self.log(f'{step_type}_zenke_upper_l2', upper_l2, on_step=False, on_epoch=True)
            
            if num_snn_neurons > 0 and self.hparams.flexible_spiking:
                num_spikes_per_neuron = snn_spikes.sum(dim=[0, 1])
                neurons_spiking_at_least_once = (num_spikes_per_neuron > 0).float()
                percent_spiking_neurons_spiking_once = neurons_spiking_at_least_once.sum()/(gamma_to_pass.sum().item()+1e-6) * 100.0
                
                total_spikes_across_time = snn_spikes.sum(dim=[1])
                mean_spikes_per_batch = total_spikes_across_time.mean(dim=[0])

                average_spikes_per_spiking_neuron = mean_spikes_per_batch.sum().item()/(gamma_to_pass.sum().item()+1e-6)
            else:
                spiked_neurons = (snn_spikes.sum(dim=1) > 0).float()
                percent_spiking_neurons_spiking_once = spiked_neurons.mean().item() * 100.0
                average_spikes_per_spiking_neuron = snn_spikes.sum().item() / self.num_snn_neurons_in_layer

            self.log(f'{step_type}_avg_spikes_per_spiking_neuron', average_spikes_per_spiking_neuron, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{step_type}_percent_snn_neurons_spiking_once', percent_spiking_neurons_spiking_once, on_step=False, on_epoch=True, prog_bar=True)
            avg_firing_rate = snn_spikes.mean() * 1000 / self.hparams.data_config["time_step"]
            self.log(f'{step_type}_avg_firing_rate_hz', avg_firing_rate, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{step_type}_total_spikes', snn_spikes.sum(), on_step=False, on_epoch=True, prog_bar=True)

        if self.hparams.flexible_spiking:
            gamma_loss = self._gamma_loss() # Uses the new two-layer gamma loss
            total_loss += gamma_loss
            self.log(f'{step_type}_gamma_loss', gamma_loss, on_step=False, on_epoch=True)
            
            # Gamma metrics using the concatenated gamma
            all_gammas = torch.cat([self.gamma1, self.gamma2])
            
            num_spiking = (all_gammas >= 0.5).sum().item()
            self.log(f'{step_type}_num_spiking_neurons', num_spiking, on_step=False, on_epoch=True, prog_bar=True)
            
            avg_spiking_gamma = all_gammas[all_gammas >= 0.5].mean() if (all_gammas >= 0.5).any() else 0
            self.log(f'{step_type}_avg_spiking_gamma', avg_spiking_gamma, on_step=False, on_epoch=True, prog_bar=True)
            
            num_nonspiking = (all_gammas < 0.5).sum().item()
            self.log(f'{step_type}_num_nonspiking_neurons', num_nonspiking, on_step=False, on_epoch=True, prog_bar=True)
            
            avg_nonspiking_gamma = all_gammas[all_gammas < 0.5].mean() if (all_gammas < 0.5).any() else 0
            self.log(f'{step_type}_avg_nonspiking_gamma', avg_nonspiking_gamma, on_step=False, on_epoch=True, prog_bar=True)


        self.log(f'{step_type}_main_loss', main_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{step_type}_total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def binarize_gamma(self):
        """
        Set all gamma values to 0 or 1, whichever they are closest to for both layers.
        """
        if self.gamma1 is not None:
            with torch.no_grad():
                self.gamma1.data.copy_((self.gamma1 >= 0.5).float())
        if self.gamma2 is not None:
            with torch.no_grad():
                self.gamma2.data.copy_((self.gamma2 >= 0.5).float())