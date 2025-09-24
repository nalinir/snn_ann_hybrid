import os
import json
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

from class_based_implementation.surr_grad import SurrGradSpike
from class_based_implementation.zenke_regularizer import regularization_loss_zenke

class BaseTemporalModel(pl.LightningModule):
    """
    A base class for temporal models that can represent SNNs, ANNs, NSNs, and hybrid architectures.
    """
    def __init__(self,
                 input_features: int,
                 hidden_features: int,
                 output_features: int,
                 data_config: dict,
                 spike_fn = SurrGradSpike.apply,
                 recurrent: bool = True,
                 learning_rate: float = 1e-3,
                 loss_fn: nn.Module = nn.CrossEntropyLoss(),
                 zenke_config: dict = None,
                 optimizer_name: str = "Adam",
                 spike_grad_scale: float = 100.0,
                 weight_scale: float = 1,
                 dtype: torch.dtype = torch.float32,
                 max_logit: bool = True,
                 gradient_clip_val: float = None,
                 gradient_clip_algorithm: str = None,
                 percent_snn: float = 1.0,
                 flexible_spiking: bool = False,
                 separate_recurrent_paths: bool = None, # Changed 9/24
                 model_type: str = 'BaseTemporalModel', # Should be updated in subclasses
                 non_spiking_type: str = 'ann', # Can be 'ann' or 'nsn'
                 shuffle_neurons: bool = False, # <<< NEW PARAMETER ADDED HERE
                 gamma_penalty_strength: float = 1,
                 gamma_init_type: str = 'rand',
                 snn_mask: torch.Tensor = None,
                 new_class_implementation: str = "added back gamma for regularization",
                 gamma_fixed: bool = False,
                 reset_nsn: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=['spike_fn', 'loss_fn'])

        self._validate_non_spiking_type()
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        self.loss_fn = loss_fn
        self.spike_fn = spike_fn

        self._init_time_constants(data_config)
        self._init_accuracy_metrics(output_features)
        self._init_weights(dtype, weight_scale)
        self._init_recurrent(dtype, weight_scale)
        self._init_gamma_and_mask(hidden_features, dtype, snn_mask, model_type)
        self._init_zenke(zenke_config)

    def _validate_non_spiking_type(self):
        if self.hparams.non_spiking_type not in ['ann', 'nsn']:
            raise ValueError("non_spiking_type must be either 'ann' or 'nsn'")

    def _init_time_constants(self, data_config):
        self.alpha = float(np.exp(-data_config["time_step"] / data_config["tau_syn"]))
        self.beta = float(np.exp(-data_config["time_step"] / data_config["tau_mem"]))
        self.beta_out = float(np.exp(-data_config["time_step"] / data_config["tau_readout"]))

    def _init_accuracy_metrics(self, output_features):
        self.train_accuracy = Accuracy(task="multiclass", num_classes=output_features)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=output_features)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=output_features)

    def _init_weights(self, dtype, weight_scale):
        w1 = torch.empty((self.input_features, self.hidden_features), dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(w1, mean=0.0, std=weight_scale / np.sqrt(self.input_features))
        self.w1 = nn.Parameter(w1)

        w2 = torch.empty((self.hidden_features, self.output_features), dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(w2, mean=0.0, std=weight_scale / np.sqrt(self.hidden_features))
        self.w2 = nn.Parameter(w2)

    def _init_recurrent(self, dtype, weight_scale):
        if self.hparams.recurrent:
            v1 = torch.empty((self.hidden_features, self.hidden_features), dtype=dtype, requires_grad=True)
            torch.nn.init.normal_(v1, mean=0.0, std=weight_scale / np.sqrt(self.hidden_features))
            self.v1 = nn.Parameter(v1)
        else:
            self.v1 = None

    def _init_gamma_and_mask(self, hidden_features, dtype, snn_mask, model_type):
        self.gamma = None
        if self.hparams.flexible_spiking:
            self.gamma = self._init_gamma(hidden_features, dtype)
            self.num_snn_neurons_in_layer = hidden_features
        else:
            if not hasattr(self, 'snn_mask'):
                mask, num_snn = self._init_snn_mask(hidden_features, dtype, snn_mask, model_type)
                self.register_buffer('snn_mask', mask.unsqueeze(0))
                self.num_snn_neurons_in_layer = num_snn
            else:
                self.num_snn_neurons_in_layer = int(self.snn_mask.sum().item())

    def _init_gamma(self, hidden_features, dtype):
        gamma_init = torch.empty(hidden_features, dtype=dtype, requires_grad=True)
        init_type = self.hparams.gamma_init_type
        if init_type == 'rand':
            torch.nn.init.uniform_(gamma_init, a=0.0, b=1.0)
        elif init_type == 'const_half':
            torch.nn.init.constant_(gamma_init, 0.5)
        elif init_type == 'const_zero':
            torch.nn.init.constant_(gamma_init, 0.0)
        elif init_type == 'const_one':
            torch.nn.init.constant_(gamma_init, 1.0)
        if self.hparams.gamma_fixed and 'const' in init_type:  
            if hasattr(self, 'gamma'):
                delattr(self, 'gamma')
            self.register_buffer('gamma', gamma_init)
            return self.gamma
        elif not self.hparams.gamma_fixed:
            return nn.Parameter(gamma_init)
        else:
            return ValueError("If gamma_fixed is True, gamma_init_type must be one of 'const_half', 'const_zero', or 'const_one'.")
        
    def _init_snn_mask(self, hidden_features, dtype, snn_mask, model_type):
        if snn_mask is not None:
            mask = snn_mask.to(dtype)
            num_snn = int(mask.sum().item())
        else:
            percent_snn = self._resolve_percent_snn(model_type)
            num_snn = int(percent_snn * hidden_features)
            mask = torch.zeros(hidden_features, dtype=dtype)
            mask[:num_snn] = 1.0
            if self.hparams.shuffle_neurons:
                idx = torch.randperm(mask.nelement())
                mask = mask.view(-1)[idx].view(mask.size())
        return mask, num_snn

    def _resolve_percent_snn(self, model_type):
        if self.hparams.percent_snn is None and model_type == 'SNN':
            return 1.0
        elif self.hparams.percent_snn is None and model_type.startswith('Hybrid'):
            return 0.5
        elif self.hparams.percent_snn is None:
            return 0.0
        elif not (0.0 <= self.hparams.percent_snn <= 1.0):
            raise ValueError("percent_snn must be between 0.0 and 1.0")
        return self.hparams.percent_snn

    def _init_zenke(self, zenke_config):
        self.zenke_enabled = zenke_config and any(zenke_config.get(k, 0) != 0 for k in ["l2_lower", "l1_upper", "l2_upper"])
        self.zenke_config = zenke_config if self.zenke_enabled else {}

    def _lif_state_update(self, current_input, mem, syn, reset_signal):
        """Core state update logic for any LIF-based neuron (SNN or NSN)."""
        rst = reset_signal.detach()
        new_syn = self.alpha * syn + current_input
        new_mem = (self.beta * mem + (1 - self.beta) * syn) * (1.0 - rst)
        return new_mem, new_syn

    def _snn_neuron_update_step(self, current_input, mem, syn):
        """SNN dynamics: Spike output and hard reset."""
        mthr = mem - 1.0
        spikes = self.spike_fn(mthr, self.hparams.spike_grad_scale)
        new_mem, new_syn = self._lif_state_update(current_input, mem, syn, spikes)
        return new_mem, new_syn, spikes

    def _nsn_neuron_update_step(self, current_input, mem, syn):
        """NSN dynamics: Graded potential output and soft reset."""
        graded_output = torch.clamp(mem, 0, 1)
        if self.hparams.reset_nsn:
            reset = graded_output
        else: # Just make this all 0s
            reset = torch.zeros_like(graded_output)
        new_mem, new_syn = self._lif_state_update(current_input, mem, syn, reset)
        return new_mem, new_syn, graded_output

    def _ann_neuron_update_step(self, current_input):
        """Standard ANN neuron dynamics using tanh activation."""
        return torch.tanh(current_input)

    def _process_hidden_layer_timestep(self, h1_input_t, mem, syn, prev_hidden_out):
        """
        Processes a single timestep with a direct, unconstrained gamma parameter.
        """
        # 1. Determine the neuron behavior mixing factor
        snn_mix = self.gamma if self.hparams.flexible_spiking else self.snn_mask.to(h1_input_t.device)
        ann_mix = 1.0 - snn_mix

        # 2. Calculate recurrent input
        recurrent_input = 0.0
        if self.hparams.recurrent and self.v1 is not None:
            if self.hparams.separate_recurrent_paths:
                rec_from_snn = torch.einsum("ab,bc->ac", prev_hidden_out * snn_mix, self.v1)
                rec_from_ann = torch.einsum("ab,bc->ac", prev_hidden_out * ann_mix, self.v1)
                recurrent_input = (rec_from_snn * snn_mix) + (rec_from_ann * ann_mix)
            else:
                recurrent_input = torch.einsum("ab,bc->ac", prev_hidden_out, self.v1)
        
        total_input_t = h1_input_t + recurrent_input
        if self.hparams.flexible_spiking or (snn_mix > 0).any(): 
        # 3. Calculate potential SNN updates
            mem_snn_next, syn_snn_next, spikes_t = self._snn_neuron_update_step(total_input_t, mem, syn)
        else:
            mem_snn_next, syn_snn_next, spikes_t = torch.zeros_like(mem), torch.zeros_like(syn), torch.zeros_like(mem)
        
        # 4. Calculate potential non-spiking updates
        if self.hparams.flexible_spiking or (ann_mix > 0).any():
            if self.hparams.non_spiking_type == 'ann':
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
            
        return final_mem, final_syn, hidden_out_t, snn_mix*spikes_t, rate_activations_t

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor]:
        batch_size, num_time_steps, _ = x.shape
        device = x.device
        mem = torch.zeros((batch_size, self.hidden_features), device=device, dtype=torch.float32)
        syn = torch.zeros((batch_size, self.hidden_features), device=device, dtype=torch.float32)
        prev_hidden_out = torch.zeros((batch_size, self.hidden_features), device=device, dtype=torch.float32)
        hidden_out_rec, spikes_rec, ann_activations_rec, mem_rec, syn_rec = [], [], [], [], []
        for t in range(num_time_steps):
            h1_input_t = torch.einsum("ab,bc->ac", (x[:, t], self.w1))
            mem, syn, hidden_out_t, spikes_t, ann_activations_t = \
                self._process_hidden_layer_timestep(h1_input_t, mem, syn, prev_hidden_out)
            hidden_out_rec.append(hidden_out_t)
            spikes_rec.append(spikes_t)
            ann_activations_rec.append(ann_activations_t)
            mem_rec.append(mem)
            syn_rec.append(syn)
            prev_hidden_out = hidden_out_t
        hidden_out_tensor = torch.stack(hidden_out_rec, dim=1)
        spikes_tensor = torch.stack(spikes_rec, dim=1)
        ann_activations_tensor = torch.stack(ann_activations_rec, dim=1)
        mem_tensor = torch.stack(mem_rec, dim=1)
        syn_tensor = torch.stack(syn_rec, dim=1)
        h2_input = torch.einsum("abc,cd->abd", (hidden_out_tensor, self.w2))
        predictions = self._apply_readout_layer(h2_input)
        auxiliary_outputs = {
            'spikes': spikes_tensor,
            'ann_activations': ann_activations_tensor,
            'membrane_potentials': mem_tensor,
            'synaptic_currents': syn_tensor
        }
        return predictions, auxiliary_outputs, x, hidden_out_tensor
    
    def _apply_readout_layer(self, h2_input: torch.Tensor) -> torch.Tensor:
        flt = torch.zeros((h2_input.shape[0], self.output_features), device=h2_input.device, dtype=torch.float32)
        out = torch.zeros((h2_input.shape[0], self.output_features), device=h2_input.device, dtype=torch.float32)
        out_rec = []
        for t in range(h2_input.shape[1]):
            flt = self.alpha * flt + h2_input[:, t]
            out = self.beta_out * out + (1 - self.beta_out) * flt
            out_rec.append(out)
        return torch.stack([torch.zeros_like(out_rec[0])] + out_rec, dim=1)

    def _gamma_loss(self):
        """Regularization to encourage gamma values towards 0 or 1."""
        if self.gamma is None:
            return 0.0
        gamma_reg = ((self.gamma * (1.0 - self.gamma)) ** 2).mean()
        return self.hparams.gamma_penalty_strength * gamma_reg

    # TO LOOK INTO HOW THIS AGGREGATES IN WANDB ACROSS BATCHES
    def _common_step(self, batch, step_type: str):
        inputs, targets = batch
        predictions, auxiliary_outputs, _, _ = self(inputs)
        print(f"[DEBUG] predictions.shape: {predictions.shape}")
        print(f"[DEBUG] targets.shape: {targets.shape}")
        output_logits = torch.max(predictions, dim=1).values if self.hparams.max_logit else torch.sum(predictions, dim=1)
        print(f"[DEBUG] output_logits.shape: {output_logits.shape}")
        main_loss = self.loss_fn(output_logits, targets)
        print(f"[DEBUG] main_loss: {main_loss}")
        total_loss = main_loss
        acc_metric = getattr(self, f'{step_type}_accuracy')
        acc_metric.update(output_logits, targets)
        self.log(f'{step_type}_accuracy', acc_metric, on_step=False, on_epoch=True, prog_bar=True)
        if 'spikes' in auxiliary_outputs and self.num_snn_neurons_in_layer > 0:
            spikes = auxiliary_outputs['spikes']
            print(f"[DEBUG] spikes.shape: {spikes.shape}")
            # Include all indices for snn_spikes calculation
            if self.hparams.flexible_spiking:
                snn_spikes = spikes
            else:
                snn_neuron_indices = self.snn_mask.squeeze(0).bool()
                print(f"[DEBUG] snn_neuron_indices.sum(): {snn_neuron_indices.sum().item()}")
                snn_spikes = spikes[:, :, snn_neuron_indices]
                print(f"[DEBUG] snn_spikes.shape: {snn_spikes.shape}")
            if self.zenke_enabled:
                gamma_to_pass = self.gamma if self.hparams.flexible_spiking else None
                print(f"[DEBUG] gamma_to_pass.shape: {gamma_to_pass.shape if gamma_to_pass is not None else None}")
                zenke_loss, lower_l2, upper_l1, upper_l2 = regularization_loss_zenke(
                    snn_spikes, self.zenke_config, gamma=gamma_to_pass
                )
                print(f"[DEBUG] zenke_loss: {zenke_loss}")
                total_loss += zenke_loss
                # Log the regularization losses
                self.log(f'{step_type}_zenke_loss', zenke_loss, on_step=False, on_epoch=True)
                self.log(f'{step_type}_zenke_lower_l2', lower_l2, on_step=False, on_epoch=True)
                self.log(f'{step_type}_zenke_upper_l1', upper_l1, on_step=False, on_epoch=True)
                self.log(f'{step_type}_zenke_upper_l2', upper_l2, on_step=False, on_epoch=True)

            # Calculate percent of SNN neurons that spiked at least once during the entire sequence
            if self.hparams.flexible_spiking:
                num_snn_neurons = (self.gamma >= 0.5).sum().item()
                print(f"[DEBUG] num_snn_neurons (flexible): {num_snn_neurons}")
                num_spikes_per_neuron = snn_spikes.sum(dim=[0, 1])
                neurons_spiking_at_least_once = (num_spikes_per_neuron > 0).float()
                percent_spiking_neurons_spiking_once = neurons_spiking_at_least_once.sum()/(self.gamma.sum().item()+1e-6) * 100.0
                
                total_spikes_across_time = snn_spikes.sum(dim=[1])
                mean_spikes_per_batch = total_spikes_across_time.mean(dim=[0])

                average_spikes_per_spiking_neuron = mean_spikes_per_batch.sum().item()/(self.gamma.sum().item()+1e-6)
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
            gamma_loss = self._gamma_loss()
            print(f"[DEBUG] gamma_loss: {gamma_loss}")
            total_loss += gamma_loss
            self.log(f'{step_type}_gamma_loss', gamma_loss, on_step=False, on_epoch=True)
            # Count spiking
            num_spiking = (self.gamma >= 0.5).sum().item()
            print(f"[DEBUG] num_spiking: {num_spiking}")
            self.log(f'{step_type}_num_spiking_neurons', num_spiking, on_step=False, on_epoch=True, prog_bar=True)
            avg_spiking_gamma = self.gamma[self.gamma >= 0.5].mean() if (self.gamma >= 0.5).any() else 0
            print(f"[DEBUG] avg_spiking_gamma: {avg_spiking_gamma}")
            self.log(f'{step_type}_avg_spiking_gamma', avg_spiking_gamma, on_step=False, on_epoch=True, prog_bar=True)
            # Count non-spiking
            num_nonspiking = (self.gamma < 0.5).sum().item()
            print(f"[DEBUG] num_nonspiking: {num_nonspiking}")
            self.log(f'{step_type}_num_nonspiking_neurons', num_nonspiking, on_step=False, on_epoch=True, prog_bar=True)
            avg_nonspiking_gamma = self.gamma[self.gamma < 0.5].mean() if (self.gamma < 0.5).any() else 0
            print(f"[DEBUG] avg_nonspiking_gamma: {avg_nonspiking_gamma}")
            self.log(f'{step_type}_avg_nonspiking_gamma', avg_nonspiking_gamma, on_step=False, on_epoch=True, prog_bar=True)

            self.log(f'{step_type}_percent_spiking_neurons_spiking_once', percent_spiking_neurons_spiking_once, on_step=False, on_epoch=True, prog_bar=True)

        self.log(f'{step_type}_main_loss', main_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{step_type}_total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def training_step(self, batch, batch_idx): return self._common_step(batch, "train")
    def validation_step(self, batch, batch_idx): self._common_step(batch, "val")
    def test_step(self, batch, batch_idx): self._common_step(batch, "test")
    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.hparams.optimizer_name, None)
        if optimizer_class is None: raise ValueError(f"Optimizer {self.hparams.optimizer_name} not supported.")
        return optimizer_class(self.parameters(), lr=self.hparams.learning_rate)
    
    # Run after training to binarize gamma values - currently not called automatically
    def binarize_gamma(self):
        """
        Set all gamma values to 0 or 1, whichever they are closest to.
        """
        if self.gamma is not None:
            with torch.no_grad():
                self.gamma.data.copy_((self.gamma >= 0.5).float())

    def on_test_end(self):
        """
        Computes final test accuracy and saves it locally.
        It prioritizes a custom `local_artifact_dir` if set on the model,
        otherwise it defaults to the logger's save directory.
        """
        final_test_acc = self.test_accuracy.compute().item()
        print(f"\nFinal Test Accuracy: {final_test_acc:.4f}")

        # Prioritize the custom directory passed from the training script
        save_dir = getattr(self, 'local_artifact_dir', None)

        # Fallback to the logger's directory if no custom path is provided
        if save_dir is None:
            if self.logger and self.logger.save_dir:
                save_dir = self.logger.save_dir

        if save_dir:
            # Ensure the directory exists
            os.makedirs(save_dir, exist_ok=True)
            
            artifact_path = os.path.join(save_dir, "test_results.json")
            results = {"final_test_accuracy": final_test_acc}

            with open(artifact_path, "w") as f:
                json.dump(results, f, indent=4)
            
            print(f"✅ Test artifact saved locally to: {artifact_path}")
        else:
            print("⚠️ No save directory found. Could not save local test artifact.")