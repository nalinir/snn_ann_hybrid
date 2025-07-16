import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

from class_based_implementation.surr_grad import SurrGradSpike
from class_based_implementation.zenke_regularizer import regularization_loss_zenke

valid_models = ["SNN", "ANN_with_LIF_output", "Hybrid_RNN_SNN_rec", "Hybrid_RNN_SNN_V1_same_layer"]

class BaseTemporalModel(pl.LightningModule):
    """
    Base class for SNN, ANN, and Hybrid models to share common functionalities
    like parameter initialization and the final readout layer.
    """
    def __init__(self,
                 input_features: int,
                 hidden_features: int,
                 output_features: int,
                 data_config: dict,
                 model_type: str,
                 spike_fn = None, # SurrGradSpike class or similar
                 recurrent: bool = True,
                 learning_rate: float = 1e-3,
                 snn_mask: torch.Tensor = None,
                 loss_fn = None,
                 zenke_config: dict = None,
                 optimizer_name: str = "Adam",
                 spike_grad_scale: float = 100.0,
                 weight_scale: float = 1,
                 dtype: torch.dtype = torch.float32,
                 max_logit: bool = True,
                 shuffle_neurons: bool = False, # New property
                 new_class_implementation: str = "Fixed units" # Tracking for wandb
                ):
        super().__init__()

        self.save_hyperparameters(ignore=['spike_fn', 'snn_mask', 'loss_fn', 'zenke_config'])

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        self.alpha = float(np.exp(-data_config["time_step"] / data_config["tau_syn"]))
        self.beta=float(np.exp(-data_config["time_step"] / data_config["tau_mem"]))
        self.beta_out=float(np.exp(-data_config["time_step"] / data_config["tau_readout"]))
        self.spike_grad_scale = spike_grad_scale # Stored as a hyperparameter
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.max_logit = max_logit  # If not max, then sum

        self.zenke_config = zenke_config if zenke_config is not None else {}
        self.zenke_enabled = bool(self.zenke_config) and (self.zenke_config.get("l2_lower", 0) != 0 or \
                                                         self.zenke_config.get("l1_upper", 0) != 0 or \
                                                         self.zenke_config.get("l2_upper", 0) != 0)

        if self.loss_fn is None:
            raise ValueError("A loss function instance must be provided to BaseTemporalModel.")

        # Initialize Accuracy metrics for different stages
        self.train_accuracy = Accuracy(task="multiclass", num_classes=output_features)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=output_features)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=output_features)

        w1 = torch.empty((input_features, hidden_features), dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(w1, mean=0.0, std=weight_scale / np.sqrt(input_features))
        self.w1 = nn.Parameter(w1)

        w2 = torch.empty((hidden_features, output_features), dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(w2, mean=0.0, std=weight_scale / np.sqrt(hidden_features))
        self.w2 = nn.Parameter(w2)

        if self.recurrent:
            v1 = torch.empty((hidden_features, hidden_features), dtype=dtype, requires_grad=True)
            torch.nn.init.normal_(v1, mean=0.0, std=weight_scale / np.sqrt(hidden_features))
            self.v1 = nn.Parameter(v1)
        else:
            self.v1 = None

        if model_type not in valid_models:
            raise ValueError(f"{model_type} is an invalid model.")
        self.model_type = model_type

        if snn_mask is None:
            # Reverted to unsqueeze(0) for consistent (1, Hidden) shape for broadcasting
            if model_type == "ANN_with_LIF_output":
                snn_mask = torch.zeros(self.hidden_features)
            elif model_type == "SNN":
                snn_mask = torch.ones(self.hidden_features)

        self.num_snn_neurons_in_layer = torch.sum(snn_mask).item() # Use the original 1D mask for this count

        self.snn_mask = snn_mask.float().unsqueeze(0)
        if self.num_snn_neurons_in_layer > self.hidden_features:
            raise ValueError(f"Dimensions of SNN mask {snn_mask.shape} lead to {self.num_snn_neurons_in_layer}, which is larger than number of hidden features {self.hidden_features}")
        if self.num_snn_neurons_in_layer < 0:
            raise ValueError(f"{self.num_snn_neurons_in_layer} is less than 0")

    def _apply_readout_layer(self, h2_input: torch.Tensor) -> torch.Tensor:
        flt = torch.zeros(
            (h2_input.shape[0], self.output_features), 
            device=h2_input.device, 
            dtype=torch.float32
        )
        out = torch.zeros(
            (h2_input.shape[0], self.output_features), 
            device=h2_input.device, 
            dtype=torch.float32
        )
        out_rec = [out]

        for t in range(h2_input.shape[1]):
            new_flt = self.alpha * flt + h2_input[:, t]
            new_out = self.beta_out * out + (1 - self.beta_out) * flt

            flt = new_flt
            out = new_out

            out_rec.append(out)
        return torch.stack(out_rec, dim=1)

    def _snn_neuron_update_step(self,
                                h1_snn_t: torch.Tensor,
                                mem_snn: torch.Tensor,
                                syn_snn: torch.Tensor,
                                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs one timestep update for SNN neurons.
        Args:
            h1_snn_t: Current input (input + recurrent) to SNN neurons for this timestep.
            mem_snn: Current membrane potential for SNN neurons.
            syn_snn: Current synaptic current for SNN neurons.
        Returns:
            new_mem_snn, new_syn_snn, spikes_snn_t
        """
        mthr = mem_snn - 1.0
        spikes_snn_t = self.spike_fn(mthr, self.spike_grad_scale)
        rst = spikes_snn_t.detach() # Reset signal

        new_syn_snn = self.alpha * syn_snn + h1_snn_t
        new_mem_snn = (self.beta * mem_snn + (1 - self.beta) * syn_snn) * (1.0 - rst)
        return new_mem_snn, new_syn_snn, spikes_snn_t


    def _ann_neuron_update_step(self,
                                h1_ann_t: torch.Tensor,
                                ) -> torch.Tensor:
        """
        Performs one timestep update for ANN neurons.
        Args:
            h1_ann_t: Current input (input + recurrent) to ANN neurons for this timestep.
        Returns:
            ann_activations_t
        """
        # print(f"h1_ann_t output: {h1_ann_t}")
        ann_activations_t = torch.tanh(h1_ann_t)
        # print(f"ANN output: {ann_activations_t}")
        return ann_activations_t

    def _process_hidden_layer_timestep(self,
                                        h1_input_t: torch.Tensor, # Input from w1 (Batch, Hidden)
                                        mem: torch.Tensor, # Full hidden layer mem state (Batch, Hidden)
                                        syn: torch.Tensor, # Full hidden layer syn state (Batch, Hidden)
                                        prev_hidden_out: torch.Tensor, # Combined output from previous step (Batch, Hidden)
                                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes a single timestep for the hidden layer, accounting for SNN/ANN dynamics.
        Returns updated (mem, syn, hidden_out, spike_out, ann_out).
        """
        batch_size = h1_input_t.shape[0]
        device = h1_input_t.device
        hidden_out_combined = torch.zeros((batch_size, self.hidden_features), device=device, dtype=torch.float32)
        spikes_t = torch.zeros((batch_size, self.hidden_features), device=device, dtype=torch.float32)
        ann_activations_t = torch.zeros((batch_size, self.hidden_features), device=device, dtype=torch.float32)

        # Use self.snn_mask directly, as it's already unsqueezed for broadcasting (1, hidden_features)
        snn_mask_effective =  self.snn_mask.to(device) # <--- ADDED .to(device) HERE
        ann_mask_effective = (1.0 - snn_mask_effective)
        
        recurrent_term_snn = 0.0
        recurrent_term_ann = 0.0

        if self.recurrent:
            # Calculate potential recurrent contributions from SNN-sourced and ANN-sourced previous outputs
            recurrent_from_snn_output = torch.einsum("ab,bc->ac", (prev_hidden_out * snn_mask_effective, self.v1))
            recurrent_from_ann_output = torch.einsum("ab,bc->ac", (prev_hidden_out * ann_mask_effective, self.v1))

            if self.model_type == "Hybrid_RNN_SNN_rec": # Using self.model_type
                # SNN output recurs to SNN neurons, ANN output recurs to ANN neurons
                recurrent_term_snn = recurrent_from_snn_output
                recurrent_term_ann = recurrent_from_ann_output
            else: # SNN, ANN, Hybrid_Shared
                # Combined output recurs to all neurons
                recurrent_term_snn = recurrent_from_snn_output + recurrent_from_ann_output
                recurrent_term_ann = recurrent_from_snn_output + recurrent_from_ann_output

        # Process SNN neurons
        if self.model_type in ["SNN", "Hybrid_RNN_SNN_rec", "Hybrid_RNN_SNN_V1_same_layer"]: # Using self.model_type
            # Input to SNN part includes h1_input and its specific recurrent term, then masked
            input_to_snn_neurons = (h1_input_t + recurrent_term_snn) * snn_mask_effective
            current_mem_snn_masked = mem * snn_mask_effective
            current_syn_snn_masked = syn * snn_mask_effective

            new_mem_snn_masked, new_syn_snn_masked, spikes_snn_part = \
                self._snn_neuron_update_step(input_to_snn_neurons, current_mem_snn_masked, current_syn_snn_masked)

            # Apply updates only to SNN masked parts of mem and syn
            mem = new_mem_snn_masked + (mem * ann_mask_effective) # keep ANN parts of mem/syn
            syn = new_syn_snn_masked + (syn * ann_mask_effective) # keep ANN parts of mem/syn
            spikes_t = spikes_snn_part # Store spikes for regularization/auxiliary output

            hidden_out_combined += spikes_snn_part # SNN outputs contribute to combined hidden output

        # Process ANN neurons
        if self.model_type in ["ANN_with_LIF_output", "Hybrid_RNN_SNN_rec", "Hybrid_RNN_SNN_V1_same_layer"]: # Using self.model_type
            # Input to ANN part includes h1_input and its specific recurrent term, then masked
            input_to_ann_neurons = (h1_input_t + recurrent_term_ann) * ann_mask_effective

            ann_activations_part = self._ann_neuron_update_step(input_to_ann_neurons)
            ann_activations_t = ann_activations_part # Store ANN activations for auxiliary output

            hidden_out_combined += ann_activations_part # ANN outputs contribute to combined hidden output

        return mem, syn, hidden_out_combined, spikes_t, ann_activations_t

    def _common_step(self, batch, step_type: str):
        inputs, targets = batch
        predictions, auxiliary_outputs = self(inputs)

        if self.max_logit:
            logit_output, _ = torch.max(predictions, dim=1)
        else:
            logit_output = torch.sum(predictions, dim=1)

        main_loss = self.loss_fn(logit_output, targets)
        total_loss = main_loss

        # Calculate and log Accuracy
        if step_type == "train":
            self.train_accuracy.update(logit_output, targets)
            self.log(f'{step_type}_accuracy', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        elif step_type == "val":
            self.val_accuracy.update(logit_output, targets)
            self.log(f'{step_type}_accuracy', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        elif step_type == "test":
            self.test_accuracy.update(logit_output, targets)
            self.log(f'{step_type}_accuracy', self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # --- Zenke Spike Regularization & Metrics ---
        if 'spikes' in auxiliary_outputs:
            spikes = auxiliary_outputs['spikes'] # Shape: (Batch, Time, Hidden) - binary output

            # Filter spikes for SNN neurons if a mask is present
            if self.snn_mask is not None:
                # Need to convert 1xN mask to 1D boolean for indexing
                snn_spikes = spikes[:, :, self.snn_mask.squeeze(0).bool()]
            else:
                snn_spikes = spikes
        
            spikes_per_neuron_per_sample = torch.sum(snn_spikes, dim=1) # (Batch, N_snn)
            neurons_spiked_at_least_once_mask = (spikes_per_neuron_per_sample > 0).float()
            num_neurons_spiked_per_sample = torch.sum(neurons_spiked_at_least_once_mask, dim=1) #(Batch,)

            percent_snn_neurons_spiking_per_sample = torch.mean(num_neurons_spiked_per_sample/self.num_snn_neurons_in_layer, dim=0) #(N_snn)
            avg_spikes_per_neuron = torch.mean(torch.mean(spikes_per_neuron_per_sample, dim=1), dim=0) # (Batch,) neuron average then total average (mean probably also ok)

            # Zenke and Vogels regularization
            if self.zenke_enabled:
                zenke_loss, lower_l2, upper_l1, upper_l2 = regularization_loss_zenke(snn_spikes, self.zenke_config)
                total_loss += zenke_loss

                self.log(f'{step_type}_zenke_reg_loss', zenke_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                self.log(f'{step_type}_zenke_lower_l2', lower_l2, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                self.log(f'{step_type}_zenke_upper_l1', upper_l1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                self.log(f'{step_type}_zenke_upper_l2', upper_l2, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            total_spikes = snn_spikes.sum()
            self.log(f'{step_type}_percent_neurons_spiking_per_sample', percent_snn_neurons_spiking_per_sample * 100, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{step_type}_total_spikes', total_spikes, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{step_type}_avg_spikes_per_neuron', avg_spikes_per_neuron, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.log(f'{step_type}_main_loss', main_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{step_type}_total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, "test")

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # Need to add momentum logic back in if we do this
        # elif self.hparams.optimizer_name == "SGD":
        #     return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum)
        elif self.hparams.optimizer_name == "Adamax":
            return torch.optim.Adamax(self.parameters(), lr=self.hparams.learning_rate)
        else:
            raise ValueError(f"Optimizer {self.hparams.optimizer_name} not supported.")





