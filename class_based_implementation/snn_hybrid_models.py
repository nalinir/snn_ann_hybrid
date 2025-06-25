import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

# --- Zenke's SurrGradSpike Function (Maren implementation) ---
class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    @staticmethod
    def forward(ctx, input, scale): # scale added as an argument
        ctx.scale = scale
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (ctx.scale * torch.abs(input) + 1.0) ** 2
        return grad, None # Return None for the scale gradient as it's not a learnable parameter here


# heaviside_spike = SurrGradSpike.apply

# --- Regularization Modules (Unchanged) ---
def bound_regularizer(spk, v_t, l_t, l1, upper_bound=True, population_level=True):
    multiplier = 1 if upper_bound else -1
    
    if population_level:
        cnt = torch.mean(spk, dim=(0, 1)) # (N,)
    else:
        cnt = torch.sum(spk, dim=0) # (B, N)

    reg = torch.relu(multiplier * (cnt - v_t))
    return l_t * (torch.mean(torch.abs(reg)) if l1 else torch.mean(torch.square(reg)))


def regularization_loss_zenke(spks, config):
    lower_l2 = bound_regularizer(
        spks,
        config["v2_lower"],
        config["l2_lower"],
        l1=False,
        upper_bound=False,
        population_level=False,
    )
    upper_l1 = bound_regularizer(
        spks,
        config["v1_upper"],
        config["l1_upper"],
        l1=True,
        upper_bound=True,
        population_level=True,
    )
    upper_l2 = bound_regularizer(
        spks,
        config["v2_upper"],
        config["l2_upper"],
        l1=False,
        upper_bound=True,
        population_level=False,
    )
    return lower_l2 + upper_l1 + upper_l2


# --- Base Class for Shared Functionality (Modified for new spiking metric) ---
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
                 spike_fn = None,
                 recurrent: bool = True,
                 learning_rate: float = 1e-3,
                 snn_mask: torch.Tensor = None,
                 loss_fn = None,
                 zenke_config: dict = None,
                 optimizer_name: str = "Adam",
                 spike_grad_scale: float = 100.0, # Added spike_grad_scale
                ):
        super().__init__()
        
        self.save_hyperparameters(ignore=['spike_fn', 'snn_mask', 'loss_fn', 'zenke_config'])

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        self.alpha = float(np.exp(-data_config["time_step"] / data_config["tau_syn"]))
        self.beta=float(np.exp(-data_config["time_step"] / data_config["tau_mem"]))
        self.spike_grad_scale = spike_grad_scale # Stored as a hyperparameter
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn

        self.zenke_config = zenke_config if zenke_config is not None else {}
        self.zenke_enabled = bool(self.zenke_config) and (self.zenke_config.get("l2_lower", 0) > 0 or \
                                                         self.zenke_config.get("l1_upper", 0) > 0 or \
                                                         self.zenke_config.get("l2_upper", 0) > 0)
        
        if self.loss_fn is None:
            raise ValueError("A loss function instance must be provided to BaseTemporalModel.")

        # Initialize Accuracy metrics for different stages
        self.train_accuracy = Accuracy(task="multiclass", num_classes=output_features)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=output_features)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=output_features)

        self.w1 = nn.Parameter(torch.randn(input_features, hidden_features))
        self.w2 = nn.Parameter(torch.randn(hidden_features, output_features))

        if self.recurrent:
            self.v1 = nn.Parameter(torch.randn(hidden_features, hidden_features))
        else:
            self.v1 = None

        if snn_mask is not None:
            # snn_mask indicates which hidden neurons are SNN (1.0) and which are ANN (0.0)
            self.register_buffer('snn_mask', snn_mask.float())
            self.num_snn_neurons_in_layer = torch.sum(self.snn_mask).item()
        else:
            self.snn_mask = None
            # If no mask, assume all hidden neurons are SNN (pure SNN model)
            self.num_snn_neurons_in_layer = self.hidden_features 


    def _apply_readout_layer(self, h2_input: torch.Tensor) -> torch.Tensor:
        flt = torch.zeros(
            (h2_input.shape[0], self.output_features), device=h2_input.device, dtype=torch.float32
        )
        out = torch.zeros(
            (h2_input.shape[0], self.output_features), device=h2_input.device, dtype=torch.float32
        )
        out_rec = [out]

        for t in range(h2_input.shape[1]):
            new_flt = self.alpha * flt + h2_input[:, t]
            new_out = self.beta * out + flt

            flt = new_flt
            out = new_out

            out_rec.append(out)
        return torch.stack(out_rec, dim=1)

    def permute_batch_dim(self, inputs):
        permute_order = list(np.arange(len(inputs.size())))
        permute_order[:2] = [1, 0]

        inputs = inputs.permute(permute_order)

        return inputs

    # To deal with the SHD case where input > 3 dimensions
    def _prep_forward(self, inputs):
        if len(inputs.size()) > 3:
            print("Input shape before reshape:", inputs.size())
            inputs = self.permute_batch_dim(
                inputs
            )  # since time dim was padded before -> now set batch first
            sz = inputs.size()
            print("Input shape after permutation:", inputs.size())
            a, nb_steps = sz[0], sz[1]
            inputs = inputs.reshape((a, nb_steps, -1))
            print("Input shape after reshape:", inputs.size())
        return inputs

    def _common_step(self, batch, step_type: str):
        inputs, targets = batch
        # inputs = self._prep_forward(inputs)  # Ensure inputs are in the correct shape
        predictions, auxiliary_outputs = self(inputs)

        max_over_time, _ = torch.max(predictions, dim=1)  # [batch, num_classes]

        main_loss = self.loss_fn(max_over_time, targets)
        total_loss = main_loss

        # Calculate and log Accuracy
        if step_type == "train":
            self.train_accuracy.update(max_over_time, targets)
            self.log(f'{step_type}_accuracy', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        elif step_type == "val":
            self.val_accuracy.update(max_over_time, targets)
            self.log(f'{step_type}_accuracy', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        elif step_type == "test":
            self.test_accuracy.update(max_over_time, targets)
            self.log(f'{step_type}_accuracy', self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # --- Zenke Spike Regularization & Metrics ---
        if self.zenke_enabled and 'spikes' in auxiliary_outputs:
            spikes = auxiliary_outputs['spikes'] # Shape: (Batch, Time, Hidden) - binary output

            zenke_loss = regularization_loss_zenke(spikes, self.zenke_config)
            total_loss += zenke_loss

            self.log(f'{step_type}_main_loss', main_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{step_type}_zenke_reg_loss', zenke_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            
            spikes_per_neuron_per_sample = torch.sum(spikes, dim=1) 
            
            # Mask to indicate which neurons spiked at least once in each sample: (Batch, Hidden)
            neurons_spiked_at_least_once_mask = (spikes_per_neuron_per_sample > 0).float()

            if self.snn_mask is not None:
                # For hybrid models, apply the mask to only consider SNN neurons
                # snn_mask is (Hidden,), unsqueeze(0) makes it (1, Hidden) for broadcasting
                snn_neurons_spiked_flagged = neurons_spiked_at_least_once_mask * self.snn_mask.unsqueeze(0) # (Batch, Hidden)
                
                # Count how many SNN neurons spiked at least once per sample: (Batch,)
                num_snn_neurons_spiked_per_sample = torch.sum(snn_neurons_spiked_flagged, dim=1)
                
                # Calculate percentage, averaging over the batch: scalar
                if self.num_snn_neurons_in_layer > 0:
                    percent_snn_neurons_spiking_per_sample = (num_snn_neurons_spiked_per_sample / self.num_snn_neurons_in_layer).mean()

                else: # Handle case with no SNN neurons (shouldn't happen if spk_rec is provided)
                    percent_snn_neurons_spiking_per_sample = torch.tensor(0.0, device=self.device)
            else: 
                # For pure SNN models, all hidden neurons are SNN
                # Count how many hidden neurons spiked at least once per sample: (Batch,)
                num_neurons_spiked_per_sample = torch.sum(neurons_spiked_at_least_once_mask, dim=1)
                # Calculate percentage, averaging over the batch: scalar
                percent_snn_neurons_spiking_per_sample = (num_neurons_spiked_per_sample / self.hidden_features).mean()

            total_spikes = spikes.sum()
            self.log(f'{step_type}_percent_neurons_spiking_per_sample', percent_snn_neurons_spiking_per_sample * 100, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{step_type}_total_spikes', total_spikes, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        else: # For models without spikes (e.g., pure ANN or if zenke_enabled is False)
            self.log(f'{step_type}_loss', main_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

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
        elif self.hparams.optimizer_name == "SGD":
            return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum)

# --- SNN Model Implementation (Unchanged, inherits accuracy/spiking logging) ---
class SNN(BaseTemporalModel):
    """
    SNN model inheriting from BaseTemporalModel.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.recurrent:
            print("Warning: SNN model typically uses recurrent connections.")
        if self.spike_fn is None:
            raise ValueError("spike_fn must be provided for SNN.")

    def forward(self, inputs: torch.Tensor):
        inputs = inputs.to(self.device)

        syn = torch.zeros((inputs.shape[0], self.hidden_features), dtype=torch.float32, device=self.device)
        mem = torch.zeros((inputs.shape[0], self.hidden_features), dtype=torch.float32, device=self.device)
        out = torch.zeros((inputs.shape[0], self.hidden_features), dtype=torch.float32, device=self.device)

        mem_rec = []
        spk_rec = []

        h1_from_input = torch.einsum("abc,cd->abd", (inputs, self.w1))

        for t in range(inputs.shape[1]):
            h1 = h1_from_input[:, t]
            if self.recurrent:
                h1 += torch.einsum("ab,bc->ac", (out, self.v1))

            mthr = mem - 1.0
            out = self.spike_fn(mthr, self.spike_grad_scale)
            rst = out.detach()

            new_syn = self.alpha * syn + h1
            new_mem = (self.beta * mem + syn) * (1.0 - rst)

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn

        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)

        h2_input = torch.einsum("abc,cd->abd", (spk_rec, self.w2))
        out_rec = self._apply_readout_layer(h2_input)

        auxiliary_outputs = {
            'mem_rec': mem_rec,
            'spikes': spk_rec
        }
        return out_rec, auxiliary_outputs

# --- ANN with LIF Output Model Implementation (Unchanged, inherits logging) ---
class ANN_with_LIF_output(BaseTemporalModel):
    """
    ANN model with a LIF-like output layer, inheriting from BaseTemporalModel.
    Spike regularization is not typically applied to ANN hidden units.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.spike_fn is not None:
             print("Note: spike_fn is generally not used for ANN hidden layers in this model. Only for LIF-like output.")

    def forward(self, inputs: torch.Tensor):
        inputs = inputs.to(self.device)

        out = torch.zeros((inputs.shape[0], self.hidden_features), dtype=torch.float32, device=self.device)

        ann_rec = []

        h1_from_input = torch.einsum("abc,cd->abd", (inputs, self.w1))

        for t in range(inputs.shape[1]):
            layer_weights = h1_from_input[:, t]
            if self.recurrent:
                layer_weights += torch.einsum("ab,bc->ac", (out, self.v1))

            out = torch.tanh(layer_weights)
            ann_rec.append(out)

        ann_rec = torch.stack(ann_rec, dim=1)

        h2_input = torch.einsum("abc,cd->abd", (ann_rec, self.w2))
        out_rec = self._apply_readout_layer(h2_input)

        auxiliary_outputs = {
            'ann_hidden_activations': ann_rec,
        }
        return out_rec, auxiliary_outputs

# --- Hybrid RNN-SNN (Separate Recurrent Connections) Implementation (Unchanged, inherits logging) ---
class Hybrid_RNN_SNN_rec(BaseTemporalModel):
    """
    Hybrid RNN-SNN model with separate recurrent connections for ANN and SNN parts.
    Inherits from BaseTemporalModel.
    """
    def __init__(self,
                 input_features: int,
                 hidden_features: int,
                 output_features: int,
                 data_config: dict,
                #  alpha: float,
                #  beta: float,
                 spike_fn = None,
                 recurrent: bool = True,
                 learning_rate: float = 1e-3,
                 loss_fn = None,
                 zenke_config: dict = None,
                 optimizer_name: str = "Adam",
                 spike_grad_scale: float = 100.0, # <--- ADD THIS LINE                 
                ):
        
        if hidden_features % 2 != 0:
            raise ValueError("hidden_features must be an even number for an exact 50/50 SNN/ANN split.")
        num_snn_neurons = hidden_features // 2
        snn_mask_elements = torch.cat([
            torch.ones(num_snn_neurons),
            torch.zeros(hidden_features - num_snn_neurons)
        ])
        snn_mask_tensor = snn_mask_elements[torch.randperm(hidden_features)]

        super().__init__(
            input_features=input_features,
            hidden_features=hidden_features,
            output_features=output_features,
            # alpha=alpha,
            # beta=beta,
            data_config=data_config,
            spike_fn=spike_fn,
            recurrent=recurrent,
            learning_rate=learning_rate,
            snn_mask=snn_mask_tensor,
            loss_fn=loss_fn,
            zenke_config=zenke_config,
            optimizer_name=optimizer_name,
            spike_grad_scale=spike_grad_scale
        )

        if self.snn_mask is None:
            raise ValueError("snn_mask must be provided (or generated) for Hybrid_RNN_SNN_rec.")
        if self.spike_fn is None:
            raise ValueError("spike_fn must be provided for Hybrid_RNN_SNN_rec.")
        if not self.recurrent:
             print("Warning: Hybrid_RNN_SNN_rec typically uses recurrent connections.")

    def forward(self, inputs: torch.Tensor):
        inputs = inputs.to(self.device)

        h1_full = torch.einsum("abc,cd->abd", (inputs, self.w1))

        h1_ann_input = h1_full * (1.0 - self.snn_mask)
        h1_snn_input = h1_full * self.snn_mask

        syn = torch.zeros((inputs.shape[0], self.hidden_features), device=self.device, dtype=torch.float32)
        mem = torch.zeros((inputs.shape[0], self.hidden_features), device=self.device, dtype=torch.float32)

        mem_rec = []
        spk_rec = []
        ann_rec = []

        out_ann = torch.zeros((inputs.shape[0], self.hidden_features), device=self.device, dtype=torch.float32)
        out_snn = torch.zeros((inputs.shape[0], self.hidden_features), device=self.device, dtype=torch.float32)

        for t in range(inputs.shape[1]):
            # SNN part
            h1_snn_t = h1_snn_input[:, t]
            if self.recurrent:
                h1_snn_t += torch.einsum("ab,bc->ac", (out_snn, self.v1)) * self.snn_mask

            current_mem_snn_masked = mem * self.snn_mask
            current_syn_snn_masked = syn * self.snn_mask

            mthr = current_mem_snn_masked - 1.0
            out_snn_t_raw = self.spike_fn(mthr, self.spike_grad_scale)
            out_snn_t = out_snn_t_raw * self.snn_mask
            rst = out_snn_t.detach()

            new_syn_snn = self.alpha * current_syn_snn_masked + h1_snn_t
            new_mem_snn = (self.beta * current_mem_snn_masked + current_syn_snn_masked) * (1.0 - rst)

            # ANN part
            h1_ann_t = h1_ann_input[:, t]
            if self.recurrent:
                h1_ann_t += torch.einsum("ab,bc->ac", (out_ann, self.v1)) * (1.0 - self.snn_mask)
            out_ann_t = torch.tanh(h1_ann_t) * (1.0 - self.snn_mask)

            mem = new_mem_snn + (mem * (1.0 - self.snn_mask))
            syn = new_syn_snn + (syn * (1.0 - self.snn_mask))
            
            out_ann = out_ann_t
            out_snn = out_snn_t

            mem_rec.append(mem)
            spk_rec.append(out_snn_t)
            ann_rec.append(out_ann_t)

        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)
        ann_rec = torch.stack(ann_rec, dim=1)

        h2_snn = torch.einsum("abc,cd->abd", (spk_rec, self.w2))
        h2_ann = torch.einsum("abc,cd->abd", (ann_rec, self.w2))
        h2_input = h2_snn + h2_ann

        out_rec = self._apply_readout_layer(h2_input)

        auxiliary_outputs = {
            'mem_rec': mem_rec,
            'spikes': spk_rec,
            'ann_hidden_activations': ann_rec
        }
        return out_rec, auxiliary_outputs

# --- Hybrid RNN-SNN (Shared Recurrent Connection) Implementation (Unchanged, inherits logging) ---
class Hybrid_RNN_SNN_V1_same_layer(BaseTemporalModel):
    """
    Hybrid RNN-SNN model with shared recurrent connections in the hidden layer.
    Inherits from BaseTemporalModel.
    """
    def __init__(self,
                 input_features: int,
                 hidden_features: int,
                 output_features: int,
                #  alpha: float,
                #  beta: float,
                 data_config: dict,
                 spike_fn = None,
                 recurrent: bool = True,
                 learning_rate: float = 1e-3,
                 loss_fn = None,
                 zenke_config: dict = None,
                 optimizer_name: str = "Adam",
                 spike_grad_scale: float = 100.0, # <--- ADD THIS LINE
                ):

        if hidden_features % 2 != 0:
            raise ValueError("hidden_features must be an even number for an exact 50/50 SNN/ANN split.")
        num_snn_neurons = hidden_features // 2
        snn_mask_elements = torch.cat([
            torch.ones(num_snn_neurons),
            torch.zeros(hidden_features - num_snn_neurons)
        ])
        snn_mask_tensor = snn_mask_elements[torch.randperm(hidden_features)]

        super().__init__(
            input_features=input_features,
            hidden_features=hidden_features,
            output_features=output_features,
            # alpha=alpha,
            # beta=beta,
            data_config=data_config,
            spike_fn=spike_fn,
            recurrent=recurrent,
            learning_rate=learning_rate,
            snn_mask=snn_mask_tensor,
            loss_fn=loss_fn,
            zenke_config=zenke_config,
            optimizer_name=optimizer_name,
            spike_grad_scale=spike_grad_scale
        )

        if self.snn_mask is None:
            raise ValueError("snn_mask must be provided (or generated) for Hybrid_RNN_SNN_V1_same_layer.")
        if self.spike_fn is None:
            raise ValueError("spike_fn must be provided for Hybrid_RNN_SNN_V1_same_layer.")
        if not self.recurrent:
            raise ValueError("Must be recurrent for Hybrid_RNN_SNN_V1_same_layer.")

    def forward(self, inputs: torch.Tensor):
        inputs = inputs.to(self.device)

        h1_full = torch.einsum("abc,cd->abd", (inputs, self.w1))

        h1_ann_input = h1_full * (1.0 - self.snn_mask)
        h1_snn_input = h1_full * self.snn_mask

        syn = torch.zeros((inputs.shape[0], self.hidden_features), device=self.device, dtype=torch.float32)
        mem = torch.zeros((inputs.shape[0], self.hidden_features), device=self.device, dtype=torch.float32)

        mem_rec = []
        spk_rec = []
        ann_rec = []

        out_ann = torch.zeros((inputs.shape[0], self.hidden_features), device=self.device, dtype=torch.float32)
        out_snn = torch.zeros((inputs.shape[0], self.hidden_features), device=self.device, dtype=torch.float32)

        for t in range(inputs.shape[1]):
            out_combined_prev = (out_snn + out_ann)
            # recurrent_term = 0.0
            # if self.recurrent:
            recurrent_term = torch.einsum("ab,bc->ac", (out_combined_prev, self.v1))

            # SNN part
            h1_snn_t = h1_snn_input[:, t] + recurrent_term * self.snn_mask
            
            current_mem_snn_masked = mem * self.snn_mask
            current_syn_snn_masked = syn * self.snn_mask

            mthr = current_mem_snn_masked - 1.0
            out_snn_t_raw = self.spike_fn(mthr, self.spike_grad_scale)
            out_snn_t = out_snn_t_raw * self.snn_mask
            rst = out_snn_t.detach()

            new_syn_snn = self.alpha * current_syn_snn_masked + h1_snn_t
            new_mem_snn = (self.beta * current_mem_snn_masked + current_syn_snn_masked) * (1.0 - rst)

            # ANN part
            h1_ann_t = h1_ann_input[:, t] + recurrent_term * (1.0 - self.snn_mask)
            out_ann_t = torch.tanh(h1_ann_t) * (1.0 - self.snn_mask)

            mem = new_mem_snn + (mem * (1.0 - self.snn_mask))
            syn = new_syn_snn + (syn * (1.0 - self.snn_mask))

            out_ann = out_ann_t
            out_snn = out_snn_t

            mem_rec.append(mem)
            spk_rec.append(out_snn_t)
            ann_rec.append(out_ann_t)

        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)
        ann_rec = torch.stack(ann_rec, dim=1)

        h2_snn = torch.einsum("abc,cd->abd", (spk_rec, self.w2))
        h2_ann = torch.einsum("abc,cd->abd", (ann_rec, self.w2))
        h2_input = h2_snn + h2_ann

        out_rec = self._apply_readout_layer(h2_input)

        auxiliary_outputs = {
            'mem_rec': mem_rec,
            'spikes': spk_rec,
            'ann_hidden_activations': ann_rec
        }
        return out_rec, auxiliary_outputs