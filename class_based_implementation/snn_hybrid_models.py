import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy
from class_based_implementation.Regularizers import UpperBoundL1, UpperBoundL2, LowerBoundL2

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


# # --- Attention Mechanism Loss ---
# def parameter_free_attention(mem, n):
#     # Input current should be the membrane potential for a given neuron
#     # Threshold is 1 for this model (just don't do mthr)
#     # Other should be 0


#     # The membrane potential for each timestep has shape (batch, nb_hidden)
#     first_part = (1-mem)**2
#     # This is just the resulting membrane potential from all other neurons other than a given neuron
#     second_part_interim = (0-mem)**2
#     total_sum_mem = torch.sum(second_part_interim)
#     second_part = (total_sum_mem - second_part_interim)/(n-1)
#     # We take all except a given index and then take the mean
#     return second_part + first_part

# def attention_loss(mem, w1, n, config):
#     """
#     Computes the attention loss with L2 regularization.
#     """
#     # mem should be of shape (batch, nb_hidden)
#     # n is the number of neurons in the layer
#     ## To allow for L2 regularization for model
#     if mem is None:
#         attn_loss = 0
#     else:
#         attn_loss = torch.sum(parameter_free_attention(mem, n))
#     l2_loss = torch.sum(config["l2"] * (w1**2))
#     return attn_loss + l2_loss


# # --- Regularization Modules (Unchanged) ---
# NOW -> ONLY SNN SPIKES SHOULD BE INPUT! NO OTHER EXTRA DIMENSION FOR ANN!
def bound_regularizer(spk, v_t, l_t, exp, upper_bound=True, population_level=True):
    # B, T, N = spk.shape
    cnt = torch.sum(spk, dim=1)  # get spikecount over time (B, N) -> do the batch level averaging last!
    if upper_bound:
        diff = cnt-v_t # (N,)
        mean_diff = torch.mean(diff, dim=1) # (B,)
        relu_result = torch.relu(mean_diff)**exp #(B,)
        # Then take the average
        return l_t * torch.mean(relu_result) # I think the -1 was just a typo/artifact????
    else: # lower bound
        r_diff = torch.relu(v_t - cnt) # (B, N)
        exp_r_diff = torch.pow(r_diff, exp) # (B, N)
        mean_diff = torch.mean(exp_r_diff, dim=1) # (B,)
        return l_t * torch.mean(mean_diff)

def regularization_loss_zenke(spks, config):
    lower_l2 = bound_regularizer(
        spks,
        config["v2_lower"],
        config["l2_lower"],
        2,
        upper_bound=False,
        population_level=False,
    )
    
    upper_l1 = bound_regularizer(
        spks,
        config["v1_upper"],
        config["l1_upper"],
        1,
        upper_bound=True,
        population_level=True,
    )
    upper_l2 = bound_regularizer(
        spks,
        config["v2_upper"],
        config["l2_upper"],
        2,
        upper_bound=True,
        population_level=False,
    )
    return lower_l2 + upper_l1 + upper_l2, lower_l2, upper_l1, upper_l2

## GEMINI:
# --- Reintroduced regularization_loss_zenke function using class-based regularizers ---
# def regularization_loss_zenke(spks, config):
#     """
#     Calculates Zenke-style regularization loss using class-based regularizers.

#     Args:
#         spks (torch.Tensor): The spike tensor from the hidden layer (Batch, Time, Hidden).
#         config (dict): A dictionary containing regularization parameters like:
#                        "v2_lower", "l2_lower" for lower L2 bound
#                        "v1_upper", "l1_upper" for upper L1 bound
#                        "v2_upper", "l2_upper" for upper L2 bound
#     Returns:
#         torch.Tensor: The total regularization loss.
#     """
#     total_reg_loss = torch.tensor(0.0, device=spks.device)

#     # Initialize and apply LowerBoundL2
#     # In original `bound_regularizer`, `population_level=False` meant `cnt = torch.sum(spk, dim=0)`
#     # which results in (Time, Units) if spk is (B,T,N). This was then averaged.
#     # To mimic this with ActivityRegularizer, if `spks` is (B,T,N), then `sum(dim=1)` makes it (B,N).
#     # Setting `dims=False` in ActivityRegularizer means no further averaging is done, so it's (B,N).
#     # Then `calc_regloss` applies `torch.mean` over the entire (B,N) tensor.
#     # This is equivalent to taking mean over (0,1) for the (B,N) cnt.
#     # So `dims=False` is the correct equivalent here for "per-neuron" in your original sense.
#     # THIS IS THE PROBLEM! g
#     if config.get("l2_lower", 0) != 0:
#         lower_l2_reg = LowerBoundL2(
#             strength=config["l2_lower"],
#             threshold=config["v2_lower"],
#             dims=False # Corresponds to your original `population_level=False`
#         )
#         total_reg_loss += lower_l2_reg(spks)

#     # Initialize and apply UpperBoundL1
#     # In original `bound_regularizer`, `population_level=True` meant `cnt = torch.mean(spk, dim=(0, 1))`
#     # which results in (N,). With ActivityRegularizer, `sum(dim=1)` yields (B,N).
#     # Then `dims=-1` averages over N, resulting in (B,).
#     # The final `torch.mean` in `calc_regloss` averages over B. This is the closest match.
#     if config.get("l1_upper", 0) != 0:
#         upper_l1_reg = UpperBoundL1(
#             strength=config["l1_upper"],
#             threshold=config["v1_upper"],
#             dims=-1 # Corresponds to your original `population_level=True`
#         )
#         total_reg_loss += upper_l1_reg(spks)

#     # Initialize and apply UpperBoundL2
#     if config.get("l2_upper", 0) != 0:
#         upper_l2_reg = UpperBoundL2(
#             strength=config["l2_upper"],
#             threshold=config["v2_upper"],
#             dims=False # Corresponds to your original `population_level=False`
#         )
#         total_reg_loss += upper_l2_reg(spks)
        
#     return total_reg_loss

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
                 weight_scale: float = 1,  # <-- Add this line
                 dtype: torch.dtype = torch.float32,  # <-- Add this line
                 max_logit: bool = True,  # If not max, then sum
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

        if snn_mask is not None:
            # snn_mask indicates which hidden neurons are SNN (1.0) and which are ANN (0.0)
            self.register_buffer('snn_mask', snn_mask.float())
            self.num_snn_neurons_in_layer = torch.sum(self.snn_mask).item()
        else:
            self.snn_mask = None
            # If no mask, assume all hidden neurons are SNN (pure SNN model)
            self.num_snn_neurons_in_layer = self.hidden_features 
        self.snn_ratio =  self.num_snn_neurons_in_layer/self.hidden_features
        if self.snn_ratio == 1 and snn_mask is not None:
            raise ValueError("snn ratio is 1 but SNN mask is not None, check SNN mask")


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
            new_out = self.beta_out * out + flt

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

        # This should maybe not be max over time but sum?
        if self.max_logit:
            logit_output, _ = torch.max(predictions, dim=1)  # [batch, num_classes]
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
        if self.zenke_enabled and 'spikes' in auxiliary_outputs:
            spikes = auxiliary_outputs['spikes'] # Shape: (Batch, Time, Hidden) - binary output
            # MAYBE WE JUST ADJUST THE DIMENSIONS OF SPIKES TO ACCOUNT FOR THE NUMBER OF NEURONS?


            if self.snn_mask is not None:
                # For hybrid models, apply the mask to only consider SNN neurons
                # snn_mask is (Hidden,), unsqueeze(0) makes it (1, Hidden) for broadcasting
                # THIS IS ACTUALY REDUNDANT - LET"S FILTER IT OUT
                # snn_neurons_spiked_flagged = neurons_spiked_at_least_once_mask * self.snn_mask.unsqueeze(0) # (Batch, Hidden)
                snn_spikes = spikes[:, :, self.snn_mask.bool()]
                
                # Count how many SNN neurons spiked at least once per sample: (Batch,)
                
                # # Calculate percentage, averaging over the batch: scalar
                # if self.num_snn_neurons_in_layer > 0:
                #     percent_snn_neurons_spiking_per_sample = (num_snn_neurons_spiked_per_sample / self.num_snn_neurons_in_layer).mean()
                #     # We have spikes per neuron per sample (batch, hidden)
                #     # Next -> we need to figure out the average number of spikes per neuron in a given batch
                #     # We can do this by just taking the average (BUT it should be scaled by the number of neurons in layer so it's not penalized)
                #     avg_spikes_per_neuron_batch_level = torch.mean(spikes_per_neuron_per_sample, dim=1)*self.snn_ratio # (batch,) RESCALED
                #     avg_spikes_per_neuron = torch.mean(avg_spikes_per_neuron_batch_level, dim=0)
                # else: # Handle case with no SNN neurons (shouldn't happen if spk_rec is provided)
                #     raise ValueError("no SNNs but spk_rec provided")

            else: 
                snn_spikes = spikes
                # For pure SNN models, all hidden neurons are SNN
                # Count how many hidden neurons spiked at least once per sample: (Batch,)
                # num_neurons_spiked_per_sample = torch.sum(neurons_spiked_at_least_once_mask, dim=1)
                # # Calculate percentage, averaging over the batch: scalar
                # percent_snn_neurons_spiking_per_sample = (num_neurons_spiked_per_sample / self.hidden_features).mean()
                # avg_spikes_per_neuron = torch.mean(torch.mean(spikes_per_neuron_per_sample, dim=1), dim=0)  # mean over hidden, then batch

            spikes_per_neuron_per_sample = torch.sum(snn_spikes, dim=1) # Total count of number of spiking neurons per neuron: shape (batch, hidden)
            # Mask to indicate which neurons spiked at least once in each sample: (Batch, Hidden)
            neurons_spiked_at_least_once_mask = (spikes_per_neuron_per_sample > 0).float()
            num_neurons_spiked_per_sample = torch.sum(neurons_spiked_at_least_once_mask, dim=1)
            percent_snn_neurons_spiking_per_sample = (num_neurons_spiked_per_sample / self.hidden_features).mean()
            avg_spikes_per_neuron = torch.mean(torch.mean(spikes_per_neuron_per_sample, dim=1), dim=0)  # mean over hidden, then batch

            # Put in the average spikes per neuron at the batch level or make sure that logic is implemented in zenke loss
            zenke_loss, lower_l2, upper_l1, upper_l2 = regularization_loss_zenke(snn_spikes, self.zenke_config)
            total_loss += zenke_loss
            
            self.log(f'{step_type}_zenke_reg_loss', zenke_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log(f'{step_type}_zenke_lower_l2', lower_l2, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log(f'{step_type}_zenke_upper_l1', upper_l1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log(f'{step_type}_zenke_upper_l2', upper_l2, on_step=False, on_epoch=True, prog_bar=False, logger=True)            

            total_spikes = spikes.sum()
            self.log(f'{step_type}_percent_neurons_spiking_per_sample', percent_snn_neurons_spiking_per_sample * 100, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{step_type}_total_spikes', total_spikes, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            # --- NEW: Average spikes per neuron and per batch ---
            # spikes: (Batch, Time, Hidden)

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
        elif self.hparams.optimizer_name == "SGD":
            return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum)
        elif self.hparams.optimizer_name == "Adamax":
            return torch.optim.Adamax(self.parameters(), lr=self.hparams.learning_rate)

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
        # inputs = inputs.to(self.device)

        syn = torch.zeros((inputs.shape[0], self.hidden_features), 
                        dtype=torch.float32, 
                        device=self.device
                        )
        mem = torch.zeros((inputs.shape[0], self.hidden_features), 
                        dtype=torch.float32, 
                        device=self.device
                        )
        out = torch.zeros((inputs.shape[0], self.hidden_features), 
                            dtype=torch.float32, 
                            device=self.device
                            )

        mem_rec = []
        spk_rec = []

        h1_from_input = torch.einsum("abc,cd->abd", (inputs, self.w1))

        for t in range(inputs.shape[1]):
            h1 = h1_from_input[:, t]
            if self.recurrent:
                h1 += torch.einsum("ab,bc->ac", (out, self.v1))

            mthr = mem - 1.0
            out = SurrGradSpike.apply(mthr, self.spike_grad_scale)
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
        # inputs = inputs.to(self.device)

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
        # snn_mask_tensor = snn_mask_elements[torch.randperm(hidden_features)]

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
            snn_mask=snn_mask_elements,
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
        # inputs = inputs.to(self.device)

        h1_full = torch.einsum("abc,cd->abd", (inputs, self.w1))

        h1_ann_input = h1_full * (1.0 - self.snn_mask)
        h1_snn_input = h1_full * self.snn_mask

        syn = torch.zeros((inputs.shape[0], self.hidden_features),
                            device=self.device, 
                            dtype=torch.float32)
        mem = torch.zeros((inputs.shape[0], self.hidden_features), 
                            device=self.device, 
                            dtype=torch.float32)

        mem_rec = []
        spk_rec = []
        ann_rec = []

        out_ann = torch.zeros((inputs.shape[0], self.hidden_features), 
                                device=self.device, 
                                dtype=torch.float32)
        out_snn = torch.zeros((inputs.shape[0], self.hidden_features), 
                                device=self.device, 
                                dtype=torch.float32)

        for t in range(inputs.shape[1]):
            # SNN part
            h1_snn_t = h1_snn_input[:, t]
            if self.recurrent:
                h1_snn_t += torch.einsum("ab,bc->ac", (out_snn, self.v1)) * self.snn_mask

            current_mem_snn_masked = mem * self.snn_mask
            current_syn_snn_masked = syn * self.snn_mask

            mthr = current_mem_snn_masked - 1.0
            out_snn_t_raw = SurrGradSpike.apply(mthr, self.spike_grad_scale)
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
        # snn_mask_tensor = snn_mask_elements[torch.randperm(hidden_features)]

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
            snn_mask=snn_mask_elements,
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
        # inputs = inputs.to(self.device)

        h1_full = torch.einsum("abc,cd->abd", (inputs, self.w1))

        h1_ann_input = h1_full * (1.0 - self.snn_mask)
        h1_snn_input = h1_full * self.snn_mask

        syn = torch.zeros((inputs.shape[0], self.hidden_features), 
                            device=self.device, 
                            dtype=torch.float32)
        mem = torch.zeros((inputs.shape[0], self.hidden_features), 
                            device=self.device, 
                            dtype=torch.float32)

        mem_rec = []
        spk_rec = []
        ann_rec = []

        out_ann = torch.zeros((inputs.shape[0], self.hidden_features), 
                                device=self.device, 
                                dtype=torch.float32)
        out_snn = torch.zeros((inputs.shape[0], self.hidden_features), 
                                device=self.device, 
                                dtype=torch.float32)

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
            out_snn_t_raw = SurrGradSpike.apply(mthr, self.spike_grad_scale)
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