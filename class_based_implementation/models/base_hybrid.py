import torch
from class_based_implementation.models.base_temporal_model import BaseTemporalModel

class HybridBaseModel(BaseTemporalModel): # Inherit directly from BaseTemporalModel
    def __init__(self,
                 input_features: int,
                 hidden_features: int,
                 output_features: int,
                 data_config: dict,
                 model_type: str, # model_type will be passed by concrete hybrid classes
                 spike_fn = None, # SurrGradSpike class or similar
                 recurrent: bool = True,
                 learning_rate: float = 1e-3,
                 loss_fn = None,
                 zenke_config: dict = None,
                 optimizer_name: str = "Adam",
                 spike_grad_scale: float = 100.0,
                 shuffle_neurons: bool = True,
                ):

        # Common __init__ logic for hybrid models
        if hidden_features % 2 != 0:
            raise ValueError("hidden_features must be an even number for an exact 50/50 SNN/ANN split.")
        num_snn_neurons = hidden_features // 2
        snn_mask_elements = torch.cat([
            torch.ones(num_snn_neurons),
            torch.zeros(hidden_features - num_snn_neurons)
        ])

        if shuffle_neurons:
            snn_mask_elements = snn_mask_elements[torch.randperm(hidden_features)]

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
            snn_mask=snn_mask_elements, # Pass the generated mask
            loss_fn=loss_fn,
            zenke_config=zenke_config,
            optimizer_name=optimizer_name,
            spike_grad_scale=spike_grad_scale,
            shuffle_neurons=shuffle_neurons
        )
        self.save_hyperparameters() # Save hyperparameters after super().__init__

        # Common validation checks for hybrid models
        if self.snn_mask is None:
            raise ValueError("snn_mask must be provided (or generated) for HybridBaseModel subclasses.")
        if self.spike_fn is None:
            raise ValueError("spike_fn must be provided for HybridBaseModel subclasses.")
        # if not isinstance(self.spike_fn, type) or not issubclass(self.spike_fn, torch.autograd.Function):
        #      raise ValueError("spike_fn must be a torch.autograd.Function class (e.g., SurrGradSpike).")

    # The entire forward method, as it's identical for both hybrid models
    def forward(self, inputs: torch.Tensor):
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        h1_full = torch.einsum("abc,cd->abd", (inputs, self.w1))

        syn = torch.zeros((batch_size, self.hidden_features),
                            device=self.device,
                            dtype=torch.float32)
        mem = torch.zeros((batch_size, self.hidden_features),
                            device=self.device,
                            dtype=torch.float32)
        prev_hidden_out = torch.zeros((batch_size, self.hidden_features),
                                        dtype=torch.float32,
                                        device=self.device)

        mem_rec = []
        spk_rec = []
        ann_rec = []

        for t in range(seq_len):
            current_h1_input = h1_full[:, t]

            mem, syn, hidden_out_t, spikes_t, ann_activations_t = \
                self._process_hidden_layer_timestep(
                    h1_input_t=current_h1_input,
                    mem=mem,
                    syn=syn,
                    prev_hidden_out=prev_hidden_out,
                )

            mem_rec.append(mem)
            spk_rec.append(spikes_t)
            ann_rec.append(ann_activations_t)

            prev_hidden_out = hidden_out_t

        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)
        ann_rec = torch.stack(ann_rec, dim=1)

        h2_input = torch.einsum("abc,cd->abd", (spk_rec + ann_rec, self.w2))
        out_rec = self._apply_readout_layer(h2_input)

        auxiliary_outputs = {
            'mem_rec': mem_rec,
            'spikes': spk_rec,
            'ann_hidden_activations': ann_rec
        }
        return out_rec, auxiliary_outputs

