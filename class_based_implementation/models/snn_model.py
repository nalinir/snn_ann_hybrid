import torch
from class_based_implementation.models.base_temporal_model import BaseTemporalModel

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
        # Ensure spike_fn is the SurrGradSpike class itself, not an instance
        # if not isinstance(self.spike_fn, type) or not issubclass(self.spike_fn, torch.autograd.Function):
        #      raise ValueError("spike_fn must be a torch.autograd.Function class (e.g., SurrGradSpike).")
        if self.snn_mask is not None and not torch.all(self.snn_mask == 1.0):
            raise ValueError("SNN model should have all SNN neurons (snn_mask should be None or all ones).")


    def forward(self, inputs: torch.Tensor):
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        syn = torch.zeros((batch_size, self.hidden_features),
                        dtype=torch.float32,
                        device=self.device
                        )
        mem = torch.zeros((batch_size, self.hidden_features),
                        dtype=torch.float32,
                        device=self.device
                        )
        prev_hidden_out = torch.zeros((batch_size, self.hidden_features),
                                        dtype=torch.float32,
                                        device=self.device
                                        )

        mem_rec = []
        spk_rec = []
        # ANN activations will be all zeros for a pure SNN
        # ann_rec = []

        h1_from_input = torch.einsum("abc,cd->abd", (inputs, self.w1))

        # For pure SNN, the effective SNN mask is all ones.
        # It's passed to _process_hidden_layer_timestep as a parameter
        # but the internal logic of _process_hidden_layer_timestep will use
        # self.snn_mask if it exists, otherwise it defaults to all ones for SNN.
        # So we don't need to pass a 'snn_mask_timestep' here anymore.

        for t in range(seq_len):
            current_h1_input = h1_from_input[:, t]

            mem, syn, hidden_out_t, spikes_t, _ = \
                self._process_hidden_layer_timestep(
                    h1_input_t=current_h1_input,
                    mem=mem,
                    syn=syn,
                    prev_hidden_out=prev_hidden_out,
                )

            mem_rec.append(mem)
            spk_rec.append(spikes_t)
            # ann_rec.append(ann_activations_t) # will be all zeros

            prev_hidden_out = hidden_out_t # For next recurrent step

        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)
        # ann_rec = torch.stack(ann_rec, dim=1)

        h2_input = torch.einsum("abc,cd->abd", (spk_rec, self.w2)) # Output from SNN neurons
        out_rec = self._apply_readout_layer(h2_input)

        auxiliary_outputs = {
            'mem_rec': mem_rec,
            'spikes': spk_rec,
            # 'ann_hidden_activations': ann_rec,
        }
        return out_rec, auxiliary_outputs