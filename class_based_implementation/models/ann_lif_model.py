import torch
from class_based_implementation.models.base_temporal_model import BaseTemporalModel

class ANN_with_LIF_output(BaseTemporalModel):
    """
    ANN model with a LIF-like output layer, inheriting from BaseTemporalModel.
    Spike regularization is not typically applied to ANN hidden units.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.spike_fn is not None:
             print("Note: spike_fn is generally not used for ANN hidden layers in this model. Only for LIF-like output if _apply_readout_layer were spiking, but it's not.")
        # Ensure that snn_mask is None or all zeros for pure ANN
        if self.snn_mask is not None and self.snn_mask.sum().item() > 0:
            raise ValueError("ANN_with_LIF_output should not have SNN neurons in the hidden layer (snn_mask should be None or all zeros).")


    def forward(self, inputs: torch.Tensor):
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        # For ANN, mem and syn are not used in the hidden layer, but keep for _process_hidden_layer_timestep signature
        # mem = torch.zeros((batch_size, self.hidden_features), dtype=torch.float32, device=self.device)
        # syn = torch.zeros((batch_size, self.hidden_features), dtype=torch.float32, device=self.device)
        prev_hidden_out = torch.zeros((batch_size, self.hidden_features), dtype=torch.float32, device=self.device)

        ann_rec = []
        # spk_rec = [] # will be all zeros for a pure ANN

        h1_from_input = torch.einsum("abc,cd->abd", (inputs, self.w1))

        # No snn_mask_timestep needed here, handled internally in _process_hidden_layer_timestep
        # where it defaults to all zeros (for ANN) if self.snn_mask is not set to all ones.

        for t in range(seq_len):
            current_h1_input = h1_from_input[:, t]

            _, _, hidden_out_t, _, ann_activations_t = \
                self._process_hidden_layer_timestep(
                    h1_input_t=current_h1_input,
                    mem=None,
                    syn=None,
                    prev_hidden_out=prev_hidden_out,
                )

            ann_rec.append(ann_activations_t)
            # spk_rec.append(spikes_t) # will be all zeros

            prev_hidden_out = hidden_out_t

        ann_rec = torch.stack(ann_rec, dim=1)
        # spk_rec = torch.stack(spk_rec, dim=1)


        h2_input = torch.einsum("abc,cd->abd", (ann_rec, self.w2))
        out_rec = self._apply_readout_layer(h2_input)

        auxiliary_outputs = {
            'ann_hidden_activations': ann_rec,
            # 'spikes': spk_rec, # For consistency, even if empty
        }
        return out_rec, auxiliary_outputs