import torch


def SNN(inputs, w1, w2, v1, alpha, beta, spike_fn, device, recurrent, snn_mask):
    inputs = inputs.to(device)
    # Make these the same shape as w1 and w2 respectively
    syn = torch.zeros(
        size=(inputs.shape[0], w1.shape[1]), dtype=torch.float32, device=device
    )
    mem = torch.zeros(
        size=(inputs.shape[0], w1.shape[1]), dtype=torch.float32, device=device
    )
    mem_rec = []
    spk_rec = []
    out = torch.zeros(
        size=(inputs.shape[0], w1.shape[1]), dtype=torch.float32, device=device
    )
    h1_from_input = torch.einsum("abc,cd->abd", (inputs, w1))
    # Compute hidden layer activity
    for t in range(inputs.shape[1]):
        h1 = h1_from_input[:, t]
        if recurrent:
            h1 += torch.einsum("ab,bc->ac", (out, v1))
        mthr = mem - 1.0
        out = spike_fn(mthr)
        rst = out.detach()  # We do not want to backprop through the reset

        new_syn = alpha * syn + h1
        new_mem = (beta * mem + syn) * (1.0 - rst)

        mem_rec.append(mem)
        spk_rec.append(out)

        mem = new_mem
        syn = new_syn

    mem_rec = torch.stack(mem_rec, dim=1)
    spk_rec = torch.stack(spk_rec, dim=1)

    # Readout layer
    h2 = torch.einsum("abc,cd->abd", (spk_rec, w2))
    flt = torch.zeros(
        size=(inputs.shape[0], w2.shape[1]), dtype=torch.float32, device=device
    )
    out = torch.zeros(
        size=(inputs.shape[0], w2.shape[1]), dtype=torch.float32, device=device
    )
    out_rec = [out]
    for t in range(inputs.shape[1]):
        new_flt = alpha * flt + h2[:, t]
        new_out = beta * out + flt

        flt = new_flt
        out = new_out

        out_rec.append(out)

    out_rec = torch.stack(out_rec, dim=1)
    other_recs = [mem_rec, spk_rec]
    return out_rec, other_recs


def ANN_with_LIF_output(
    inputs, w1, w2, v1, alpha, beta, spike_fn, device, recurrent, snn_mask
):
    inputs = inputs.to(device)

    # Weights for the hidden layer for RNN is just w1 -- multiplying by inputs just gives us the output at each timestep
    h1 = torch.einsum("abc,cd->abd", (inputs, w1))
    # We could also add a bias term but let's exclude for now to make the merging more simple

    interim_rec = []

    # The difference for a pure ANN is that the hidden and the output layer usually happen at the same time
    # It's really fine that they're different here though if we just compute them sequentially
    out = torch.zeros(
        size=(inputs.shape[0], w1.shape[1]), dtype=torch.float32, device=device
    )

    for t in range(inputs.shape[1]):

        layer_weights = h1[:, t]
        if recurrent:
            layer_weights += torch.einsum("ab,bc->ac", (out, v1))

        # Use a tanh function similar to what's done in other models
        out = torch.tanh(layer_weights)
        interim_rec.append(out)

    interim_rec = torch.stack(interim_rec, dim=1)
    # We will use the same LIF model as before, and just use the raw output as an input to start since it has a window of 1

    # Readout layer
    h2 = torch.einsum("abc,cd->abd", (interim_rec, w2))
    flt = torch.zeros(
        size=(inputs.shape[0], w2.shape[1]), dtype=torch.float32, device=device
    )
    out = torch.zeros(
        size=(inputs.shape[0], w2.shape[1]), dtype=torch.float32, device=device
    )
    out_rec = [out]

    # Convert the h2 readout layer to a rate that can be transmitted to the output layer

    for t in range(inputs.shape[1]):
        new_flt = alpha * flt + h2[:, t]
        new_out = beta * out + flt

        flt = new_flt
        out = new_out

        out_rec.append(out)

    out_rec = torch.stack(out_rec, dim=1)
    other_recs = interim_rec
    return out_rec, other_recs


def Hybrid_RNN_SNN_rec(
    inputs, w1, w2, v1, alpha, beta, spike_fn, device, recurrent, snn_mask
):
    inputs = inputs.to(device)
    # Weight matrix for training, I think this can be used for both SNN and ANN but the training may have to be different
    h1 = torch.einsum("abc,cd->abd", (inputs, w1))
    # Let's just do 2 matrices to keep it clean for now
    h1_ann = h1.clone() * (1.0 - snn_mask)  # ANN neurons
    h1_snn_input = h1.clone() * snn_mask  # SNN neurons
    syn = torch.zeros(
        (inputs.shape[0], w1.shape[1]), device=device, dtype=torch.float32
    )
    mem = torch.zeros(
        (inputs.shape[0], w1.shape[1]), device=device, dtype=torch.float32
    )

    mem_rec = []
    spk_rec = []
    ann_rec = []

    out_ann = torch.zeros(
        (inputs.shape[0], w1.shape[1]), device=device, dtype=torch.float32
    )
    out_snn = torch.zeros(
        (inputs.shape[0], w1.shape[1]), device=device, dtype=torch.float32
    )
    # Compute hidden layer activity
    for t in range(inputs.shape[1]):
        h1_snn = h1_snn_input[:, t]
        if recurrent:
            h1_snn += torch.einsum("ab,bc->ac", (out_snn, v1))
        # SNN neurons

        # Apply the mask to the synaptic input to ensure SNN neurons are only updated with spiking rules
        mem = (mem * snn_mask).clone()
        syn = (syn * snn_mask).clone()

        mthr = mem - 1.0
        out_snn = spike_fn(mthr)
        rst = out_snn.detach()  # We do not want to backprop through the reset

        new_syn = alpha * syn + h1_snn
        new_mem = (beta * mem + syn) * (1.0 - rst)

        mem_rec.append(mem)
        spk_rec.append(out_snn)

        # ANN neurons - don't interact with the synaptic input at all to start
        out_ann = torch.tanh(h1_ann[:, t])
        if recurrent:
            out_ann += torch.einsum("ab,bc->ac", (out_ann, v1))
        ann_rec.append(out_ann)

        mem = new_mem
        syn = new_syn

    mem_rec = torch.stack(mem_rec, dim=1)
    spk_rec = torch.stack(spk_rec, dim=1)
    ann_rec = torch.stack(ann_rec, dim=1)

    h2_snn = torch.einsum("abc,cd->abd", (spk_rec, w2))
    h2_ann = torch.einsum("abc,cd->abd", (ann_rec, w2))

    # We can add the two together to get the final output since the two do not interact at all
    h2 = h2_snn + h2_ann

    flt = torch.zeros(
        (inputs.shape[0], w2.shape[1]), device=device, dtype=torch.float32
    )
    out = torch.zeros(
        (inputs.shape[0], w2.shape[1]), device=device, dtype=torch.float32
    )
    out_rec = [out]
    for t in range(inputs.shape[1]):
        new_flt = alpha * flt + h2[:, t]
        new_out = beta * out + flt

        flt = new_flt
        out = new_out

        out_rec.append(out)

    out_rec = torch.stack(out_rec, dim=1)
    other_recs = [mem_rec, spk_rec, ann_rec]
    return out_rec, other_recs


def Hybrid_RNN_SNN_V1_same_layer(
    inputs, w1, w2, v1, alpha, beta, spike_fn, device, recurrent, snn_mask
):
    inputs = inputs.to(device)
    # Weight matrix for training, I think this can be used for both SNN and ANN but the training may have to be different
    h1 = torch.einsum("abc,cd->abd", (inputs, w1))
    # Let's just do 2 matrices to keep it clean for now
    h1_ann = h1.clone() * (1.0 - snn_mask)  # ANN neurons
    h1_snn_input = h1.clone() * snn_mask  # SNN neurons
    syn = torch.zeros(
        (inputs.shape[0], w1.shape[1]), device=device, dtype=torch.float32
    )
    mem = torch.zeros(
        (inputs.shape[0], w1.shape[1]), device=device, dtype=torch.float32
    )

    mem_rec = []
    spk_rec = []
    ann_rec = []
    # (O_A + O_S) * W_2 = (O_A * W_2 + O_S * W_2) by distributive property, no theoretical change there apart from H1
    # Prove/think about this after more after test it
    out_ann = torch.zeros(
        (inputs.shape[0], w1.shape[1]), device=device, dtype=torch.float32
    )
    out_snn = torch.zeros(
        (inputs.shape[0], w1.shape[1]), device=device, dtype=torch.float32
    )
    # Compute hidden layer activity
    for t in range(inputs.shape[1]):
        out = out_ann + out_snn
        h1_snn = h1_snn_input[:, t] + torch.einsum("ab,bc->ac", (out, v1))
        # SNN neurons

        # Apply the mask to the synaptic input to ensure SNN neurons are only updated with spiking rules
        mem = mem * snn_mask
        syn = syn * snn_mask

        mthr = mem - 1.0
        out_snn = spike_fn(mthr)
        rst = out_snn.detach()  # We do not want to backprop through the reset

        new_syn = alpha * syn + h1_snn
        new_mem = (beta * mem + syn) * (1.0 - rst)

        mem_rec.append(mem)
        spk_rec.append(out_snn)

        # ANN neurons - don't interact with the synaptic input at all to start
        out_ann = torch.tanh(h1_ann[:, t]) + torch.einsum("ab,bc->ac", (out, v1))
        ann_rec.append(out_ann)

        mem = new_mem
        syn = new_syn

    mem_rec = torch.stack(mem_rec, dim=1)
    spk_rec = torch.stack(spk_rec, dim=1)
    ann_rec = torch.stack(ann_rec, dim=1)

    h2_snn = torch.einsum("abc,cd->abd", (spk_rec, w2))
    h2_ann = torch.einsum("abc,cd->abd", (ann_rec, w2))

    # We can add the two together to get the final output since the two do not interact at all
    h2 = h2_snn + h2_ann

    flt = torch.zeros(
        (inputs.shape[0], w2.shape[1]), device=device, dtype=torch.float32
    )
    out = torch.zeros(
        (inputs.shape[0], w2.shape[1]), device=device, dtype=torch.float32
    )
    out_rec = [out]
    for t in range(inputs.shape[1]):
        new_flt = alpha * flt + h2[:, t]
        new_out = beta * out + flt

        flt = new_flt
        out = new_out

        out_rec.append(out)

    out_rec = torch.stack(out_rec, dim=1)
    other_recs = [mem_rec, spk_rec, ann_rec]
    return out_rec, other_recs

def Hybrid_RNN_SNN_V1_same_layer_attn(
    inputs, w1, w2, v1, alpha, beta, spike_fn,
    device, recurrent, snn_mask,
    num_heads: int = 4               
):
    inputs = inputs.to(device)
    B, T, _ = inputs.shape
    H       = w1.shape[1]             

    h1  = torch.einsum("bti,ih->bth", inputs, w1)
    h1_ann      = h1 * (1.0 - snn_mask)   
    h1_snn_inp  = h1 * snn_mask         

    syn = torch.zeros(B, H, device=device)
    mem = torch.zeros_like(syn)

    out_ann = torch.zeros_like(syn)
    out_snn = torch.zeros_like(syn)

    mem_rec, spk_rec, ann_rec = [], [], []

    attn = nn.MultiheadAttention(
                embed_dim=H,
                num_heads=num_heads,
                batch_first=True,
                bias=False                
           ).to(device)

    for t in range(T):
        out_mix = out_ann + out_snn                       

        h1_snn = h1_snn_inp[:, t] + torch.matmul(out_mix, v1)

        q = out_snn.unsqueeze(1)       
        k = v = out_ann.unsqueeze(1)    
        attn_out, _ = attn(q, k, v)     
        h1_snn = h1_snn + attn_out.squeeze(1)

        mem = mem * snn_mask
        syn = syn * snn_mask

        z_thr   = spike_fn(mem - 1.0)      
        rst     = z_thr.detach()
        syn     = alpha * syn + h1_snn
        mem     = (beta * mem + syn) * (1.0 - rst)
        out_snn = z_thr

        out_ann = torch.tanh(h1_ann[:, t]) + torch.matmul(out_mix, v1)

        mem_rec.append(mem)
        spk_rec.append(out_snn)
        ann_rec.append(out_ann)

    mem_rec = torch.stack(mem_rec, 1)      
    spk_rec = torch.stack(spk_rec, 1)
    ann_rec = torch.stack(ann_rec, 1)

    h2  = (torch.einsum("bth,ho->bto", spk_rec, w2) +
           torch.einsum("bth,ho->bto", ann_rec,  w2))

    flt = torch.zeros(B, w2.shape[1], device=device)
    out = torch.zeros_like(flt)
    out_rec = [out]

    for t in range(T):
        flt = alpha * flt + h2[:, t]
        out = beta  * out + flt
        out_rec.append(out)

    out_rec = torch.stack(out_rec, 1)
    return out_rec, [mem_rec, spk_rec, ann_rec]