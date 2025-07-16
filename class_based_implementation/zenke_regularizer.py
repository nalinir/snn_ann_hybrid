import torch
# from class_based_implementation.archive.Regularizers import UpperBoundL1, UpperBoundL2, LowerBoundL2
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

