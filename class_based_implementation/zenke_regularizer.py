import torch

# Import the updated, gamma-aware regularizer classes
from class_based_implementation.Regularizers import LowerBoundL2, UpperBoundL1, UpperBoundL2

def regularization_loss_zenke(spks, config, gamma=None):
    """
    Calculates Zenke-style regularization loss using the updated, gamma-aware classes.
    """

    # Instantiate the regularizers from the config
    lower_l2_reg = LowerBoundL2(
        strength=config.get("l2_lower", 0),
        threshold=config.get("v2_lower", 0),
        dims=False
    )
    upper_l1_reg = UpperBoundL1(
        strength=config.get("l1_upper", 0),
        threshold=config.get("v1_upper", 0),
        dims=-1     # Population-level
    )
    upper_l2_reg = UpperBoundL2(
        strength=config.get("l2_upper", 0),
        threshold=config.get("v2_upper", 0),
        dims=-1     # Population-level
    )

    # Call each regularizer, passing the optional gamma.
    # The classes themselves now handle the specific logic internally.
    lower_l2 = lower_l2_reg(spks, gamma=gamma)
    upper_l1 = upper_l1_reg(spks, gamma=gamma)
    upper_l2 = upper_l2_reg(spks, gamma=gamma)

    total_loss = lower_l2 + upper_l1 + upper_l2
    return total_loss, lower_l2, upper_l1, upper_l2