# class_based_implementation/models/mine_implementation.py

# Based off of https://github.com/gtegner/mine-pytorch/blob/master/mine/models/mine.py

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import wandb
from torch.utils.data import Dataset, DataLoader # Keep DataLoader for the final wrapped datasets
import math

# New code: EMA Loss Function for running EMA of the MINE loss
from class_based_implementation.models.mine_layers import ConcatLayer

EPS = 1e-6

class MINEDataTupleDataset(Dataset):
    def __init__(self, x_data: torch.Tensor, h_data: torch.Tensor):
        # x_data and h_data should already be correctly prepared (e.g., all H_full, or all H_no_spatial)
        # and should already be on CPU if you want to offload from GPU memory,
        # or kept on GPU if there's enough VRAM and num_workers=0.
        assert len(x_data) == len(h_data), "X and H data must have the same number of samples."
        self.x_data = x_data
        self.h_data = h_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.h_data[idx]


class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema

    return t_log, running_mean



# --- MINE Base Network Definition (LSTM-based) ---
class MINEBaseNetwork(nn.Module):
    def __init__(self, x_feature_dim: int, h_feature_dim: int, hidden_size: int = 100, num_lstm_layers: int = 2, alpha: float = 0.01, dropout=0.1, clamp_value: float = 100):
        super(MINEBaseNetwork, self).__init__()
        lstm_input_size = x_feature_dim + h_feature_dim
        self.concat = ConcatLayer(dim=2)  # Concatenate along the feature dimension given time steps
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout,  # Dropout for regularization
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.running_mean = 0
        self.alpha = alpha  # EMA decay factor
        self.clamp_value = clamp_value  # Clamp value for output if needed

    def T(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get intermediate output before final linear layer.
        This is useful for calculating MINE loss.
        """
        inputs_combined = self.concat(x, h)
        lstm_out, _ = self.lstm(inputs_combined)
        last_timestep_output = lstm_out[:, -1, :]
        output = self.fc(last_timestep_output)
        output = torch.clamp(output, min=-self.clamp_value, max=self.clamp_value)
        return output

    def forward(self, x: torch.Tensor, h_joint: torch.Tensor, h_marginal: torch.Tensor) -> torch.Tensor:
        T_output_joint = self.T(x, h_joint).mean()
        T_output_marginal = self.T(x, h_marginal)

        second_term, self.running_mean = ema_loss(T_output_marginal, self.running_mean, self.alpha)

        return T_output_joint - second_term # We maximize this in optuna settings, so no need for 2 different values


# --- MINELightningModule (PyTorch Lightning Module for a single MINE network) ---
class MINELightningModule(pl.LightningModule):
    def __init__(self, mine_base_network: MINEBaseNetwork, mine_lr: float, mi_type_name: str):
        super().__init__()
        self.mine_base_network = mine_base_network
        self.mine_lr = mine_lr
        self.mi_type_name = mi_type_name
        self.save_hyperparameters(ignore=['mine_base_network'])

    def forward(self, x: torch.Tensor,  h_joint: torch.Tensor, h_marginal: torch.Tensor) -> torch.Tensor:
        return self.mine_base_network(x, h_joint, h_marginal)

    def _common_step(self, batch, step_type: str):
        """
        Common step logic for training, validation, and testing.
        This is used to avoid code duplication in training_step, validation_step, and test_step.
        """
        x_data, h_data = batch

        # Create shuffled H samples within the batch for marginal product
        h_shuffled = h_data[torch.randperm(h_data.size(0), device=h_data.device)]
        mi = self.forward(x_data, h_data, h_shuffled)
        # The version will be in the name
        self.log(f"{step_type}_mi", mi, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return mi

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test")

    def configure_optimizers(self):
        return optim.Adam(self.mine_base_network.parameters(), lr=self.mine_lr)

# --- MIEstimator Class (Updated to coordinate MINELightningModules with pre-processed data) ---
class MIEstimator:
    def __init__(self, x_feature_dim: int, h_feature_dim: int, mine_hidden_size: int = 128, mine_lr: float = 0.001,
                 num_lstm_layers: int = 2, alpha: float = 0.01, dropout: float = 0.1,  # ADD THESE TO MIEstimator's init
                 clamp_value: float = 100,
                 chosen_model: str = "Unknown"):
        """
        Initializes parameters common to all MINE networks.
        Args:
            x_feature_dim (int): Feature dimension of input X per timestep.
            h_feature_dim (int): Feature dimension of hidden layer H per timestep.
            mine_hidden_size (int): Hidden layer size for MINE networks' LSTM.
            mine_lr (float): Learning rate for MINE optimizers.
        """
        self.x_feature_dim = x_feature_dim
        self.h_feature_dim = h_feature_dim
        self.mine_hidden_size = mine_hidden_size
        self.mine_lr = mine_lr
        self.chosen_model_name = chosen_model  # Placeholder for the base network, initialized in each MINE training
        self.num_lstm_layers = num_lstm_layers
        self.alpha = alpha  # EMA decay factor for MINE loss
        self.dropout = dropout  # Dropout rate for MINE LSTM layers
        self.clamp_value = clamp_value  # Clamp value for MINE outputs, if needed
    def estimate_all_mi_and_rates(
        self,
        # These are now pre-processed DataLoaders containing MINEDataTupleDataset
        train_loader_full: DataLoader, train_loader_no_spatial: DataLoader,
        train_loader_no_temporal: DataLoader, train_loader_only_activation: DataLoader,
        val_loader_full: DataLoader, val_loader_no_spatial: DataLoader,
        val_loader_no_temporal: DataLoader, val_loader_only_activation: DataLoader,
        test_loader_full: DataLoader, test_loader_no_spatial: DataLoader,
        test_loader_no_temporal: DataLoader, test_loader_only_activation: DataLoader,
        mine_epochs: int, mine_analysis_project_name: str,
    ) -> dict:
        """
        Estimates all four mutual information values by training separate MINE networks
        and then calculates exploitation rates.
        """
        print("Starting MINE analysis with pre-processed data...")

        mi_values = {}

        # Define the variants and corresponding loaders
        # Note: These loaders are now `torch.utils.data.DataLoader` instances wrapping `MINEDataTupleDataset`
        variant_loaders = {
            "full": (train_loader_full, val_loader_full, test_loader_full),
            "no_spatial": (train_loader_no_spatial, val_loader_no_spatial, test_loader_no_spatial),
            "no_temporal": (train_loader_no_temporal, val_loader_no_temporal, test_loader_no_temporal),
            "only_activation": (train_loader_only_activation, val_loader_only_activation, test_loader_only_activation),
        }

        # Determine MINE training device from the first batch of data
        # Data will already be on CPU from preprocessing, so MINE will train on CPU unless moved to GPU here
        # It's best to let PL Trainer decide based on `accelerator`
        
        for mi_type, (train_loader_variant, val_loader_variant, test_loader_variant) in variant_loaders.items():
            print(f"\n--- Training MINE for I(X; {mi_type.replace('_', ' ').title()}) ---")
            
            # Initialize a new MINEBaseNetwork and MINELightningModule for each MI type
            # This ensures independent training and thus independent MI estimates.
            mine_base_network = MINEBaseNetwork(self.x_feature_dim, self.h_feature_dim, self.mine_hidden_size, self.num_lstm_layers, self.alpha, self.dropout, self.clamp_value)
            mine_lightning_module = MINELightningModule(mine_base_network, self.mine_lr, mi_type)

            # Initialize a new Wandb run for each MINE training process for proper logging
            wandb.init(
                project=mine_analysis_project_name,
                group=f"MINE_{mi_type}_for_{self.chosen_model_name}",
                name=f"MINE_{mi_type}_{self.chosen_model_name}",
                reinit=True,
                config={
                    "mi_type": mi_type,
                    "mine_epochs": mine_epochs,
                    "mine_lr": self.mine_lr,
                    "mine_hidden_size": self.mine_hidden_size,
                    "mine_num_lstm_layers": self.num_lstm_layers, # Also log them for Wandb clarity
                    "mine_alpha": self.alpha,
                    "lstm_dropout": self.dropout,
                    "chosen_model_name": self.chosen_model_name,
                    "clamp_value": self.clamp_value,
                }
            )
            
            # Create a Lightning Trainer for this MINE module
            trainer = pl.Trainer(
                max_epochs=mine_epochs,
                accelerator="gpu" if torch.cuda.is_available() else "cpu", # Use available device
                gradient_clip_val=1.0,  # <-- clip gradients to max norm of 1.0
                devices=1, # One device per MINE training process
                logger=pl.loggers.WandbLogger(log_model=False, project=mine_analysis_project_name), # Log to the current Wandb run
                enable_checkpointing=False,
                enable_progress_bar=True, # Show progress for MINE training
                callbacks=[pl.callbacks.EarlyStopping(
                    monitor=f'val_mi', # Monitor validation MI
                    mode='max',
                    patience=10, # Stop if MI doesn't improve for 10 epochs
                    verbose=False
                )]
            )

            try:
                trainer.fit(mine_lightning_module, train_loader_variant, val_loader_variant)
                
                # After training, run test_step to get the final MI estimate on test set
                # trainer.test returns a list of dictionaries, take the first one
                test_results = trainer.test(mine_lightning_module, dataloaders=test_loader_variant)
                
                final_mi_estimate = 0.0
                if test_results and len(test_results) > 0:
                    final_mi_estimate = test_results[0].get(f'test_{mi_type}_mi', 0.0)
                else:
                    print(f"Warning: No test results found for {mi_type}. MI set to 0.0.")

            except Exception as e:
                print(f"Error training MINE for {mi_type}: {e}")
                import traceback
                traceback.print_exc()
                wandb.log({"mine_training_failed": True, "error_message": str(e)})
                final_mi_estimate = 0.0 # Return 0.0 on error

            mi_values[mi_type] = final_mi_estimate
            print(f"  Final Estimated I(X; {mi_type.replace('_', ' ').title()}): {final_mi_estimate:.4f}")
            wandb.finish() # End the current Wandb run for this MINE training

        print("\nCalculating Exploitation Rates...")
        exploitation_rates = calculate_exploitation_rates(
            ix_si_ti_ai=mi_values.get('full', 0.0), # Use .get with default 0.0 to be safe
            ix_ti_ai=mi_values.get('no_spatial', 0.0),
            ix_si_ai=mi_values.get('no_temporal', 0.0),
            ix_ai=mi_values.get('only_activation', 0.0)
        )
        print(f"Calculated Exploitation Rates: {exploitation_rates}")

        return {
            "mi_full": mi_values.get('full', 0.0),
            "mi_no_spatial": mi_values.get('no_spatial', 0.0),
            "mi_no_temporal": mi_values.get('no_temporal', 0.0),
            "mi_only_activation": mi_values.get('only_activation', 0.0),
            "exploitation_rates": exploitation_rates
        }


# --- Exploitation Rates Calculator (remains unchanged) ---
def calculate_exploitation_rates(
    ix_si_ti_ai: float,
    ix_ti_ai: float,
    ix_si_ai: float,
    ix_ai: float
) -> dict:
    """
    Calculates Spatial Exploitation Rate (SER), Temporal Exploitation Rate (TER),
    and Activation Exploitation Rate (AER) based on provided mutual information values.
    """
    # ... (Your existing calculate_exploitation_rates function) ...
    if ix_si_ti_ai <= 0:
        print("Error: I(X; Si, Ti, Ai) must be positive to calculate exploitation rates.")
        return {
            "SER": None,
            "TER": None,
            "AER": None
        }

    ser = None
    ter = None
    aer = None

    try:
        ser = (ix_si_ti_ai - ix_ti_ai) / ix_si_ti_ai
        ser = max(0.0, min(1.0, ser)) # Clamp to [0, 1]
    except ZeroDivisionError:
        print("Warning: Division by zero for SER calculation. Check ix_si_ti_ai.")
    except Exception as e:
        print(f"Error calculating SER: {e}")

    try:
        ter = (ix_si_ti_ai - ix_si_ai) / ix_si_ti_ai
        ter = max(0.0, min(1.0, ter)) # Clamp to [0, 1]
    except ZeroDivisionError:
        print("Warning: Division by zero for TER calculation. Check ix_si_ti_ai.")
    except Exception as e:
        print(f"Error calculating TER: {e}")

    try:
        aer = ix_ai / ix_si_ti_ai
        aer = max(0.0, min(1.0, aer)) # Clamp to [0, 1]
    except ZeroDivisionError:
        print("Warning: Division by zero for AER calculation. Check ix_si_ti_ai.")
    except Exception as e:
        print(f"Error calculating AER: {e}")

    return {
        "SER": ser,
        "TER": ter,
        "AER": aer
    }
