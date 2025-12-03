# class_based_implementation/hparams_config.py
import numpy as np
import optuna

# --- CENTRALIZED HYPERPARAMETER DEFINITIONS ---
# Define search ranges for TPE.
# For GridSampler:
# - If 'grid_values' is specified, those exact values are used.
# - Otherwise, for float/int with 'step', np.arange will generate the grid values.
# - For float (log) or int (no step), you MUST specify 'grid_values' for GridSearch.
# - For categorical, 'choices' are used as grid values.
hyperparameter_definitions = {
    "alpha": {"type": "float", "low": 0.5, "high": 0.9, "step": 0.05},
    "beta": {"type": "float", "low": 0.8, "high": 0.99, "step": 0.01},
    "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True,
           "grid_values": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]},
    "hidden_features": {"type": "int", "low": 128, "high": 512, "step": 64},
    "zenke_enabled": {"type": "categorical", "choices": [True, False]},

    # Conditional Zenke parameters
    "l2_lower": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True, "grid_values": [1e-5, 1e-4, 1e-3, 1e-2]},
    "v2_lower": {"type": "float", "low": 0.001, "high": 0.01, "log": True, "grid_values": [0.001, 0.005, 0.01]},
    "l1_upper": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True, "grid_values": [1e-5, 1e-4, 1e-3, 1e-2]},
    "v1_upper": {"type": "float", "low": 0.1, "high": 5.0, "log": True, "grid_values": [0.1, 1.0, 5.0]},
    "l2_upper": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True, "grid_values": [1e-5, 1e-4, 1e-3, 1e-2]},
    "v2_upper": {"type": "float", "low": 0.01, "high": 0.5, "log": True, "grid_values": [0.01, 0.1, 0.5]},
}

def get_grid_search_space():
    """Generates the grid search space dictionary from hyperparameter_definitions."""
    grid_search_space = {}
    for param_name, definition in hyperparameter_definitions.items():
        if "grid_values" in definition:
            grid_search_space[param_name] = definition["grid_values"]
        elif definition["type"] == "float" and "step" in definition:
            grid_search_space[param_name] = list(np.arange(definition["low"], definition["high"] + definition["step"], definition["step"]))
        elif definition["type"] == "int" and "step" in definition:
            grid_search_space[param_name] = list(range(definition["low"], definition["high"] + definition["step"], definition["step"]))
        elif definition["type"] == "categorical":
            grid_search_space[param_name] = definition["choices"]
        else:
            raise ValueError(
                f"Parameter '{param_name}' does not have 'grid_values' specified, "
                "and cannot be automatically generated for GridSearch (e.g., log-uniform floats or int without step)."
                "Please add 'grid_values' to its definition in hparams_config.py."
            )
    return grid_search_space

def suggest_tpe_params(trial: optuna.Trial):
    """Suggests parameters for a TPE trial based on hyperparameter_definitions."""
    params = {}
    for name, definition in hyperparameter_definitions.items():
        if definition["type"] == "float":
            if "step" in definition:
                params[name] = trial.suggest_float(name, definition["low"], definition["high"], step=definition["step"])
            elif "log" in definition and definition["log"]:
                params[name] = trial.suggest_float(name, definition["low"], definition["high"], log=True)
            else:
                params[name] = trial.suggest_float(name, definition["low"], definition["high"])
        elif definition["type"] == "int":
            if "step" in definition:
                params[name] = trial.suggest_int(name, definition["low"], definition["high"], step=definition["step"])
            else:
                params[name] = trial.suggest_int(name, definition["low"], definition["high"])
        elif definition["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, definition["choices"])
        else:
            raise ValueError(f"Unsupported parameter type in definition for {name}: {definition['type']}")
    return params