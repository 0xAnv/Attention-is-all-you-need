import yaml
import wandb
import jax
from pathlib import Path

def load_config(config_path: str | Path) -> dict:
    """Loads a YAML configuration file into a dictionary."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # 1. Preparation: Load Configuration
    config_path = Path("configs/base_config.yaml")
    config = load_config(config_path)
    
    # Let's print the config to verify it loaded correctly
    print("Configuration loaded successfully. Initializing WandB...")

    # 2. Initialization phase
    # Initialize the WandB run
    # 'project' categorizes your work on the dashboard
    # The config argument automatically syncs your hyperparameters to the UI
    wandb.init(
        project="flax-transformer",
        name=f"run_dmodel{config['d_model']}_bs{config['batch_size']}",
        config=config
    )
    
    print("\n--- WandB Config Sync Complete ---")
    print("Hyperparameters being tracked:")
    for k, v in wandb.config.items():
        print(f"  {k}: {v}")

    # TODO: Initialize datasets, model architecture, scheduler, optimizer, etc.
    
    # Graceful Exit
    # At the very end of your script, finish the wandb run to stop tracking
    wandb.finish()

if __name__ == "__main__":
    main()
