import yaml
from utils.env_config import get_hf_user

def main():
    # Get HF username from env
    hf_user = get_hf_user()
    print(f"HF Username: {hf_user}")

    # Test with a sample config
    sample_config = {
        "model_name": "AngelRaychev/test_model",  # Source model - keep as is
        "hub_model_id": "AngelRaychev/test_model_2",  # Target model - replace username
        "policy_model": "AngelRaychev/policy_iteration_",  # Source model - keep as is
        "value_model": "AngelRaychev/value_iteration_"   # Source model - keep as is
    }

    # Only update the hub_model_id for saving, not source models
    updated_config = sample_config.copy()
    if "hub_model_id" in updated_config:
        parts = updated_config["hub_model_id"].split("/")
        if len(parts) > 1:
            updated_config["hub_model_id"] = f"{hf_user}/{parts[1]}"

    # Print before and after
    print("\nOriginal config:")
    print(f"  model_name (source): {sample_config['model_name']}")
    print(f"  hub_model_id (target): {sample_config['hub_model_id']}")

    print("\nUpdated config:")
    print(f"  model_name (source - unchanged): {updated_config['model_name']}")
    print(f"  hub_model_id (target - updated): {updated_config['hub_model_id']}")

    print(f"\nNow loads FROM {updated_config['model_name']} and saves TO {updated_config['hub_model_id']}")

if __name__ == "__main__":
    main()