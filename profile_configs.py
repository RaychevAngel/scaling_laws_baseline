import yaml
import os
import subprocess
import time
import json
from pathlib import Path
import argparse
import copy
import datetime
import sys  # Added for pip install call
from tabulate import tabulate  # Add tabulate for better formatting

def run_profile(model_type="policy", configs=None, max_steps=None):
    """run a series of profiling experiments with different configs"""
    if configs is None:
        # default configs to test if none provided
        configs = [
            {
                "name": "baseline",
                "changes": {}  # no changes from base config
            },
            {
                "name": "small_batch",
                "changes": {"per_device_train_batch_size": 64, "accumulation_steps": 4}
            },
            {
                "name": "large_batch",
                "changes": {"per_device_train_batch_size": 256, "accumulation_steps": 1}
            },
            {
                "name": "medium_batch_accum",
                "changes": {"per_device_train_batch_size": 128, "accumulation_steps": 2}
            },
            {
                "name": "gradient_checkpointing",
                "changes": {"use_gradient_checkpointing": True}
            },
            {
                "name": "checkpointing_small_batch",
                "changes": {
                    "use_gradient_checkpointing": True,
                    "per_device_train_batch_size": 64,
                    "accumulation_steps": 4
                }
            },
            {
                "name": "workers_2",
                "changes": {"dataloader_num_workers": 2}
            },
            {
                "name": "workers_4",
                "changes": {"dataloader_num_workers": 4}
            },
            {
                "name": "low_memory",
                "changes": {
                    "per_device_train_batch_size": 32,
                    "use_gradient_checkpointing": True,
                    "dataloader_num_workers": 1
                }
            },
            {
                "name": "high_performance",
                "changes": {
                    "per_device_train_batch_size": 256,
                    "dataloader_num_workers": 4,
                    "prefetch_factor": 2,
                    "compile_model": True
                }
            }
        ]

    # determine which config file and test script to use
    if model_type == "policy":
        base_config_path = "train/config_policy.yaml"
        test_script = "test_policy.py"
    else:
        base_config_path = "train/config_value.yaml"
        test_script = "test_value.py"

    # load base config
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # create timestamp for this profiling run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # create main profiling results directory if it doesn't exist
    results_base_dir = Path("profiling_results")
    results_base_dir.mkdir(exist_ok=True)

    # create timestamp subdirectory for this specific run
    results_dir = results_base_dir / f"{model_type}_{timestamp}"
    results_dir.mkdir(exist_ok=True)

    # save original config for reference
    with open(results_dir / "original_config.yaml", "w") as f:
        yaml.dump(base_config, f, default_flow_style=False)

    results = []

    # run each configuration
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Running configuration: {config['name']}")
        print(f"Changes: {config['changes']}")
        print(f"{'='*50}\n")

        # create a deep copy of the base config
        modified_config = copy.deepcopy(base_config)

        # apply changes from this config
        for key, value in config["changes"].items():
            modified_config[key] = value

        # enable profiling for this run
        modified_config["enable_profiling"] = True

        # save to a temporary config file
        temp_config_path = results_dir / f"config_{model_type}_{config['name']}.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(modified_config, f, default_flow_style=False)

        # set environment variable to use this config file
        env = os.environ.copy()
        env["PROFILE_CONFIG_PATH"] = str(temp_config_path)

        # add max_steps limiting if specified
        env_vars = []
        if max_steps:
            env["MAX_STEPS"] = str(max_steps)
            env_vars.append(f"MAX_STEPS={max_steps}")

        # record start time
        start_time = time.time()

        # create temporary wrapper script
        wrapper_script = results_dir / f"run_{config['name']}.py"
        with open(wrapper_script, "w") as f:
            f.write(f"""
import yaml
import os
import sys

# load the specified config
config_path = os.environ.get('PROFILE_CONFIG_PATH')
if not config_path:
    print("No config path specified in PROFILE_CONFIG_PATH")
    sys.exit(1)

# limit steps if specified
max_steps = os.environ.get('MAX_STEPS')
if max_steps:
    print(f"Limiting run to {{max_steps}} steps")

# load the original script
sys.path.insert(0, '.')
with open('{test_script}') as f:
    exec(f.read())
""")

        # run training with this config
        cmd = ["python", str(wrapper_script)]
        print(f"Running: {' '.join(cmd)} with {' '.join(env_vars)}")

        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True
        )

        # record elapsed time
        elapsed = time.time() - start_time

        # save logs
        with open(results_dir / f"{config['name']}_log.txt", "w") as f:
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)

        # extract memory usage from output if available
        max_memory = "N/A"
        memory_mb = float('inf')  # Default for sorting

        for line in result.stdout.split("\n"):
            if "CUDA memory" in line or "GPU memory" in line:
                max_memory = line.strip()
                # Try to extract MB value for sorting
                try:
                    # Simple extraction assuming format like "CUDA memory: 2345 MB"
                    memory_parts = max_memory.split()
                    for i, part in enumerate(memory_parts):
                        if part.isdigit():
                            memory_mb = float(part)
                            break
                except:
                    pass

        # Extract loss values if available
        best_loss = float('inf')
        for line in result.stdout.split("\n"):
            if "eval_loss" in line:
                try:
                    # Try to extract loss value
                    loss_parts = line.split("eval_loss")
                    if len(loss_parts) > 1:
                        loss_value = loss_parts[1].strip().split()[0].strip(":")
                        current_loss = float(loss_value)
                        best_loss = min(best_loss, current_loss)
                except:
                    pass

        results.append({
            "config_name": config["name"],
            "elapsed_time": elapsed,
            "exit_code": result.returncode,
            "max_memory": max_memory,
            "memory_mb": memory_mb,
            "best_loss": best_loss if best_loss != float('inf') else None,
            "config_changes": config["changes"]
        })

        print(f"Completed in {elapsed:.2f}s with exit code {result.returncode}")

    # save overall results
    with open(results_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # create a readable summary
    with open(results_dir / "summary.md", "w") as f:
        f.write(f"# Profiling Results - {timestamp}\n\n")
        f.write(f"Model type: {model_type}\n\n")
        f.write("| Configuration | Time (s) | Exit Code | Max Memory | Best Loss |\n")
        f.write("|--------------|----------|-----------|------------|----------|\n")
        for result in results:
            loss_str = f"{result['best_loss']:.6f}" if result['best_loss'] is not None else "N/A"
            f.write(f"| {result['config_name']} | {result['elapsed_time']:.2f} | {result['exit_code']} | {result['max_memory']} | {loss_str} |\n")

        f.write("\n\n## Configuration Details\n\n")
        for result in results:
            f.write(f"### {result['config_name']}\n")
            f.write("```python\n")
            f.write(json.dumps(result['config_changes'], indent=2))
            f.write("\n```\n\n")

    # Initialize ranking variables to avoid UnboundLocalError
    successful_results = []
    time_sorted = []
    memory_sorted = []
    loss_sorted = []
    sorted_combined = []

    # Create rankings in multiple categories
    try:
        # Filter for successful runs
        successful_results = [r for r in results if r["exit_code"] == 0]

        time_sorted = sorted(successful_results, key=lambda x: x["elapsed_time"])
        memory_sorted = sorted(successful_results, key=lambda x: x["memory_mb"])

        # Only sort by loss if we have valid loss values
        loss_results = [r for r in successful_results if r["best_loss"] is not None]
        if loss_results:
            loss_sorted = sorted(loss_results, key=lambda x: x["best_loss"])
        else:
            loss_sorted = []

        # Create ranking tables
        time_table = []
        for i, result in enumerate(time_sorted[:5]):  # Top 5
            time_table.append([
                i+1,
                result["config_name"],
                f"{result['elapsed_time']:.2f}s",
                result["max_memory"]
            ])

        memory_table = []
        for i, result in enumerate(memory_sorted[:5]):  # Top 5
            memory_table.append([
                i+1,
                result["config_name"],
                result["max_memory"],
                f"{result['elapsed_time']:.2f}s"
            ])

        loss_table = []
        for i, result in enumerate(loss_sorted[:5]):  # Top 5
            loss_table.append([
                i+1,
                result["config_name"],
                f"{result['best_loss']:.6f}",
                f"{result['elapsed_time']:.2f}s"
            ])

        # Save to rankings file
        with open(results_dir / "rankings.md", "w") as f:
            f.write(f"# Configuration Rankings\n\n")

            f.write("## Fastest Configurations\n\n")
            f.write(tabulate(time_table, headers=["Rank", "Configuration", "Time", "Memory"], tablefmt="pipe"))
            f.write("\n\n")

            f.write("## Lowest Memory Configurations\n\n")
            f.write(tabulate(memory_table, headers=["Rank", "Configuration", "Memory", "Time"], tablefmt="pipe"))
            f.write("\n\n")

            if loss_table:
                f.write("## Best Loss Configurations\n\n")
                f.write(tabulate(loss_table, headers=["Rank", "Configuration", "Loss", "Time"], tablefmt="pipe"))
                f.write("\n\n")

            # Build combined score
            f.write("## Best Overall Configurations\n\n")
            f.write("*Based on combined ranking across time, memory, and loss (if available)*\n\n")

            combined_scores = {}
            for i, result in enumerate(time_sorted):
                name = result["config_name"]
                if name not in combined_scores:
                    combined_scores[name] = {}
                combined_scores[name]["time_rank"] = i+1

            for i, result in enumerate(memory_sorted):
                name = result["config_name"]
                if name not in combined_scores:
                    combined_scores[name] = {}
                combined_scores[name]["memory_rank"] = i+1

            for i, result in enumerate(loss_sorted):
                name = result["config_name"]
                if name not in combined_scores:
                    combined_scores[name] = {}
                combined_scores[name]["loss_rank"] = i+1

            # Calculate average ranking
            for name, scores in combined_scores.items():
                valid_ranks = [v for k, v in scores.items() if "rank" in k]
                combined_scores[name]["avg_rank"] = sum(valid_ranks) / len(valid_ranks)

            # Sort by average rank
            sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1]["avg_rank"])

            combined_table = []
            for i, (name, scores) in enumerate(sorted_combined[:5]):  # Top 5
                time_rank = scores.get("time_rank", "N/A")
                memory_rank = scores.get("memory_rank", "N/A")
                loss_rank = scores.get("loss_rank", "N/A")
                combined_table.append([
                    i+1,
                    name,
                    f"{scores['avg_rank']:.1f}",
                    time_rank,
                    memory_rank,
                    loss_rank
                ])

            f.write(tabulate(combined_table,
                             headers=["Rank", "Configuration", "Avg Rank", "Time Rank", "Memory Rank", "Loss Rank"],
                             tablefmt="pipe"))
    except Exception as e:
        print(f"Error generating rankings: {e}")

    # Print summary to console
    print("\n" + "="*60)
    print(f"PROFILING RESULTS SUMMARY ({len(successful_results)} successful runs)")
    print("="*60)

    # Print top 3 by speed
    if time_sorted:
        print("\nFastest configurations:")
        for i, result in enumerate(time_sorted[:3]):
            print(f"  {i+1}. {result['config_name']} ({result['elapsed_time']:.2f}s)")

    # Print top 3 by memory
    if memory_sorted:
        print("\nLowest memory usage:")
        for i, result in enumerate(memory_sorted[:3]):
            print(f"  {i+1}. {result['config_name']} ({result['max_memory']})")

    # Print top 3 by loss
    if loss_sorted:
        print("\nBest loss:")
        for i, result in enumerate(loss_sorted[:3]):
            print(f"  {i+1}. {result['config_name']} ({result['best_loss']:.6f})")

    print("\nRecommended configurations:")
    if sorted_combined:
        for i, (name, scores) in enumerate(sorted_combined[:3]):
            print(f"  {i+1}. {name} (avg rank: {scores['avg_rank']:.1f})")
            # Print the specific changes for this config
            config_changes = next((r["config_changes"] for r in results if r["config_name"] == name), {})
            if config_changes:
                changes_str = ", ".join(f"{k}={v}" for k, v in config_changes.items())
                print(f"     Changes: {changes_str}")
    else:
        print("  No combined rankings available")

    print(f"\nDetailed results saved to: {results_dir}")
    print(f"See {results_dir}/rankings.md for full rankings")
    return results_dir

if __name__ == "__main__":
    # Check if tabulate is installed
    try:
        import tabulate
    except ImportError:
        print("Installing tabulate package for better formatting...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
        import tabulate

    parser = argparse.ArgumentParser(description="Profile training with different configurations")
    parser.add_argument("--model", choices=["policy", "value"], default="policy", help="Which model to profile")
    parser.add_argument("--max-steps", type=int, help="Limit training to this many steps (optional)")

    args = parser.parse_args()
    run_profile(model_type=args.model, max_steps=args.max_steps)