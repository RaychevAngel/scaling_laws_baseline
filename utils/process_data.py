import json
from typing import List, Tuple, Dict
import os
from datasets import DatasetDict

class TrajectoryProcessor:
    """Processes trajectories into training data for policy and value networks."""
    
    @staticmethod
    def _create_prompt(question: str, state: List[str] = None) -> List[dict]:
        """Create prompt with question and current state (sequence of actions)."""
        if state:
            joined = ''.join(a.strip() + "\n" for a in state)
        else:
            joined = ''
        return [{"role": "user", "content": f"{question.rstrip()}\n{joined}"}]

    def process_policy_trajectory(self, policy_data: List[Tuple[str, str]]) -> List[dict]:
        """
        Process policy data for training.
        
        Args:
            policy_data: List of tuples (question, trajectory)
        
        Returns:
            List of formatted JSON dictionaries for model training
        """
        all_data = []
        for question, trajectory in policy_data:
            actions = trajectory.strip().split('\n')
            state = []
            for action in actions:
                prompt = self._create_prompt(question, state)
                completion = [{"role": "assistant", "content": action + "\n"}]
                data_item = {
                    "prompt": prompt, 
                    "completion": completion
                }
                all_data.append(data_item)
                state.append(action)
        return all_data

    def process_value_trajectory(self, value_data: List[Tuple[str, str, float]]) -> List[dict]:
        """
        Process value data for training.
        
        Args:
            value_data: List of tuples (question, trajectory, label)
        
        Returns:
            List of formatted JSON dictionaries for model training
        """
        all_data = []
        for question, trajectory, label in value_data:
            actions = trajectory.strip().split('\n')
            state = []
            label_str = "1" if label == 1 else "0"
            for action in actions:
                prompt = self._create_prompt(question, state)
                completion = [{"role": "assistant", "content": label_str}]
                data_item = {
                    "prompt": prompt, 
                    "completion": completion
                }
                all_data.append(data_item)
                state.append(action)
            # Add final state with reward
            prompt = self._create_prompt(question, state)
            completion = [{"role": "assistant", "content": label_str}]
            data_item = {
                "prompt": prompt, 
                "completion": completion
            }
            all_data.append(data_item)
        return all_data
    
    @staticmethod
    def _prepare_dataset(train_data: List[dict], dev_data: List[dict], output_dir: str) -> DatasetDict:
        """Convert JSON data to Hugging Face dataset format."""
        # Create Dataset objects directly from the lists of dictionaries
        from datasets import Dataset
        
        train_dataset = Dataset.from_list(train_data)
        dev_dataset = Dataset.from_list(dev_data)
        
        # Create and save the dataset dictionary
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "dev": dev_dataset
        })
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the dataset
        dataset_dict.save_to_disk(output_dir)
        
        return dataset_dict
    
    def export_data(self, policy_data_train: List, value_data_train: List, 
                    policy_data_dev: List, value_data_dev: List, 
                    policy_output_dir: str, value_output_dir: str) -> None:
        """
        Export processed policy and value training data to files and convert to datasets.
        
        Args:
            policy_data_train: Tuple containing (train_policy_data)
            value_data_train: Tuple containing (train_value_data)
            policy_data_dev: Tuple containing (dev_policy_data)
            value_data_dev: Tuple containing (dev_value_data)
            policy_output_dir: Directory to save the policy dataset
            value_output_dir: Directory to save the value dataset
        """
        
        # Process the data into the correct format
        policy_data_train = self.process_policy_trajectory(policy_data_train)
        value_data_train = self.process_value_trajectory(value_data_train)
        policy_data_dev = self.process_policy_trajectory(policy_data_dev)
        value_data_dev = self.process_value_trajectory(value_data_dev)
        
        # Create Hugging Face datasets from the processed data
        policy_output_dir = os.path.join(os.path.dirname(policy_output_dir), "policy")
        value_output_dir = os.path.join(os.path.dirname(value_output_dir), "value")
        
        self._prepare_dataset(policy_data_train, policy_data_dev, policy_output_dir)
        self._prepare_dataset(value_data_train, value_data_dev, value_output_dir)
        
        print(f"Policy dataset saved to {policy_output_dir}")
        print(f"Value dataset saved to {value_output_dir}")
    
    def export_evaluation_data(self, data: Dict, output_path: str) -> None:
        """
        Export evaluation results to a JSON file.
        
        Args:
            data: Dictionary containing evaluation metrics
            output_path: Path to save the evaluation results
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Evaluation data exported to {output_path}")
        except Exception as e:
            print(f"Error exporting evaluation data: {e}")
    
    @staticmethod
    def _write_data_to_file(data: List[dict], file_path: str, data_type: str) -> None:
        """Helper function to write data to a specified file path."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
            print(f"{data_type} data exported to {file_path}")
        except Exception as e:
            print(f"Error exporting {data_type} data: {e}")


if __name__ == "__main__":
    processor = TrajectoryProcessor()
    question = "3 7 11 12"
    trajectory = "3+11=14 (left: 7, 12, 14)\n7/14=0.5 (left: 12, 0.5)\n12/0.5=24.0 (left: 24.0)\nThe solution is: 12/(7/(3+11))=24.0."
    
    train_policy_data = [(question, trajectory)]
    train_value_data = [(question, trajectory, 1.0)]
    dev_policy_data = [(question, trajectory)]
    dev_value_data = [(question, trajectory, 0.0)]
        
        # Define output paths
    output_paths = {
        'train_policy_data_path': '../data/iter1/135_135/policy_training_data.jsonl',
        'train_value_data_path': '../data/iter1/135_135/value_training_data.jsonl',
        'dev_policy_data_path': '../data/iter1/135_135/policy_dev_data.jsonl',
        'dev_value_data_path': '../data/iter1/135_135/value_dev_data.jsonl',
        }
    
    processor.export_data((train_policy_data, train_value_data), (dev_policy_data, dev_value_data), output_paths)

