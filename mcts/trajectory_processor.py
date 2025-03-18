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

    @staticmethod
    def _format_data(prompt: List[dict], completion: str) -> str:
        """Format as JSON string for model training."""
        return json.dumps({"prompt": prompt, "completion": [{"role": "assistant", "content": completion}]})

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
                all_data.append({"prompt": prompt, "completion": completion})
                state.append(action)  # state grows as actions are added
        return all_data

    def process_value_trajectory(self, value_data: List[Tuple[str, str, float]]) -> List[dict]:
        """
        Process value data for training.
        
        Args:
            value_data: List of tuples (question, trajectory, reward)
        
        Returns:
            List of formatted JSON dictionaries for model training
        """
        all_data = []
        for question, trajectory, reward in value_data:
            actions = trajectory.strip().split('\n')
            state = []
            for action in actions:
                prompt = self._create_prompt(question, state)
                completion = [{"role": "assistant", "content": str(reward)}]
                all_data.append({"prompt": prompt, "completion": completion})
                state.append(action)
            # Add final state with reward
            prompt = self._create_prompt(question, state)
            completion = [{"role": "assistant", "content": str(reward)}]
            all_data.append({"prompt": prompt, "completion": completion})
        return all_data
    
    @staticmethod
    def _prepare_dataset(train_data: List[dict], val_data: List[dict], output_dir: str) -> DatasetDict:
        """Convert JSON data to Hugging Face dataset format."""
        # Create Dataset objects directly from the lists of dictionaries
        from datasets import Dataset
        
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        # Create and save the dataset dictionary
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the dataset
        dataset_dict.save_to_disk(output_dir)
        
        return dataset_dict
    
    def export_data(self, train_data: Tuple[List, List], val_data: Tuple[List, List], output_paths: Dict[str, str]) -> None:
        """
        Export processed policy and value training data to files and convert to datasets.
        
        Args:
            train_data: Tuple containing (train_policy_data, train_value_data)
            val_data: Tuple containing (val_policy_data, val_value_data)
            output_paths: Dictionary containing output paths for different data types
        """
        # Unpack the raw data
        train_policy_tuples, train_value_tuples = train_data
        val_policy_tuples, val_value_tuples = val_data
        
        # Process the data into the correct format
        train_policy_data = self.process_policy_trajectory(train_policy_tuples)
        train_value_data = self.process_value_trajectory(train_value_tuples)
        val_policy_data = self.process_policy_trajectory(val_policy_tuples)
        val_value_data = self.process_value_trajectory(val_value_tuples)
        
        # Create Hugging Face datasets from the processed data
        policy_output_dir = os.path.join(os.path.dirname(output_paths['train_policy_data_path']), "policy_ds")
        value_output_dir = os.path.join(os.path.dirname(output_paths['train_value_data_path']), "value_ds")
        
        self._prepare_dataset(train_policy_data, val_policy_data, policy_output_dir)
        self._prepare_dataset(train_value_data, val_value_data, value_output_dir)
        
        print(f"Policy dataset saved to {policy_output_dir}")
        print(f"Value dataset saved to {value_output_dir}")
    
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
    val_policy_data = [(question, trajectory)]
    val_value_data = [(question, trajectory, 0.0)]
        
        # Define output paths
    output_paths = {
        'train_policy_data_path': '../data/iter1/135_135/policy_training_data.jsonl',
        'train_value_data_path': '../data/iter1/135_135/value_training_data.jsonl',
        'val_policy_data_path': '../data/iter1/135_135/policy_validation_data.jsonl',
        'val_value_data_path': '../data/iter1/135_135/value_validation_data.jsonl',
        }
    
    processor.export_data((train_policy_data, train_value_data), (val_policy_data, val_value_data), output_paths)

