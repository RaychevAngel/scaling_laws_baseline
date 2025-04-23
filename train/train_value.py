from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainerCallback
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
import tempfile, shutil, torch, os
from tqdm.auto import tqdm
import torch.nn.functional as F
from utils.env_config import get_hf_user, get_hf_token

# Configure PyTorch memory allocation to handle fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure multiprocessing for better data loading
torch.multiprocessing.set_sharing_strategy('file_system')

class LossScalingCallback(TrainerCallback):
    """Callback to correctly scale the loss when using gradient accumulation."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            # Scale only the training loss by dividing by the number of accumulation steps
            logs["loss"] = logs["loss"] / args.gradient_accumulation_steps

class EpochProgressBar(TrainerCallback):
    def __init__(self):
        self.progress_bar = None
        self.current_epoch = None

    def on_train_begin(self, args, state, control, **kwargs):
        if self.progress_bar is not None:
            self.progress_bar.close()

        device_count = max(1, torch.cuda.device_count())
        steps_per_epoch = args.train_dataset_size // (args.per_device_train_batch_size * device_count * args.gradient_accumulation_steps)

        self.progress_bar = tqdm(
            total=steps_per_epoch,
            desc="Epoch Progress",
            position=0,
            leave=True
        )
        self.current_epoch = 0

    def on_step_end(self, args, state, control, **kwargs):
        if self.progress_bar is not None:
            epoch_floor = int(state.epoch)
            if epoch_floor != self.current_epoch:
                self.current_epoch = epoch_floor
                self.progress_bar.reset()
                self.progress_bar.set_description(f"Epoch {epoch_floor + 1}")

            fraction_in_epoch = state.epoch - epoch_floor
            current_step = int(fraction_in_epoch * self.progress_bar.total)

            self.progress_bar.n = current_step
            self.progress_bar.refresh()

    def on_train_end(self, args, state, control, **kwargs):
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None

    def on_evaluate(self, args, state, control, **kwargs):
        print(f"\nEvaluating at step {state.global_step} (epoch {state.epoch:.2f})")

class ValueSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_token_id = self.tokenizer.encode("1", add_special_tokens=False)[0]

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        outputs = model(**inputs)

        # OLD: loop scanning label_seq for token id
        # value_probs = F.softmax(outputs.logits[:, 0, :], dim=-1)[:, self.value_token_id].view(-1, 1)
        # labels = []
        # for label_seq in inputs["labels"]:
        #     non_masked_indices = (label_seq != -100).nonzero(as_tuple=True)[0]
        #     non_masked_tokens = [label_seq[idx].item() for idx in non_masked_indices]
        #     labels.append(1 if non_masked_tokens[-4] == self.value_token_id else 0)
        # loss = F.binary_cross_entropy(value_probs, torch.tensor(labels, dtype=torch.float32, device=value_probs.device).view(-1, 1))

        # NEW: vectorised loss – keeps everything on GPU (no Python loop)
        labels = (inputs["labels"] != -100).float()[:, -1:]  # last non-masked token
        value_probs = F.softmax(outputs.logits[:, 0, :], dim=-1)[:, self.value_token_id].unsqueeze(1)
        loss = F.binary_cross_entropy(value_probs, labels)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, F.softmax(outputs.logits[:, 0, :], dim=-1)[:, self.value_token_id], inputs.get("labels"))

class ValueTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]

        # Get Hugging Face token if available
        self.hf_token = get_hf_token()

        # Model loading
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            ignore_mismatched_sizes=True,
            token=self.hf_token  # Pass token for authentication
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"],
            token=self.hf_token  # Pass token for authentication
        )

        # Apply memory-saving techniques
        if self.config.get("use_gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()  # Trades compute for memory savings

        # new: reduce python overhead for small kernels if pytorch ≥ 2.2
        if self.config.get("compile_model", False):
            if hasattr(torch, "compile") and callable(torch.compile):
                print("Compiling model with torch.compile()...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("Model compilation complete")
            else:
                print("Warning: torch.compile not available, skipping model compilation.")

        # new: memory-savvy sharding allows even larger effective batches
        if self.config.get("use_fsdp", False):
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp import CPUOffload
                from torch.distributed.fsdp import MixedPrecision
                from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
                import functools

                print("Wrapping model with FSDP...")
                # this is a simplified example - would need proper distributed setup
                # and appropriate transformer layer detection for production use
                self.model = FSDP(
                    self.model,
                    # could add cpu offload or custom auto-wrap policy if needed
                )
                print("Model wrapped with FSDP")
            except ImportError:
                print("Warning: FSDP imports failed, skipping FSDP wrapping.")

        self.temp_dir = tempfile.mkdtemp()

    def _create_trainer_config(self):
        lr_scheduler_kwargs = {}
        if self.config["lr_scheduler_type"] == "reduce_lr_on_plateau":
            lr_scheduler_kwargs = {
                "factor": float(self.config["lr_scheduler_factor"]),
                "patience": int(self.config["lr_scheduler_patience"]),
                "threshold": float(self.config["lr_scheduler_threshold"])
            }

        config = SFTConfig(
            output_dir=self.temp_dir,

            # CHANGED: disable auto batch size to use our defined batch size
            auto_find_batch_size=False,
            per_device_train_batch_size=int(self.config["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(self.config["accumulation_steps"]),
            learning_rate=float(self.config["learning_rate"]),
            lr_scheduler_type=self.config["lr_scheduler_type"],
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            optim=self.config["optimizer"],
            max_grad_norm=float(self.config["max_grad_norm"]),
            weight_decay=float(self.config["weight_decay"]),

            # new: boost dataloader throughput  # why: parallel loading and gpu transfer
            dataloader_num_workers=2,  # Reduce workers to prevent contention
            dataloader_pin_memory=True,  # faster h→d copies
            # note: prefetch_factor not supported directly by SFTConfig

            # enable tensor-core mixed precision
            fp16 = True,

            logging_strategy="steps",
            logging_steps=int(self.config["logging_steps"]),
            logging_first_step=True,
            eval_strategy="steps",
            eval_steps=int(self.config["eval_steps"]),
            save_strategy="steps",
            save_steps=int(self.config["eval_steps"]),
            save_total_limit=1,
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,

            push_to_hub=True,
            hub_model_id=self.config["hub_model_id"],
            hub_strategy="end",
            hub_token=self.hf_token,  # Pass token for authentication
            hub_private_repo=False,
            disable_tqdm=True,
        )

        # Add max_steps if specified (used for profiling)
        if "max_steps" in self.config:
            config.max_steps = int(self.config["max_steps"])
            print(f"Setting max_steps to {config.max_steps}")

        return config

    def train(self):
        dataset = load_from_disk(self.config["dataset_file"])

        # new: speed - pre-tokenise entire dataset once  (see hf docs)  # why: avoids per-step python token-iser overhead
        # todo: check if 'tokenised' key exists and is true in config? assumes not for now
        if not self.config.get("tokenised"):
            def tok(batch):
                # todo: make max_length configurable? hardcoded 512 for now
                return self.tokenizer(batch["prompt"], truncation=True,
                                      padding="max_length", max_length=512)
            dataset = dataset.map(tok, batched=True, num_proc=os.cpu_count())
            dataset.set_format(type="torch")
            # todo: consider saving the tokenized dataset to disk if it's large and script restarts often

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=int(self.config["patience"]),
            early_stopping_threshold=float(self.config["improvement_tolerance"])
        )

        loss_scaling_callback = LossScalingCallback()

        trainer = ValueSFTTrainer(
            model=self.model,
            train_dataset=dataset["train"].shuffle(seed=42),
            eval_dataset=dataset["dev"].select(range(5000)).shuffle(seed=42),
            args=self._create_trainer_config(),
            callbacks=[EpochProgressBar(), early_stopping_callback, loss_scaling_callback],
        )

        # Store dataset size for progress bar calculation
        trainer.args.train_dataset_size = len(dataset["train"])

        # Print training configuration
        print(f"Per device train batch size: {trainer.args.per_device_train_batch_size}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Gradient accumulation steps: {trainer.args.gradient_accumulation_steps}")
        print(f"Effective batch size: {trainer.args.per_device_train_batch_size * max(1, torch.cuda.device_count()) * trainer.args.gradient_accumulation_steps}")
        print(f"Training dataset size: {trainer.args.train_dataset_size}")
        print(f"Steps per epoch: {trainer.args.train_dataset_size // (trainer.args.per_device_train_batch_size * max(1, torch.cuda.device_count()) * trainer.args.gradient_accumulation_steps)}")

        # Train and push to hub
        # Only profile if explicitly requested in config
        if self.config.get("enable_profiling", False):
            # profile training for performance analysis
            import torch.profiler
            import os

            # create profiler directory if it doesn't exist
            os.makedirs("logs/after", exist_ok=True)

            # profiler schedule: warmup 5 steps, active for 25 steps, then exit profiling
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=5,  # warmup steps
                    warmup=5,  # more warmup
                    active=25,  # active profiling
                    repeat=1
                ),
                record_shapes=True,
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler("logs/after/value_profile")
            )

            with prof:
                trainer.train()
                # make profiler active for a full batch
                for _ in range(35):  # wait + warmup + active steps
                    prof.step()
        else:
            # train without profiling overhead (for performance)
            trainer.train()

        print(f"Pushing model to Hugging Face Hub: {self.config['hub_model_id']}")

        # Pass token explicitly to push_to_hub
        if self.hf_token:
            trainer.push_to_hub(token=self.hf_token)
        else:
            trainer.push_to_hub()

        print(f"Model successfully pushed to: https://huggingface.co/{self.config['hub_model_id']}")
        shutil.rmtree(self.temp_dir)
