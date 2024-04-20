import torch
from torch.cuda import is_available as cuda_available, is_bf16_supported
from torch.backends.mps import is_available as mps_available
import yaml
import json
import os
import random
from torch import Tensor, argmax
from transformers import AutoModelForCausalLM, BertConfig, TransfoXLConfig, TransfoXLLMHeadModel, Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from evaluate import load as load_metric
from data_loader import JSONDataset
import sys
import argparse
from tqdm import tqdm


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.normpath(r"C:\Users\Keshav\Desktop\QMUL\Research\PhraseBuilder_new\PhraseBuilder\configs\configs_windows.yaml"),
                    help="Path to the config file")
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    configs = yaml.safe_load(f)

batch_size = 32
learning_rate = 0.00025
epochs = 30

# Artifact folder
artifact_folder = configs['raw_data']['artifact_folder']
# Load encoder tokenizer json file dictionary
tokenizer_filepath = os.path.join(artifact_folder, "tokenizer.json")
# Load the tokenizer dictionary
with open(tokenizer_filepath, 'r') as f:
    tokenizer = json.load(f)

    
# Open the train, validation, and test sets json files
with open(os.path.join(artifact_folder, "train_file_list.json"), "r") as f:
    train_file_list = json.load(f)
with open(os.path.join(artifact_folder, "valid_file_list.json"), "r") as f:
    valid_file_list = json.load(f)
with open(os.path.join(artifact_folder, "test_file_list.json"), "r") as f:
    test_file_list = json.load(f)

# Print length of train, validation, and test sets
print("Length of train set: ", len(train_file_list))
print("Length of validation set: ", len(valid_file_list))
print("Length of test set: ", len(test_file_list))

# Load the dataset
train_dataset = JSONDataset(configs, train_file_list, mode="train")
valid_dataset = JSONDataset(configs, valid_file_list, mode="eval")
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Get the vocab size
vocab_size = len(tokenizer)
# Get the phrase length
train_phrase_length = len(train_dataset.file_number_phrase_number)

# Create the model
config_decoder = TransfoXLConfig()
config_decoder.vocab_size = vocab_size
# config_decoder.max_position_embeddings = 4096
# config_decoder.max_length = 4096
config_decoder.pad_token_id = 0
config_decoder.bos_token_id = tokenizer["BOS"]
config_decoder.eos_token_id = tokenizer["EOS"]
config_decoder.num_hidden_layers = 5
config_decoder.num_attention_heads = 4
config_decoder.d_model = 512
config_decoder.d_inner = 2048
config_decoder.d_embed = 512
config_decoder.cutoffs = [0, vocab_size]
config_decoder.div_val = 1
config_decoder.untie_r = True
config_decoder.tie_projs = [False]

print(config_decoder)

# Create the model
model = TransfoXLLMHeadModel(config=config_decoder)

# Print the number of parameters in the model
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in the model: {num_params}")

# Create config for the Trainer
USE_CUDA = cuda_available()
print(f"USE_CUDA: {USE_CUDA}")
if not cuda_available():
    FP16 = FP16_EVAL = BF16 = BF16_EVAL = False
elif is_bf16_supported():
    BF16 = BF16_EVAL = True
    FP16 = FP16_EVAL = False
else:
    BF16 = BF16_EVAL = False
    FP16 = FP16_EVAL = True
USE_MPS = not USE_CUDA and mps_available()


class AccuracyMetricsCallback(TrainerCallback):
    """
    Callback to calculate accuracy metrics during evaluation.
    """
    def __init__(self, val_data):
        self.val_data = val_data

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """
        Calculate accuracy metrics on the validation data during evaluation.
        """
        model.eval()  # Set the model to evaluation mode

        total_tokens = 0
        correct_tokens = 0

        with torch.no_grad():
            
            for inputs in tqdm(self.val_data):
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']

                input_ids = input_ids.unsqueeze(0).to("cuda")

                # Get the model's predictions
                outputs = model(input_ids)
                logits = outputs.logits

                # Calculate the number of correct and total tokens
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()

                # Exclude padding tokens from calculation
                not_pad_mask = shift_labels != 0
                correct_tokens += (shift_logits.argmax(dim=-1) == shift_labels)[not_pad_mask].sum().item()
                total_tokens += not_pad_mask.sum().item()

        accuracy = correct_tokens / total_tokens

        metrics = {
            'accuracy': accuracy,
            'correct_tokens': correct_tokens,
            'total_tokens': total_tokens,
        }

        # Log the metrics to the Trainer's console
        print(f"Validation Accuracy: {metrics['accuracy']}")

        # # Add the metrics to the Trainer's evaluation results
        # control.load_state_dict(metrics)

        return control

# Create your custom callback
accuracy_metrics_callback = AccuracyMetricsCallback(val_data=valid_dataset)

# Define the training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(artifact_folder, "pop_music_transformer"),
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_strategy="steps",  # "steps" or "epoch"
    save_steps=1000,
    save_total_limit=1,
    learning_rate=learning_rate,
    max_steps=int(train_phrase_length//batch_size)*epochs,
    evaluation_strategy="steps",
    eval_steps=1000,
    gradient_accumulation_steps=1,
    warmup_steps=500,
    gradient_checkpointing=False,
    optim="adafactor",
    seed=444,
    logging_strategy="steps",
    logging_steps=100,
    logging_dir=os.path.join(artifact_folder, "pop_music_transformer", "logs"),
    no_cuda=not USE_CUDA,
    fp16=FP16,
    fp16_full_eval=FP16_EVAL,
    bf16=BF16,
    bf16_full_eval=BF16_EVAL,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="tensorboard",
    run_name="pop_music_transformer",
    push_to_hub=False
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    # compute_metrics=compute_metrics,
    # preprocess_logits_for_metrics=preprocess_logits,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] #accuracy_metrics_callback, 
)

# Train and save the model
train_result = trainer.train()
trainer.save_model()  # Saves the tokenizer too
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()
