import torch
from torch.cuda import is_available as cuda_available, is_bf16_supported
from torch.backends.mps import is_available as mps_available
import yaml
import json
import os
import random
from torch import Tensor, argmax
from transformers import EncoderDecoderModel, EncoderDecoderConfig, BertConfig, Trainer, TrainingArguments, EarlyStoppingCallback
from evaluate import load as load_metric
from data_loader import JSONDataset
import sys
import argparse


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.normpath("/homes/kb658/yinyang/configs/configs_os.yaml"),
                    help="Path to the config file")
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    configs = yaml.safe_load(f)

batch_size = configs['training']['phrase_generation']['batch_size']
learning_rate = configs['training']['phrase_generation']['learning_rate']
epochs = configs['training']['phrase_generation']['epochs']

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

# Create the encoder-decoder model
config_encoder = BertConfig()
config_encoder.vocab_size = vocab_size
config_encoder.max_position_embeddings = configs['model']['phrase_generation_model']['encoder_max_sequence_length']
config_encoder.max_length = configs['model']['phrase_generation_model']['encoder_max_sequence_length']
config_encoder.pad_token_id = 0
config_encoder.bos_token_id = tokenizer["BOS"]
config_encoder.eos_token_id = tokenizer["EOS"]
config_encoder.num_hidden_layers = configs['model']['phrase_generation_model']['num_layers']
config_encoder.num_attention_heads = configs['model']['phrase_generation_model']['num_heads']
config_encoder.hidden_size = configs['model']['phrase_generation_model']['hidden_size']
config_encoder.intermediate_size = configs['model']['phrase_generation_model']['intermediate_size']

config_decoder = BertConfig()
config_decoder.vocab_size = vocab_size
config_decoder.max_position_embeddings = configs['model']['phrase_generation_model']['decoder_max_sequence_length']
config_decoder.max_length = configs['model']['phrase_generation_model']['decoder_max_sequence_length']
config_decoder.bos_token_id = tokenizer["BOS"]
config_decoder.eos_token_id = tokenizer["EOS"]
config_decoder.pad_token_id = 0
config_decoder.num_hidden_layers = configs['model']['phrase_generation_model']['num_layers']
config_decoder.num_attention_heads = configs['model']['phrase_generation_model']['num_heads']
config_decoder.hidden_size = configs['model']['phrase_generation_model']['hidden_size']
config_decoder.intermediate_size = configs['model']['phrase_generation_model']['intermediate_size']

# set decoder config to causal lm
config_decoder.is_decoder = True
config_decoder.add_cross_attention = True
config_decoder.tie_encoder_decoder = False
config_decoder.tie_word_embeddings = False

config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
model = EncoderDecoderModel(config=config)
# config.max_length = configs['model']['phrase_generation_model']['max_sequence_length']
config.decoder_start_token_id = tokenizer["BOS"]
config.pad_token_id = 0

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

metrics = {metric: load_metric(metric) for metric in ["accuracy"]}

def compute_metrics(eval_pred):
    """
    Compute metrics for pretraining.

    Must use preprocess_logits function that converts logits to predictions (argmax or sampling).

    :param eval_pred: EvalPrediction containing predictions and labels
    :return: metrics
    """
    predictions, labels = eval_pred
    not_pad_mask = labels != 0
    labels, predictions = labels[not_pad_mask], predictions[not_pad_mask]
    return metrics["accuracy"].compute(predictions=predictions.flatten(), references=labels.flatten())

def preprocess_logits(logits: Tensor, _: Tensor) -> Tensor:
    """
    Preprocess the logits before accumulating them during evaluation.

    This allows to significantly reduce the memory usage and make the training tractable.
    """
    pred_ids = argmax(logits[0], dim=-1)  # long dtype
    return pred_ids

# Define the training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(artifact_folder, "phrase_generation"),
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_strategy="steps",  # "steps" or "epoch"
    save_steps=1000,
    save_total_limit=1,
    learning_rate=learning_rate,
    weight_decay= configs['training']['phrase_generation']['weight_decay'],
    max_grad_norm=configs['training']['phrase_generation']['max_grad_norm'],
    max_steps=int(train_phrase_length//batch_size)*epochs,
    evaluation_strategy="steps",
    eval_steps=1000,
    gradient_accumulation_steps=configs['training']['phrase_generation']['gradient_accumulation_steps'],
    gradient_checkpointing=True,
    optim="adafactor",
    seed=444,
    logging_strategy="steps",
    logging_steps=100,
    logging_dir=os.path.join(artifact_folder, "phrase_generation", "logs"),
    no_cuda=not USE_CUDA,
    fp16=FP16,
    fp16_full_eval=FP16_EVAL,
    bf16=BF16,
    bf16_full_eval=BF16_EVAL,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="tensorboard",
    run_name="phrase_generation",
    push_to_hub=False
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train and save the model
train_result = trainer.train()
trainer.save_model()  # Saves the tokenizer too
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()
