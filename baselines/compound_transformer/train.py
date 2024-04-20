import torch
from torch.cuda import is_available as cuda_available, is_bf16_supported
from torch.backends.mps import is_available as mps_available
import yaml
import json
import os
import random
from torch import Tensor, argmax
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from evaluate import load as load_metric
from data_loader import JSONDataset, DataLoader
import sys
import argparse
from tqdm import tqdm
from cp_model import LinearAttentionTransformerLM

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.normpath("/homes/kb658/PhraseBuilder/configs/configs_os.yaml"),
                    help="Path to the config file")
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    configs = yaml.safe_load(f)

batch_size = 116
learning_rate = 0.0001
epochs = 30

# Artifact folder
artifact_folder = configs['raw_data']['artifact_folder']
# Load encoder tokenizer json file dictionary
tokenizer_filepath = os.path.join(artifact_folder, "cp_tokenizer.json")
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
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Get the vocab size
vocab_size = [len(tokenizer["time_signature_tokenizer"])+1, len(tokenizer["chord_tokenizer"])+1, len(tokenizer["metric_tokenizer"])+1, len(tokenizer["family_tokenizer"])+1, len(tokenizer["pitch_tokenizer"])+1, len(tokenizer["duration_tokenizer"])+1, len(tokenizer["velocity_tokenizer"])+1]
print("Vocab size: ", vocab_size)

# Get the phrase length
train_phrase_length = len(train_dataset.file_number_phrase_number)

model = LinearAttentionTransformerLM(
    num_tokens = vocab_size,
    dim = 512,
    heads = 4,
    depth = 4,
    max_seq_len = 2048,
    causal = True,                  # auto-regressive or not
    ff_dropout = 0.1,               # dropout for feedforward
    attn_layer_dropout = 0.1,       # dropout right after self-attention layer
    attn_dropout = 0.1,             # dropout post-attention
    emb_dim = 512,                  # embedding factorization, to save on memory
    dim_head = 128,                 # be able to fix the dimension of each head, making it independent of the embedding dimension and the number of heads
    blindspot_size = 64,            # this gives the q(kv) attention a blindspot of 64 tokens back in the causal case, but gives back an order of magnitude return in memory savings. should be paired with local attention of at least a window size of this setting. setting this to 1 will allow for full q(kv) attention of past
    n_local_attn_heads = 4,         # number of local attention heads for (qk)v attention. this can be a tuple specifying the exact number of local attention heads at that depth
    local_attn_window_size = 128,   # receptive field of the local attention
    reversible = True,              # use reversible nets, from Reformer paper
    ff_chunks = 2,                  # feedforward chunking, from Reformer paper
    ff_glu = True,                  # use GLU variant for feedforward
    attend_axially = False,         # will fold the sequence by the local attention window size, and do an extra strided attention followed by a feedforward with the cheap q(kv) attention
    shift_tokens = True             # add single token shifting, for great improved convergence
    )

# Print the number of parameters in the model
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in the model: {num_params}")

# Create config for the Trainer
USE_CUDA = cuda_available()
print(f"USE_CUDA: {USE_CUDA}")
if not cuda_available():
    FP16 = FP16_EVAL = BF16 = BF16_EVAL = False
elif is_bf16_supported():
    BF16 = BF16_EVAL = False #True
    FP16 = FP16_EVAL = False
else:
    BF16 = BF16_EVAL = False
    FP16 = FP16_EVAL = True
USE_MPS = not USE_CUDA and mps_available()

loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)
def calc_loss(predict, target):
    loss = loss_func(predict, target)
    return loss

# subclass trainer
class CustomTrainer(Trainer):    
    def compute_loss(self, model, inputs, return_outputs=False):
        x = inputs.pop("x")
        target = inputs.pop("y")
        # loss_mask = inputs.pop("loss_mask")

        y_tempo, y_chord, y_barbeat, y_type, y_pitch, y_duration, y_velocity = model(x, target, is_training=True)
        # reshape (b, s, f) -> (b, f, s)
        y_tempo     = y_tempo[:, ...].permute(0, 2, 1)
        y_chord     = y_chord[:, ...].permute(0, 2, 1)
        y_barbeat   = y_barbeat[:, ...].permute(0, 2, 1)
        y_type      = y_type[:, ...].permute(0, 2, 1)
        y_pitch     = y_pitch[:, ...].permute(0, 2, 1)
        y_duration  = y_duration[:, ...].permute(0, 2, 1)
        y_velocity  = y_velocity[:, ...].permute(0, 2, 1)

        # loss
        loss_tempo = calc_loss(
                y_tempo, target[..., 0])
        loss_chord = calc_loss(
                y_chord, target[..., 1])
        loss_barbeat = calc_loss(
                y_barbeat, target[..., 2])
        loss_type = calc_loss(
                y_type,  target[..., 3])
        loss_pitch = calc_loss(
                y_pitch, target[..., 4])
        loss_duration = calc_loss(
                y_duration, target[..., 5])
        loss_velocity = calc_loss(
                y_velocity, target[..., 6])

        loss = loss_tempo + loss_chord + loss_barbeat + loss_type + loss_pitch + loss_duration + loss_velocity
        loss = loss / 7
        # Convert to BFLoat16
        loss = loss.to(torch.bfloat16)
        outputs = (y_tempo, y_chord, y_barbeat, y_type, y_pitch, y_duration, y_velocity)

        return (loss, outputs) if return_outputs else loss

# Define the training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(artifact_folder, "cp_transformer"),
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
    gradient_checkpointing=False,
    max_grad_norm=3.0,
    optim="adafactor",
    seed=444,
    logging_strategy="steps",
    logging_steps=100,
    logging_dir=os.path.join(artifact_folder, "cp_transformer", "logs"),
    no_cuda=not USE_CUDA,
    fp16=FP16,
    fp16_full_eval=FP16_EVAL,
    bf16=BF16,
    bf16_full_eval=BF16_EVAL,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="tensorboard",
    run_name="cp_transformer",
    push_to_hub=False,
    label_names=["y"],
)

# Define the Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] #accuracy_metrics_callback, 
)

# Train and save the model
train_result = trainer.train()
trainer.save_model()  # Saves the tokenizer too
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()