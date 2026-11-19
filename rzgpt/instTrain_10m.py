import torch
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from model_10m import RzGPT,RzGPTConfig
from tokenizers import Tokenizer
import json
import os
from torch.nn import CrossEntropyLoss
import time
import random
import math

from collections import Counter
from datasets import load_dataset
config = RzGPTConfig()
DATA_PATH =".data/instructionData.jsonl"
TOKENIZER_PATH = ".data/tokenizer/rzgpt_tokenizer.json"
NUM_EXAMPLES = 14500
DATASET_NAME = 'databricks/databricks-dolly-15k'
CHECKPOINT_DIR = ".data/checkpoints"


MAX_SEQ_LEN = config.max_seq_len
RESPONSE_DELIM = "Response:\n"

global_step = 0
SAVE_EVERY = 500

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class RzDataset(Dataset):
    def __init__(self,chunks):
        super().__init__()
        self.chunks = chunks
    
    def __len__(self):
        return len(self.chunks)
    def __getitem__(self, index):
        input_ids,output_ids = self.chunks[index]
        
        return (torch.tensor(input_ids,dtype=torch.long),torch.tensor(output_ids,dtype=torch.long))
        

def save_checkpoint(model,optimizer,epoch,batch_idx,global_step,loss,path):
    checkpoint = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": vars(config)
    }

    torch.save(checkpoint, path)

    print(f"Checkpoint saved: {path}")

def collate_fn(batch):
    
    input_ids = []
    output_ids = []
    
    for input_id,output_id in batch:
        input_ids.append(input_id)
        output_ids.append(output_id)
        
    
    inputs = pad_sequence(input_ids,batch_first=True,padding_value=PAD_ID)
    outputs = pad_sequence(output_ids,batch_first=True,padding_value=-100)
    
    return inputs,outputs

def get_dataset():

    print("Loading Dolly instruction dataset...")

    instruction_data = []

    if os.path.exists(DATA_PATH):

        with open(DATA_PATH, 'r', encoding='utf-8') as f:

            for line in f:
                instruction_data.append(json.loads(line))

        print("Instruction dataset loaded successfully!")

        return instruction_data

    dataset = load_dataset(
        DATASET_NAME,
        split="train"
    )

    print("Dataset loaded successfully!")

    print("Total examples:", len(dataset))

    print("\nDataset columns:")
    print(dataset.column_names)

    print("\nFirst example:")
    print(dataset[0])

    dataset = dataset.shuffle(seed=42)

    dataset = dataset.select(
        range(min(NUM_EXAMPLES, len(dataset)))
    )

    for item in dataset:

        instruction = item["instruction"].strip()
        context = item["context"].strip()
        response = item["response"].strip()

        # Skip empty examples
        if not instruction or not response:
            continue

        example = {
            "instruction": instruction,
            "context": context,
            "response": response,
            "category": item["category"]
        }

        instruction_data.append(example)

    os.makedirs(
        os.path.dirname(DATA_PATH),
        exist_ok=True
    )

    with open(DATA_PATH, "w", encoding="utf-8") as f:

        for item in instruction_data:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                ) + "\n"
            )

    print("\nDataset preparation completed!")

    print("Total saved examples:", len(instruction_data))

    print("Saved path:", DATA_PATH)

    return instruction_data


def format_instruction(item):

    instruction = item["instruction"]

    context = item["context"]

    response = item["response"]

    if context:

        prompt = (

            "Instruction:\n"

            + instruction

            + "\n\n"

            + "Context:\n"

            + context

            + "\n\n"

            + "Response:\n"
        )

    else:

        prompt = (

            "Instruction:\n"

            + instruction

            + "\n\n"

            + "Response:\n"
        )

    return prompt, response




def prepare_instruction_with_limit(item, max_len=MAX_SEQ_LEN):

 
    prompt, response = format_instruction(item)
 
    # split so the delimiter is tokenized separately and can never be cut
    head, _ = prompt.rsplit(RESPONSE_DELIM, 1)
 
    head_ids = tokenizer.encode(head).ids
    delim_ids = tokenizer.encode(RESPONSE_DELIM).ids
    response_ids = tokenizer.encode(response).ids
 
    # room for prompt head + response, after [SOS], delimiter and [EOS]
    budget = max_len - 2 - len(delim_ids)
 
    if budget < 2:
        return None
 
    # if it does not fit, the response still gets at least half the budget
    keep_response = min(
        len(response_ids),
        max(budget - len(head_ids), budget // 2)
    )
 
    keep_head = budget - keep_response
 
    if keep_head < 1:
        keep_head = 1
        keep_response = budget - 1
 
    prompt_ids = head_ids[-keep_head:] + delim_ids
    response_ids = response_ids[:keep_response]
 
    full_ids = [SOS_ID] + prompt_ids + response_ids + [EOS_ID]
 
    prompt_length = len(prompt_ids) + 1
 
    input_ids = full_ids[:-1]
    labels = full_ids[1:]
 
    # supervise only the response (+EOS), mask SOS/prompt/delimiter
    labels[:prompt_length - 1] = [-100] * (prompt_length - 1)
 
    return input_ids, labels


def validate_model():

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.inference_mode():

        for batch_input, batch_target in val_loader:

            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            logits = model(batch_input)

            logits = logits.reshape(-1, config.vocab_size)
            targets = batch_target.reshape(-1)

            loss = criterion(logits, targets)

            valid_tokens = (targets != -100).sum()

            total_loss += loss.item()
            total_tokens += valid_tokens.item()

    avg_val_loss = total_loss / total_tokens

    perplexity = torch.exp(
        torch.tensor(avg_val_loss)
    ).item()

    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Perplexity: {perplexity:.4f}")

    model.train()

    return avg_val_loss

dataset = get_dataset()


category_count = Counter(item["category"] for item in dataset)

print("\nCategory distribution:")

for category, count in category_count.items():

    print(f"{category}: {count}")



empty_context = 0
non_empty_context = 0

for item in dataset:

    if item["context"].strip():
        non_empty_context += 1
    else:
        empty_context += 1


print("\nContext distribution:")

print("Empty context:", empty_context)
print("Non-empty context:", non_empty_context)

print(
    "Context percentage:",
    round(non_empty_context / len(dataset) * 100, 2),
    "%"
)


tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
 
SOS_ID = tokenizer.token_to_id("[SOS]")
EOS_ID = tokenizer.token_to_id("[EOS]")
PAD_ID = tokenizer.token_to_id("[PAD]")
UNK_ID = tokenizer.token_to_id("[UNK]")
 
print("\nTokenizer loaded successfully!")
print("Vocabulary size:", tokenizer.get_vocab_size())


chunks = []
skipped_examples = 0

for item in dataset:
    
    result  = prepare_instruction_with_limit(item)
    
    if result is None:
        skipped_examples+=1
        continue
    
    input_ids , output_ids = result
    
    chunks.append((input_ids,output_ids))
    
print("\nInstruction preparation completed!")

print("Total original examples:", len(dataset))

print("Prepared examples:", len(chunks))

print("Skipped examples:", skipped_examples)

# divide into val and test

random_generator = random.Random(42)

random_generator.shuffle(chunks)

split_index = int(len(chunks) * 0.9)

train_chunks = chunks[:split_index]

val_chunks = chunks[split_index:]


print("\nDataset split:")

print("Total examples:", len(chunks))

print("Training examples:", len(train_chunks))

print("Validation examples:", len(val_chunks))



train_dataset = RzDataset(train_chunks)

val_dataset = RzDataset(val_chunks)


print("\nTraining dataset created:", len(train_dataset))

print("Validation dataset created:", len(val_dataset))

print("\nRzDataset created successfully!")

print("Total training examples:", len(train_dataset))

train_loader = DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True,collate_fn=collate_fn,drop_last=False,num_workers=4,pin_memory=True,persistent_workers=True)
val_loader = DataLoader(val_dataset,batch_size=config.batch_size,shuffle=False,collate_fn=collate_fn,drop_last=False,num_workers=4,pin_memory=True,persistent_workers=True)




print("\nDataLoader created successfully!")

print("Training batches:", len(train_loader))

print("Validation batches:", len(val_loader))

sample_inputs, sample_labels = next(iter(train_loader))

print("\nSample batch:")

print("Input shape:", sample_inputs.shape)

print("Label shape:", sample_labels.shape)

print("Batch size:", train_loader.batch_size)


# Now Training Part

SFT_EPOCHS = 10

LEARNING_RATE = config.lr

WEIGHT_DECAY = config.weight_decay

SFT_CHECKPOINT_DIR = ".data/checkpoints/instruction"

os.makedirs(SFT_CHECKPOINT_DIR, exist_ok=True)

BASE_CHECKPOINT = ".data/checkpoints/base_10m.pt"   # pretrained, never written by SFT
SFT_LATEST = os.path.join(SFT_CHECKPOINT_DIR, "latest.pt")
SFT_BEST = os.path.join(SFT_CHECKPOINT_DIR, "best.pt")
 
# Opt-in only: run with RESUME_SFT=1 to continue from instruction/latest.pt
RESUME_SFT = os.environ.get("RESUME_SFT", "0") == "1" and os.path.exists(SFT_LATEST)



device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nUsing device:", device)

load_path = SFT_LATEST if RESUME_SFT else BASE_CHECKPOINT

print("Loading checkpoint:", load_path)

checkpoint = torch.load(load_path,map_location=device,weights_only=True)

saved_config = checkpoint['config']

for key,value in saved_config.items():
    setattr(config,key,value)
    
model = RzGPT(config=config).to(device)

model.load_state_dict(checkpoint['model_state_dict'])


SFT_LEARNING_RATE = 1e-5

# No weight decay on LayerNorm / biases / embeddings (params with ndim < 2)
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():

    if not param.requires_grad:
        continue

    if param.ndim < 2:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=SFT_LEARNING_RATE,
)

STEPS_PER_EPOCH = len(train_loader)
TOTAL_STEPS = SFT_EPOCHS * STEPS_PER_EPOCH
WARMUP_STEPS = max(1, int(0.03 * TOTAL_STEPS))


def lr_lambda(step):

    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS

    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)

    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

criterion = CrossEntropyLoss(ignore_index=-100,reduction="sum")


START_EPOCH = 0
best_val_loss = float("inf")

if RESUME_SFT:

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    START_EPOCH = checkpoint['epoch']
    global_step = checkpoint['global_step']
    best_val_loss = checkpoint['loss']

    # Continue the LR schedule from where it stopped
    scheduler.last_epoch = global_step

    print("Resuming SFT at epoch:", START_EPOCH + 1)
    print("Global step:", global_step)

if START_EPOCH >= SFT_EPOCHS:
    print("WARNING: START_EPOCH >= SFT_EPOCHS - nothing to train. Raise SFT_EPOCHS.")

for epoch in range(START_EPOCH, SFT_EPOCHS):

    model.train()

    total_train_loss = 0.0
    total_train_tokens = 0

    start_time = time.time()

    for batch_idx, (batch_input, batch_target) in enumerate(train_loader):

        batch_input = batch_input.to(device, non_blocking=True)
        batch_target = batch_target.to(device, non_blocking=True)

        # Clear gradients from previous batch
        optimizer.zero_grad(set_to_none=True)

        # Forward pass in bf16 on GPU
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda")
        ):
            logits = model(batch_input)

        # Reshape logits and targets for CrossEntropyLoss
        logits = logits.reshape(-1, config.vocab_size).float()
        targets = batch_target.reshape(-1)

        # Calculate summed loss over valid response tokens
        loss = criterion(logits, targets)

        # Count valid response tokens
        valid_tokens = (targets != -100).sum()

        # Average loss per valid token
        avg_loss = loss / valid_tokens

        # Backpropagation
        avg_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        # Update model parameters
        optimizer.step()
        scheduler.step()

        # Accumulate epoch statistics
        total_train_loss += loss.item()
        total_train_tokens += valid_tokens.item()

        global_step += 1

        if (batch_idx + 1) % 50 == 0:

            print(
                f"Epoch [{epoch+1}/{SFT_EPOCHS}] "
                f"Batch [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {avg_loss.item():.4f} "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    # Calculate average training loss per response token
    avg_train_loss = total_train_loss / total_train_tokens

    # Validate after every epoch
    avg_val_loss = validate_model()

    epoch_time = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"Epoch: {epoch+1}/{SFT_EPOCHS}")
    print(f"Training Loss: {avg_train_loss:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Time: {epoch_time:.2f} seconds")
    print("=" * 60)

    # Save latest checkpoint
    save_checkpoint(
        model,
        optimizer,
        epoch + 1,
        len(train_loader),
        global_step,
        avg_val_loss,
        SFT_LATEST
    )

    # Save best checkpoint
    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            len(train_loader),
            global_step,
            avg_val_loss,
            SFT_BEST
        )

        print("Best model checkpoint saved!")

print("\nInstruction fine-tuning completed!")