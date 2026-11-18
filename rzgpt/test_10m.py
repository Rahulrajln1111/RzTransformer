
import torch
from torch.nn import CrossEntropyLoss
from model_10m import RzGPT, RzGPTConfig
from tokenizers import Tokenizer
import os
# from train_10m import test_texts,get_chunks,DataLoader,RzDataset,collate_fn,CHECKPOINT_PATH

SFT_CHECKPOINT_DIR = ".data/checkpoints/instruction"
CHECKPOINT_PATH = os.path.join(SFT_CHECKPOINT_DIR,"best.pt")


TOKENIZER_PATH = ".data/tokenizer/rzgpt_tokenizer.json"
RESPONSE_DELIM = "Response:\n"

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.95
REPETITION_PENALTY = 1.1



device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


print("Loading tokenizer...")

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

SOS_ID = tokenizer.token_to_id("[SOS]")
EOS_ID = tokenizer.token_to_id("[EOS]")
PAD_ID = tokenizer.token_to_id("[PAD]")

print("Tokenizer loaded!")
print("Vocabulary size:", tokenizer.get_vocab_size())



print("Loading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)

print("Checkpoint loaded!")
print("Training epoch:", checkpoint["epoch"])
print("Global step:", checkpoint["global_step"])
print("Validation loss:", checkpoint["loss"])


# Use the exact configuration saved during training.
saved_config = checkpoint["config"]

config = RzGPTConfig()

for key, value in saved_config.items():
    setattr(config, key, value)



print("Creating RzGPT model...")

model = RzGPT(config=config).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

# Disable dropout and activate evaluation mode
model.eval()

print("RzGPT loaded successfully!")


def format_prompt(instruction, context=""):

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

    return prompt
# test on test data
# def train_test():
#     test_chunks = get_chunks(test_texts)

#     dataset = RzDataset(test_chunks)

#     test_dataloader = DataLoader(dataset=dataset,batch_size=config.batch_size,shuffle=False,collate_fn=collate_fn)


#     criterion = CrossEntropyLoss(
#         ignore_index=PAD_ID
#     )

#     total_loss = 0.0
#     total_tokens = 0

#     with torch.inference_mode():

#         for batch_input, batch_target in test_dataloader:

#             batch_input = batch_input.to(device)
#             batch_target = batch_target.to(device)

#             logits = model(batch_input)

#             logits = logits.reshape(
#                 -1,
#                 config.vocab_size
#             )

#             targets = batch_target.reshape(-1)

#             # Sum loss instead of averaging each batch
#             loss_sum = torch.nn.functional.cross_entropy(
#                 logits,
#                 targets,
#                 ignore_index=PAD_ID,
#                 reduction="sum"
#             )

#             valid_tokens = (targets != PAD_ID).sum()

#             total_loss += loss_sum.item()
#             total_tokens += valid_tokens.item()


#     avg_val_loss = total_loss / total_tokens

#     perplexity = torch.exp(
#         torch.tensor(avg_val_loss)
#     ).item()

#     print("Validation Loss:", avg_val_loss)
#     print("Validation Perplexity:", perplexity)




def top_k_sampling(logits, top_k=50):

    # Keep the top-k highest logits
    values, indices = torch.topk(
        logits,
        k=min(top_k, logits.size(-1)),
        dim=-1
    )

    # Create filtered logits
    filtered_logits = torch.full_like(
        logits,
        float("-inf")
    )

    # Restore only the top-k logits
    filtered_logits.scatter_(
        dim=-1,
        index=indices,
        src=values
    )

    return filtered_logits


def top_p_sampling(logits, top_p=0.95):

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
        dim=-1
    )

    probs = torch.softmax(sorted_logits, dim=-1)

    cumulative = torch.cumsum(probs, dim=-1)

    # Drop tokens that fall outside the nucleus
    drop = (cumulative - probs) > top_p

    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))

    filtered_logits = torch.full_like(logits, float("-inf"))

    filtered_logits.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_logits
    )

    return filtered_logits


@torch.inference_mode()
def generate_text(
    prompt,
    context="",
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1
):

    model.eval()

    # Build the prompt exactly like SFT did: head and delimiter tokenized
    # separately, so the ids match what the model was trained on.
    prompt_text = format_prompt(prompt, context)

    head, _ = prompt_text.rsplit(RESPONSE_DELIM, 1)

    prompt_ids = (
        tokenizer.encode(head).ids
        + tokenizer.encode(RESPONSE_DELIM).ids
    )

    # Keep the instruction tail + delimiter if the prompt is too long
    room = config.max_seq_len - 1

    if len(prompt_ids) > room:
        prompt_ids = prompt_ids[-room:]

    # Single [SOS], matching training
    generated = torch.tensor(
        [[SOS_ID] + prompt_ids],
        dtype=torch.long,
        device=device
    )

    prompt_len = generated.size(1)

    for _ in range(max_new_tokens):

        # Keep only the latest context window
        window = generated[:, -config.max_seq_len:]

        # Forward pass
        logits = model(window)[:, -1, :]

        # Apply temperature
        logits = logits / max(temperature, 1e-6)

        # Discourage repeating tokens already generated in the response
        if repetition_penalty and repetition_penalty != 1.0:

            recent = generated[0, prompt_len:].tolist()

            if recent:

                idx = torch.tensor(
                    sorted(set(recent)),
                    device=logits.device
                )

                values = logits[0, idx]

                logits[0, idx] = torch.where(
                    values < 0,
                    values * repetition_penalty,
                    values / repetition_penalty
                )

        # Apply top-k filtering
        if top_k is not None:
            logits = top_k_sampling(logits, top_k)

        # Apply top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            logits = top_p_sampling(logits, top_p)

        # Convert logits into probabilities
        probabilities = torch.softmax(logits, dim=-1)

        # Sample one token
        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        # Stop at EOS
        if next_token.item() == EOS_ID:
            break

        # Append predicted token
        generated = torch.cat(
            [generated, next_token],
            dim=1
        )

    # Decode only the newly generated response, not the prompt echo
    new_ids = generated[0, prompt_len:].tolist()

    return tokenizer.decode(new_ids).strip()



## Check loss on test data
# train_test()

print("\n" + "=" * 50)
print("           RzGPT TEXT GENERATOR")
print("=" * 50)

while True:

    prompt = input("\nEnter prompt (or 'exit'): ")

    if prompt.lower() == "exit":
        print("Exiting RzGPT...")
        break

    if not prompt.strip():
        print("Please enter a prompt.")
        continue

    result = generate_text(
        prompt=prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY
    )

    print("\nGenerated text:")
    print("-" * 50)
    print(result)
    print("-" * 50)
