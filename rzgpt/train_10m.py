import torch
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from itertools import islice
from model_10m import RzGPT,RzGPTConfig
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
import json
import os
from torch.nn import CrossEntropyLoss
import time

from datasets import load_dataset

DATA_PATH =".data/stories.jsonl"
TOKENIZER_PATH = ".data/tokenizer/rzgpt_tokenizer.json"
NUM_STORIES = 55000
DATASET_NAME = 'roneneldan/TinyStories'
CHECKPOINT_DIR = ".data/checkpoints"
CHECKPOINT_PATH = ".data/checkpoints/epoch_35.pt"


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

def get_data():
    
    stories = []

    if os.path.exists(DATA_PATH):

        # Load previously saved stories
        print("Loading stories from local disk...")

        with open(DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                stories.append(json.loads(line))

    else:

        # Download and collect stories only for the first run
        print("Downloading dataset for the first time...")

        dataset_stream = load_dataset(
            DATASET_NAME,
            split="train",
            streaming=True
        )

        dataset_stream = dataset_stream.shuffle(
            seed=42,
            buffer_size=10000
        )

        stories = list(islice(dataset_stream, NUM_STORIES))

        # Save stories locally
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            for story in stories:
                f.write(json.dumps(story, ensure_ascii=False) + "\n")

        print("Stories saved successfully!")

    print("Total stories:", len(stories))
    
    return stories

PAD_ID = 2

def collate_fn(batch):
    
    inputs = []
    outputs = []
    
    for input_ids,output_ids in batch:
        inputs.append(input_ids)
        outputs.append(output_ids)
    
    inputs = pad_sequence(inputs,batch_first=True,padding_value=PAD_ID)
    outputs = pad_sequence(outputs,batch_first=True,padding_value=PAD_ID)
    
    return inputs,outputs
        

def get_chunks(texts):
    MAX_SEQ_LEN = config.max_seq_len
    all_chunks = []
    for story in texts:

        encoded = tokenizer.encode(story)

        token_ids = (
            [tokenizer.token_to_id('[SOS]')]
            + encoded.ids
            + [tokenizer.token_to_id('[EOS]')]
        )


        for i in range(0, len(token_ids)-1, MAX_SEQ_LEN):

            chunk = token_ids[i : i + MAX_SEQ_LEN+1]

            if len(chunk) < 2:
                continue
            
            input_ids = chunk[:-1]
            output_ids = chunk[1:]
            all_chunks.append((input_ids,output_ids))
    
    return all_chunks


config = RzGPTConfig()

model = RzGPT(config=config)


stories = get_data()



train_stories = stories[:50000]
test_stories = stories[50000:]


train_texts = [ item['text'] for item in train_stories]
test_texts = [item['text'] for item in test_stories]


if os.path.exists(TOKENIZER_PATH):

    print("Loading existing tokenizer...")
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    PAD_ID = tokenizer.token_to_id('[PAD]')

else:

    print("Training tokenizer...")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    special_token = ['[SOS]','[EOS]','[PAD]','[UNK]']
    trainer = BpeTrainer(vocab_size=config.vocab_size,min_frequency=2,show_progress=True,special_tokens=special_token)

    tokenizer.train_from_iterator(
        train_texts,
        trainer=trainer,
        length=len(train_texts)
    )

    os.makedirs(
        os.path.dirname(TOKENIZER_PATH),
        exist_ok=True
    )
    PAD_ID = tokenizer.token_to_id('[PAD]')
    tokenizer.save(TOKENIZER_PATH)
    



all_chunks = get_chunks(train_texts)




dataset = RzDataset(all_chunks)

dataloader = DataLoader(dataset=dataset,batch_size=config.batch_size,shuffle=True,collate_fn=collate_fn)


## Trainig on gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device=device)

optimizer = torch.optim.AdamW(model.parameters(),lr=config.lr,weight_decay=config.weight_decay)
criterion = CrossEntropyLoss(ignore_index=PAD_ID)

print("using device :",device)


EPOCH = config.epoch
RESUME = True


START_EPOCH = 0
global_step = 0

# Resume training
if RESUME:

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    START_EPOCH = checkpoint["epoch"]
    global_step = checkpoint["global_step"]

    print("Resuming training...")
    print("Starting epoch:", START_EPOCH + 1)
    print("Global step:", global_step)


# Training loop
for epoch in range(START_EPOCH, EPOCH):
    
    total_loss = 0.0
    model.train() # set train flag
    start_time = time.time()
    
    for idx,(batch_input , batch_target) in enumerate(dataloader,start=1):
        
        # load in GPU
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)
        
        # clear gradient
        optimizer.zero_grad()
        
        # Forward Pass
        logits = model(batch_input)
        
        # Reshape
        logits = logits.reshape(-1,config.vocab_size)
        target = batch_target.reshape(-1)
        
        # calculate Loss
        loss = criterion(logits,target)
        
        # calculate gradient
        loss.backward()
        
        # update weights
        optimizer.step()
        
        #Accumulate loss
        total_loss+=loss.item()
        
        global_step += 1
        
        
        processed = min(
            idx * config.batch_size,
            len(dataloader.dataset)
        )

        print(f"[{processed}/{len(dataloader.dataset)}] "f"Loss: {loss.item():.4f}"
        )
    
    avg_loss = total_loss/(len(dataloader))
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    epoch_time = time.time() - start_time
    checkpoint_path = os.path.join(CHECKPOINT_DIR,f"epoch_{epoch+1}.pt")
    
    save_checkpoint(model=model,optimizer=optimizer,epoch=epoch + 1,batch_idx=idx,global_step=global_step,loss=avg_loss,path=checkpoint_path)
    
    
    
    print(f"Epoch {epoch+1}/{EPOCH}")
    print(f"Avg Loss: {avg_loss:.4f}")
    print(f"Epoch time : {epoch_time:.2f} seconds")
