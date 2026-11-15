import torch.nn as nn
import torch

import torch.nn.functional as F

import math

class RzGPTConfig():
    def __init__(self):
        self.vocab_size = 8000
        self.num_head = 8
        self.num_layer = 6
        self.d_model = 320
        self.dropout = 0.1
        self.dff = 1280
        self.eps = 1e-5  
        self.max_seq_len = 256
        self.batch_size = 64
        self.lr = 3e-4
        self.weight_decay = 0.01
        self.epoch = 35
        
config=RzGPTConfig()

class TokenEmbedding(nn.Module):
    def __init__(self,config:RzGPTConfig):
        super().__init__()
        
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        
        self.embedding = nn.Embedding(config.vocab_size,config.d_model)
        
    def forward(self,input_ids:torch.Tensor):
        
        return self.embedding(input_ids)
    
    
class LearnedPositionalEncoding(nn.Module):
    def __init__(self,config:RzGPTConfig):
        super().__init__()
        
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len
        
        self.pos_embedding = nn.Embedding(config.max_seq_len,config.d_model)
        
    
    def forward(self,input_ids:torch.Tensor):
        _,seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len , "Input seq len is more that configured"
        position = torch.arange(0,seq_len,device=input_ids.device)
        
        return self.pos_embedding(position)


class LayerNormalization(nn.Module):
    def __init__(self,config:RzGPTConfig):
        super().__init__()
        
        self.norm = nn.LayerNorm(normalized_shape=config.d_model,eps=config.eps)
        
    def forward(self,x:torch.Tensor):
        
        return self.norm(x)
        

class CausalAttention(nn.Module):
    def __init__(self,config:RzGPTConfig):
        super().__init__()
        
        self.d_model = config.d_model
        self.num_head = config.num_head
        self.d_k = self.d_model//self.num_head
        
        self.qkv = nn.Linear(self.d_model,3*self.d_model)
        
        self.projection = nn.Linear(self.d_model,self.d_model)
        self.att_dropout = config.dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self,x:torch.Tensor):
        batch_sz,seq_len,_ = x.shape
        
        
        qkv = self.qkv(x) # [B,T,3*D]
        
        qkv = qkv.view(batch_sz,seq_len,3,self.num_head,self.d_k) #[B,T,3,H,dk]
        
        qkv = qkv.permute(2,0,3,1,4) #[3,B,H,T,dk]
        
        q,k,v = qkv.unbind(dim=0) #[B,H,T,dk]
        
        context = F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=(self.att_dropout if self.training else 0.0)) # [B,H,T,dk]
        
        context = context.transpose(1,2).contiguous() #[B,T,H,dk]
        
        context = context.view(batch_sz,seq_len,self.d_model) #[B,T,D]
        
        output = self.projection(context)
        
        return self.dropout(output)


class FeedForward(nn.Module):
    def __init__(self,config:RzGPTConfig):
        super().__init__()
        
        self.fc1 = nn.Linear(config.d_model,config.dff)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(config.dff,config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self,x:torch.Tensor):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        
        return self.dropout(x)

class DecoderBlock(nn.Module):
    def __init__(self,config:RzGPTConfig):
        super().__init__()
        
        self.ln1 = LayerNormalization(config=config)
        self.ln2 = LayerNormalization(config=config)
        self.feed_forward = FeedForward(config=config)
        self.attention = CausalAttention(config=config)
        
    def forward(self,x:torch.Tensor):
        x = x+self.ln1(self.attention(x))
 
        x = x+self.ln2(self.feed_forward(x))
        
        return x

class RzGPT(nn.Module):
    def __init__(self,config:RzGPTConfig):
        super().__init__()
        
        self.config = config
        
        self.token_embedding = TokenEmbedding(config=config)
        self.pos_embedding = LearnedPositionalEncoding(config=config)
        self.n_dblock = nn.ModuleList([DecoderBlock(config=config) for _ in range(config.num_layer)])
        self.final_norm = LayerNormalization(config=config)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.d_model,config.vocab_size,bias=False)
        
        
        # recursively initialize all weight based on std = 0.02
        self.apply(self._init_weights) 
        
        
        
        # now initialize two residual connection output layer with lower std
        for block in self.n_dblock:
                    nn.init.normal_(
                        block.attention.projection.weight,
                        mean=0.0,
                        std=0.02 / math.sqrt(2 * config.num_layer)
                    )
        
                    nn.init.normal_(
                        block.feed_forward.fc2.weight,
                        mean=0.0,
                        std=0.02 / math.sqrt(2 * config.num_layer)
                    ) 
        #weight tie with embdeding after initialization
        self.lm_head.weight = self.token_embedding.embedding.weight
             
        
    @staticmethod
    def _init_weights(module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            
    def forward(self, input_ids: torch.Tensor):

        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError("Input exceeds maximum context length")

        # [B, T] -> [B, T, D]
        token_embeddings = self.token_embedding(input_ids)

        # [T, D]
        position_embeddings = self.pos_embedding(input_ids)

        # [B, T, D]
        x = token_embeddings + position_embeddings

        x = self.embedding_dropout(x)

        # pass embedding to transformer encoder blocks
        
        for block in self.n_dblock:
            x = block(x)

        x = self.final_norm(x)

        # [B, T, D] -> [B, T, V]
        logits = self.lm_head(x)

        return logits
        
                
        