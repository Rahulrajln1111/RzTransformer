import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputEmbedding(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
    
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,seq_len:int,dropout:float)->None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(seq_len,d_model)
        
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_trm = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000)/d_model))
        pe[:,::2] = torch.sin(position*div_trm) 
        pe[:,1::2] = torch.cos(position*div_trm)
        
        pe = pe.unsqueeze(0)
        
        # no need to learn pe
        self.register_buffer('pe',pe)
        
    
    def forward(self,x):
        x = x + self.pe[:,:x.shape[1],:]
        
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self,d_model:int = 512,eps:float = 10**-6):
        super().__init__()
        
        self.eps = eps
        self.gama = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
    
    def forward(self,x):
#          mean = x.mean(dim=-1,keepdim=True)
#          variance = x.var(dim=-1,keepdim=True,unbiased=False)
#           
#          norm = (x-mean)/torch.sqrt(variance+self.eps)
#         
#          return self.gama*norm+self.beta
        # F.layer_norm maps to the fused kernel; the manual
        # mean/var/sqrt version is several extra elementwise passes.
        return F.layer_norm(x,(x.shape[-1],),self.gama,self.beta,self.eps)
    

class FeedForwardBlock(nn.Module):
    def __init__(self,d_model:int,d_dff:int , dropout:float):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model,d_dff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 =  nn.Linear(d_dff,d_model)
        
    def forward(self,x):
        
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,num_head:int,dropout:float):
        super().__init__()
        
        self.d_model = d_model
        
        assert d_model%num_head == 0 , "d_model is not in multiple of num_head"
        
        self.d_k = d_model//num_head
        self.num_head = num_head
        
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
# -    @staticmethod
# -    def attention(q,k,v,mask,dropout:nn.Dropout):
# -        d_k = q.shape[-1]
# -        # similarity score across each head
# -        att_scores = torch.matmul(q,k.transpose(-1,-2))/math.sqrt(d_k)
# -        if mask is not None:
# -            att_scores = att_scores.masked_fill(mask==0,float('-inf'))
# -        
# -        att_scores = att_scores.softmax(dim=-1)
# -        
# -        if dropout is not None:
# -            att_scores = dropout(att_scores)
# -                    
# -        context  = torch.matmul(att_scores,v)
# -        
# -        return context,att_scores
        
    def forward(self,q,k,v,mask):
        batch_sz, seq_len , _ = q.shape
        
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)
        
        Q = Q.view(batch_sz,q.shape[1],self.num_head,self.d_k).transpose(1,2)
        K = K.view(batch_sz,k.shape[1],self.num_head,self.d_k).transpose(1,2)
        V = V.view(batch_sz,v.shape[1],self.num_head,self.d_k).transpose(1,2)
        
        # Fused attention kernel (FlashAttention / mem-efficient backend on GPU).
        # bool mask: True = token is allowed to attend, same as masked_fill(mask==0,-inf).
        context = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        
        context = context.transpose(1,2).contiguous()
        
        context = context.reshape(batch_sz,seq_len,self.d_model)
        
        output = self.w_o(context)
        
        return output
    

class ResidualConnection(nn.Module):
    def __init__(self,dropout:float,d_model:int):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)
        
    def forward(self,x,sublayer):
        return self.norm(x+self.dropout(sublayer(x)))
    
    
class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttention,feed_forward_block:FeedForwardBlock,dropout:float,d_model:int):
        super().__init__()
        
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout,d_model) for _ in range(2)])
    
    def forward(self,x,src_mask):
        x = self.residual_connections[0](x,lambda x:self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connections[1](x,self.feed_forward_block)
        
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList,d_model:int):
        super().__init__()
        
        self.layers = layers
        self.norm = LayerNormalization(d_model)
    
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        
        return self.norm(x)
    


class DecoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttention,cross_attention_block:MultiHeadAttention,feed_forward_block:FeedForwardBlock,dropout:float,d_model:int):
        super().__init__()
        
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout,d_model) for _ in range(3)])
        
    
    def forward(self,x,encoder_output,src_mask,target_mask):
        
        x = self.residual_connections[0](x,lambda x:self.self_attention_block(x,x,x,target_mask))
        x = self.residual_connections[1](x,lambda x:self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connections[2](x,self.feed_forward_block)
        
        return x


class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList,d_model:int):
        super().__init__()
        
        self.layers = layers
        self.norm = LayerNormalization(d_model)
    
    def forward(self,x,encoder_output,src_mask,target_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,target_mask)
        
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        
        self.proj = nn.Linear(d_model,vocab_size)
        
    
    def forward(self,x):
        # return raw logits: F.cross_entropy fuses log_softmax+nll and is
        # cheaper/more stable; argmax is unchanged for greedy decoding.
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,src_embed:InputEmbedding,tar_embed:InputEmbedding,src_pos:PositionalEncoding,tar_pos:PositionalEncoding,projection_layer:ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tar_embed = tar_embed
        self.src_pos = src_pos
        self.tar_pos = tar_pos
        self.projection_layer = projection_layer
        
        
    def encode(self,src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self,encoder_output,src_mask,tgt,target_mask):
        tgt = self.tar_embed(tgt)
        tgt = self.tar_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,target_mask)
    
    def project(self,x):
        return self.projection_layer(x)

def buil_transformer(src_vocab_sz:int,tgt_vocab_sz:int , src_seq_len:int,tgt_seq_len:int,d_model:int = 512,N:int = 6,num_head:int=8,dropout:float= 0.1,d_ff:int = 2048)->Transformer:
    # create the embedding Layer
    src_embed = InputEmbedding(d_model,src_vocab_sz)
    tgt_embed = InputEmbedding(d_model,tgt_vocab_sz)
    
    # Create positional Encoding 
    src_pos = PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos = PositionalEncoding(d_model,tgt_seq_len,dropout)
    
    #initialization for encoder 
    encoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttention(d_model,num_head,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        encoder = EncoderBlock(self_attention_block,feed_forward_block,dropout,d_model)
        encoder_blocks.append(encoder)
    
    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttention(d_model,num_head,dropout)
        cross_self_attention_block = MultiHeadAttention(d_model,num_head,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        decoder = DecoderBlock(self_attention_block,cross_self_attention_block,feed_forward_block,dropout,d_model)
        decoder_blocks.append(decoder)
        
    
    # Create Encoder and Decoder
    
    encoder = Encoder(nn.ModuleList(encoder_blocks),d_model)
    decoder = Decoder(nn.ModuleList(decoder_blocks),d_model)
    
    # Create Projection Layer
    
    projection_layer = ProjectionLayer(d_model,tgt_vocab_sz)
    
    # Create Transformer
    
    transformer = Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)
    
    return transformer




