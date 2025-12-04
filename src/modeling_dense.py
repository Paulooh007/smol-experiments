import torch
import torch.nn as nn
from src.modeling_moe import RopeAttention, RMSNorm

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.W_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.W_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.W_down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.W_down(self.act_fn(self.W_gate(x)) * self.W_up(x))

class LlamaDecoderDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = RopeAttention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.pre_attn_rmsnorm = RMSNorm(config.hidden_size, eps=1e-05)
        self.pre_mlp_rmsnorm = RMSNorm(config.hidden_size, eps=1e-05)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.pre_attn_rmsnorm(hidden_states)
        
        # Create causal mask logic here (simplified)
        
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states += residual
        
        residual = hidden_states
        hidden_states = self.pre_mlp_rmsnorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual
        
        return (hidden_states,)

class SmolDenseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderDense(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=1e-05)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)[0]
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return {'logits': logits}