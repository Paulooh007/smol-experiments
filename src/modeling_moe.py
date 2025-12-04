# src/modeling_moe.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. Helper Components ---

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    if cos.device != q.device:
        cos = cos.to(q.device)
        sin = sin.to(q.device)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class RotaryEmbedder(nn.Module):
    def __init__(self, dim, base):
        super().__init__()
        self.freq = 1 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))

    @torch.no_grad()
    def forward(self, x):
        pos = torch.arange(x.shape[-2], dtype=torch.long, device=x.device)
        angles = torch.einsum('f,p->pf', self.freq.to(x.device), pos.float()).unsqueeze(dim=0)
        emb = torch.cat((angles, angles), dim=-1)
        return emb.cos(), emb.sin()

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states

# --- 2. Attention Mechanism ---

class RopeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.kv_heads = config.kv_heads
        self.rope_theta = config.rope_theta

        self.W_query = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.W_key = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.W_value = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.W_output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedder(base=self.rope_theta, dim=self.head_dim)

    def forward(self, hidden_states, attention_mask=None):
        b, q, _ = hidden_states.size()

        q_states = self.W_query(hidden_states)
        k_states = self.W_key(hidden_states)
        v_states = self.W_value(hidden_states)

        q_states = q_states.view(b, q, self.num_heads, self.head_dim).transpose(1, 2)
        k_states = k_states.view(b, q, self.kv_heads, self.head_dim).transpose(1, 2)
        v_states = v_states.view(b, q, self.kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(v_states)
        q_states, k_states = apply_rotary_pos_emb(q_states, k_states, cos, sin)

        __kv_groups = self.num_heads // self.kv_heads
        k_states = repeat_kv(k_states, __kv_groups)
        v_states = repeat_kv(v_states, __kv_groups)

        attn_weights = torch.matmul(q_states, k_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(b, q, -1)
        attn_output = self.W_output(attn_output)

        return attn_output

# --- 3. Mixture of Experts (MoE) ---

class MoE(nn.Module):
    """
    An MoE layer with MLP block with swiglue activation function.
    """
    def __init__(self, num_experts_per_tok: int, num_experts: int, emb_dim: int, moe_dim: int, dtype=torch.float32):
        super().__init__()
        self.k = int(num_experts_per_tok)
        self.E = int(num_experts)
        self.D = int(emb_dim)
        self.H = int(moe_dim)

        self.gate = nn.Linear(self.D, self.E, bias=False, dtype=dtype)

        self.gate_bank = nn.Parameter(torch.empty(self.E, self.D, self.H, dtype=dtype))
        self.up_bank   = nn.Parameter(torch.empty(self.E, self.D, self.H, dtype=dtype))
        self.down_bank = nn.Parameter(torch.empty(self.E, self.H, self.D, dtype=dtype))

    def expert_utilization(self, logits):
        """Compute expert utilization per layer and load balancer loss."""
        selected = torch.argmax(logits, dim=-1)
        selected = F.one_hot(selected, num_classes=self.E)
        load = torch.mean(selected.float(), dim=(0,1))
        importance = torch.mean(F.softmax(logits, dim=-1), dim=(0,1))
        self._aux_lb = self.E * torch.sum(load * importance)
        self._expert_utilization = selected

    def forward(self, x):
        B, T, D = x.shape
        logits = self.gate(x)

        if self.training:
            logits = logits + torch.randn_like(logits) * 1e-1

        selected = torch.argmax(logits, dim=-1)

        a = torch.einsum("btd,edh->bteh", x, self.gate_bank)
        u = torch.einsum("btd,edh->bteh", x, self.up_bank)
        h = F.silu(a) * u
        y = torch.einsum("bteh,ehd->bted", h, self.down_bank)

        gather_idx = selected.view(B,T,1,1).expand(-1, -1, -1, D)
        y = torch.gather(y, dim=2, index=gather_idx).squeeze(-2)

        self.expert_utilization(logits)
        return y

# --- 4. Full Model Assembly ---

class LlamaDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = RopeAttention(config)
        self.moe = MoE(num_experts=config.num_experts,
                       num_experts_per_tok=config.num_experts_per_tok,
                       emb_dim=config.hidden_size,
                       moe_dim=config.intermediate_size)
        self.pre_attn_rmsnorm = RMSNorm(config.hidden_size, eps=1e-05)
        self.pre_moe_rmsnorm = RMSNorm(config.hidden_size, eps=1e-05)

    def forward(self, hidden_states, attention_mask):
        residual = hidden_states
        hidden_states = self.pre_attn_rmsnorm(hidden_states)
        
        # Causal Mask
        seq_len = hidden_states.shape[1]
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device, dtype=hidden_states.dtype), diagonal=1)

        hidden_states = self.self_attn(hidden_states=hidden_states, attention_mask=causal_mask)
        hidden_states += residual
        residual = hidden_states

        hidden_states = self.pre_moe_rmsnorm(hidden_states)
        
        router_logits = self.moe.gate(hidden_states)
        hidden_states = self.moe(hidden_states)
        
        self.moe.expert_utilization(router_logits)
        
        hidden_states += residual
        return hidden_states, router_logits

class SmolMoEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoder(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=1e-05)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        all_router_logits = []
        
        for decoder_layer in self.layers:
            hidden_states, router_logits = decoder_layer(hidden_states, attention_mask)
            all_router_logits.append(router_logits)
            
        hidden_states = self.norm(hidden_states)
        return hidden_states, all_router_logits

class SmolMoELM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = SmolMoEModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        hidden_states, all_router_logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        return {'logits': logits, 'router_logits': all_router_logits}

    def get_expert_utilization(self):
        lb_vals, utils = [], []
        for layer in self.model.layers:
            moe = layer.moe
            if hasattr(moe, "_aux_lb"):
                lb_vals.append(moe._aux_lb)
            if hasattr(moe, "_expert_utilization"):
                utils.append(moe._expert_utilization)
        
        if lb_vals:
            lb_loss = torch.stack([v if v.dim() == 0 else v.squeeze() for v in lb_vals]).mean()
        else:
            lb_loss = torch.tensor(0.0, device=next(self.parameters()).device)
            
        return utils, lb_loss

    def reset_weights_and_metrics(self):
        """Reset parameters for clean upcycling"""
        with torch.no_grad():
            modules = list(self.modules())[1:]
            for m in modules:
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

# --- 5. Upcycling Utility ---

def upcycle_dense_to_moe(dense_model, moe_model):
    """
    Upcycles a dense model to the SmolMoEModel.
    Copies Attention/Embeddings directly.
    Copies Dense FFN weights into every expert of the MoE layer.
    """
    print("Starting Upcycling...")
    
    # 1. Token Embeddings and Final LM Head
    moe_model.model.embed_tokens.weight.data = dense_model.model.embed_tokens.weight.data.clone()
    moe_model.lm_head.weight.data = dense_model.lm_head.weight.data.clone()

    # 2. Final RMSNorm
    moe_model.model.norm.weight.data = dense_model.model.norm.weight.data.clone()

    # 3. Transformer Blocks
    for i, dense_layer in enumerate(dense_model.model.layers):
        moe_layer = moe_model.model.layers[i]

        # a. Attention Layers
        moe_layer.self_attn.W_query.weight.data = dense_layer.self_attn.q_proj.weight.data.clone()
        moe_layer.self_attn.W_key.weight.data = dense_layer.self_attn.k_proj.weight.data.clone()
        moe_layer.self_attn.W_value.weight.data = dense_layer.self_attn.v_proj.weight.data.clone()
        moe_layer.self_attn.W_output.weight.data = dense_layer.self_attn.o_proj.weight.data.clone()

        # b. Pre-Attention RMSNorm
        moe_layer.pre_attn_rmsnorm.weight.data = dense_layer.input_layernorm.weight.data.clone()

        # c. Pre-MoE RMSNorm
        moe_layer.pre_moe_rmsnorm.weight.data = dense_layer.post_attention_layernorm.weight.data.clone()

        # d. FFN -> MoE Experts (The Core Upcycling Step for SwiGLU)
        dense_ffn_gate = dense_layer.mlp.gate_proj.weight.data.clone().T
        dense_ffn_up = dense_layer.mlp.up_proj.weight.data.clone().T
        dense_ffn_down = dense_layer.mlp.down_proj.weight.data.clone().T

        # Replicate the dense FFN weights for each expert
        moe_layer.moe.gate_bank.data.copy_(dense_ffn_gate.unsqueeze(0).expand_as(moe_layer.moe.gate_bank))
        moe_layer.moe.up_bank.data.copy_(dense_ffn_up.unsqueeze(0).expand_as(moe_layer.moe.up_bank))
        moe_layer.moe.down_bank.data.copy_(dense_ffn_down.unsqueeze(0).expand_as(moe_layer.moe.down_bank))

        # e. Initialize Router (Gate) weights to small random values
        nn.init.kaiming_uniform_(moe_layer.moe.gate.weight, a=math.sqrt(5))

    print("Upcycling complete!")