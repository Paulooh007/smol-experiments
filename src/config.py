class SmolConfig:
    vocab_size = 49152
    hidden_size = 576
    intermediate_size = 1536
    num_hidden_layers = 30
    num_heads = 9
    kv_heads = 3

class SmolMoEConfig(SmolConfig):
    num_experts = 3
    num_experts_per_tok = 1