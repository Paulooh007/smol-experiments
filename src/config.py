class SmolConfig:
    """Base configuration for SmolLM-135M"""
    vocab_size = 49152
    hidden_size = 576
    intermediate_size = 1536
    num_hidden_layers = 30
    num_heads = 9
    kv_heads = 3

class SmolMoEConfig(SmolConfig):
    """Configuration for the Upcycled MoE Model"""
    num_experts = 3
    num_experts_per_tok = 1

    expert_map = {
        0: [0], # Code -> Expert 0
        1: [1], # Math -> Expert 1
        2: [2], # Chat -> Expert 2
    }
    low_p = 0.001 # Probability for non-targeted experts