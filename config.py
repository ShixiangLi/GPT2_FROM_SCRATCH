
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": True         # Query-Key-Value bias
 }

GPT_CONFIG_355M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 1024,         # Embedding dimension
    "n_heads": 16,           # Number of attention heads
    "n_layers": 24,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
 }

GPT_CONFIG_774M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 1280,         # Embedding dimension
    "n_heads": 20,           # Number of attention heads
    "n_layers": 36,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
 }

GPT_CONFIG_1558M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 1600,         # Embedding dimension
    "n_heads": 25,           # Number of attention heads
    "n_layers": 48,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
 }


CONFIG = {
    "GPT_CONFIG_124M": GPT_CONFIG_124M,
    "GPT_CONFIG_355M": GPT_CONFIG_355M,
    "GPT_CONFIG_774M": GPT_CONFIG_774M,
    "GPT_CONFIG_1558M": GPT_CONFIG_1558M
}



