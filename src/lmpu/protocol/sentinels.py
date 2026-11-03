"""LMPU Protocol v1 - Sentinel tokens for model-plane communication."""

# Model-plane sentinels (reserved-token form, string-encoded)
PROC_OPEN = "<|proc|>"
PROC_CLOSE = "<|/proc|>"
PROC_EVENT_OPEN = "<|proc_event|>"
PROC_EVENT_CLOSE = "<|/proc_event|>"

# Protocol version
PROTOCOL_VERSION = "lmpu/1"

# Sentinel tokenization info (for documentation)
# With Qwen/Qwen3.5-0.8B tokenizer:
# <|proc|> -> ['<', '|', 'proc', '|', '>'] (5 tokens)
# <|/proc|> -> ['<', '|', '/', 'proc', '|', '>'] (6 tokens)
# <|proc_event|> -> ['<', '|', 'proc_event', '|', '>'] (5 tokens)
# <|/proc_event|> -> ['<', '|', '/', 'proc_event', '|', '>'] (6 tokens)