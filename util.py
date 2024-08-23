import streamlit as st
from config import DATA_TYPES, PARAMETERS, DATA_TYPE_SIZES, OPTIMIZERS

config_synonyms = {
    "model_type": ["model_type"],
    "vocab_size": ["vocab_size"],
    "hidden_size": ["hidden_size", "d_model"],
    "max_position_embeddings": ["max_position_embeddings"],
    "intermediate_size": ["intermediate_size", "encoder_ffn_dim", "decoder_ffn_dim"],
    "num_hidden_layers": ["num_hidden_layers", "encoder_layers", "encoder_layers"],
    "num_attention_heads": ["num_attention_heads", "encoder_attention_heads", "decoder_attention_heads"],
    "num_key_value_heads": ["num_key_value_heads", "num_kv_heads"]
}

def find_category(config, category, synonyms_dict):
    for target, synonyms in synonyms_dict.items():
        if target == category:
            for synonym in synonyms:
                if synonym in config:
                    return config[synonym]
    return 1

def weight_parameters(config):
    model_type = find_category(config, "model_type", config_synonyms)
    vocab_size = find_category(config, "vocab_size", config_synonyms)
    hidden_size = find_category(config, "hidden_size", config_synonyms)
    embed_dim = hidden_size
    max_position_embeddings = find_category(config, "max_position_embeddings", config_synonyms)
    intermediate_size = find_category(config, "intermediate_size", config_synonyms)
    num_hidden_layers = find_category(config, "num_hidden_layers", config_synonyms)
    num_attention_heads = find_category(config, "num_attention_heads", config_synonyms)
    num_key_value_heads = find_category(config, "num_key_value_heads", config_synonyms)
    num_experts = find_category(config, "num_local_experts", config_synonyms)

    head_dim = hidden_size / num_attention_heads

    if model_type == "mistral":
        num_ffn = 3
        num_layernorm = 3
        num_large_rope = 2
        num_small_rope = 1
        
    if model_type == "mixtral":
        num_ffn = 3
        num_layernorm = 1
        num_large_rope = 2
        num_small_rope = 1
        num_experts = config['num_local_experts']

    if model_type == "llama":
        num_ffn = 3
        num_layernorm = 1
        num_large_rope = 0
        num_small_rope = 1
        
    if model_type == "falcon":
        num_ffn = 2
        num_layernorm = 3
        num_large_rope = 2
        num_small_rope = 1
        intermediate_size = 4 * hidden_size

    else:
        num_ffn = 3
        num_layernorm = 1
        num_large_rope = 0
        num_small_rope = 1
        
    # if model_type == "whisper":
    #     num_ffn = 3
    #     num_layernorm = 3
    #     num_large_rope = 1
    #     num_small_rope = 1
        

    # Embedding Layers
    embedding_layers = vocab_size * hidden_size

    # Layer Norm
    layer_norm = num_layernorm * hidden_size
    
    # Attention Layers

    # Multi-headed Attention
    if num_key_value_heads == 1:
        num_attention_heads = hidden_size / head_dim
        num_key_value_heads = hidden_size / head_dim

    # Q Layer
    q_layer = embed_dim * num_attention_heads * head_dim

    # O Layer
    o_layer = embed_dim * num_attention_heads * head_dim

    # K Layer
    k_layer = embed_dim * num_key_value_heads * head_dim

    # V Layer
    v_layer = embed_dim * num_key_value_heads * head_dim

    attention_layers = q_layer + o_layer + k_layer + v_layer

    # RoPE
    rope_large = num_large_rope * max_position_embeddings * head_dim
    rope_small = num_small_rope * head_dim
    
    # Feedforward Layers
    ffn_layers = num_ffn * embed_dim * intermediate_size * num_experts

    # Linear
    # End
    end_layer = vocab_size * embed_dim

    # Total parameters
    total_params = embedding_layers + (attention_layers + ffn_layers + layer_norm + rope_large + rope_small) * num_hidden_layers + layer_norm + end_layer
    
    print(f"Weight Parameters: {total_params:,}")
    print(f"Model Weights: {total_params * 2 / 1024 / 1024 / 1024:,} GB")

    print(f"Activation Formula 1: {2048 * hidden_size * 132 * 4 / (1024**3):,} GB")
    print(f"Activation Formula 2: {2048 * hidden_size * (34 + (5 * 2048 * num_attention_heads) / hidden_size) / (1024**3) * 4:,} GB")
    return total_params / 1000**3

# Memory Functions
@st.cache_data
def get_memory(*args):
    """Convert total memory from bytes to human-readable format."""
    total = 0
    warning = False
    for arg in args:
        if arg > 0:
            total += arg
        else:
            warning = True
    # Convert bytes to human-readable format
    if total == 0:
        result = ""
    elif total < 1024:
        result = f"{total} Bytes"
    elif total < 1024**2:
        result = f"{total / 1024:.2f} KB"
    elif total < 1024**3:
        result = f"{total / (1024**2):.2f} MB"
    elif total < 1024**4:
        result = f"{total / (1024**3):.2f} GB"
    else:
        result = f"{total / (1024**4):.2f} TB"
    result += " * " if warning else ""
    return result


@st.cache_data
def get_model_weights(model_size, precision):
    """Calculate the memory required for model weights."""
    try:
        return model_size * DATA_TYPE_SIZES[precision] * (10**9)
    except:
        return 0


@st.cache_data
def get_kv_cache(
    precision, batch_size, sequence_length, hidden_size, num_hidden_layers
):
    """Calculate the memory required for key-value cache."""
    try:
        return (
            2
            * batch_size
            * sequence_length
            * num_hidden_layers
            * hidden_size
            * DATA_TYPE_SIZES[precision]
        )
    except:
        return 0


@st.cache_data
def get_activation_memory(
    batch_size, sequence_length, hidden_size, num_attention_heads
):
    """Calculate the memory required for activations."""
    precision = "float32"
    try:
        return (
            batch_size
            * sequence_length
            * hidden_size
            * (34 + (5 * sequence_length * num_attention_heads) / hidden_size)
            * DATA_TYPE_SIZES[precision]
        )
    except:
        return 0


@st.cache_data
def get_optimizer_memory(model_size, optimizer):
    """Calculate the memory required for optimizer."""
    try:
        return OPTIMIZERS[optimizer] * model_size * (10**9)
    except:
        return 0


@st.cache_data
def get_gradient_memory(model_size, precision):
    """Calculate the memory required for gradients."""
    precision = "float32"
    try:
        return DATA_TYPE_SIZES[precision] * model_size * (10**9)
    except:
        return 0


@st.cache_data
def calculate_inference_memory(
    model_size,
    precision,
    batch_size,
    sequence_length,
    hidden_size,
    num_hidden_layers,
    num_attention_heads,
):
    """Calculate the total memory required for inference."""
    model_weights = get_model_weights(model_size, precision)
    kv_cache = get_kv_cache(
        precision, batch_size, sequence_length, hidden_size, num_hidden_layers
    )
    activation_memory = get_activation_memory(
        batch_size, sequence_length, hidden_size, num_attention_heads
    )
    return {
        "model_weights": get_memory(model_weights),
        "kv_cache": get_memory(kv_cache),
        "activation_memory": get_memory(activation_memory),
        "inference_memory": get_memory(model_weights, kv_cache, activation_memory),
    }


@st.cache_data
def calculate_training_memory(
    model_size,
    precision,
    batch_size,
    sequence_length,
    hidden_size,
    num_hidden_layers,
    num_attention_heads,
    optimizer,
    trainable_parameters,
):
    """Calculate the total memory required for training."""
    model_weights = get_model_weights(model_size, precision)
    kv_cache = get_kv_cache(
        precision, batch_size, sequence_length, hidden_size, num_hidden_layers
    )
    activation_memory = get_activation_memory(
        batch_size, sequence_length, hidden_size, num_attention_heads
    )
    optimizer_memory = (
        get_optimizer_memory(model_size, optimizer) * trainable_parameters / 100
    )
    gradients_memory = (
        get_gradient_memory(model_size, precision) * trainable_parameters / 100
    )

    return {
        "model_weights": get_memory(model_weights),
        "kv_cache": get_memory(kv_cache),
        "activation_memory": get_memory(activation_memory),
        "optimizer_memory": get_memory(optimizer_memory),
        "gradients_memory": get_memory(gradients_memory),
        "training_memory": get_memory(
            model_weights,
            kv_cache,
            activation_memory,
            optimizer_memory,
            gradients_memory,
        ),
    }