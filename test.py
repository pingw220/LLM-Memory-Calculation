import json
import os
import re


PARAMETERS = {
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
    return total_params

def find_config_file(model_name):
    directory = os.path.dirname(os.path.abspath(__file__))  # Directory where the script is located
    for file in os.listdir(directory):
        if file.lower().startswith(model_name.lower()) and file.endswith('.json'):
            return os.path.join(directory, file)
    return None

def main():
        
    config_file_path = input("Enter the path to your config file: ")

    if config_file_path:
        print(f"Found config file: {config_file_path}")
    else:
        print(f"No config file found for model.")
        return

    with open(config_file_path, 'r') as f:
        config = json.load(f)

    weight_parameters(config)

if __name__ == "__main__":
    main()
