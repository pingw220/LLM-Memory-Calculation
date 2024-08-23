import streamlit as st
import json
from config import (
    DATA_TYPES,
    PARAMETERS,
    OPTIMIZERS,
    PARAMETERS_SYNONYMS,
    load_predefined_models,
)
from util import (
    calculate_inference_memory,
    calculate_training_memory,
    weight_parameters,
)


# Streamlit Setup
st.set_page_config(page_title="LLM Memory Calculation")
st.title("LLM Memory Calculation")


# Sidebar Initialization
MODELS = load_predefined_models()


def set_values():
    if st.session_state.file_upload is not None:
        config = json.load(st.session_state.file_upload)
        print(config["num_attention_heads"])
        for param in PARAMETERS:
            if PARAMETERS[param] in config:
                print(param)
                st.session_state[PARAMETERS_SYNONYMS[PARAMETERS[param]]] = config[PARAMETERS[param]]
            elif param == "model_size":
                st.session_state[param] = weight_parameters(config)
    if st.session_state.model in MODELS:
        model_info = MODELS[st.session_state.model]
        for param in PARAMETERS:
            if PARAMETERS[param] in model_info:
                print(param)
                st.session_state[PARAMETERS_SYNONYMS[PARAMETERS[param]]] = model_info[PARAMETERS[param]]
            elif param == "model_size":
                st.session_state[param] = weight_parameters(model_info)
        if st.session_state.file_upload is not None:
            st.session_state.model = None
    


# Sidebar UI
# Model Selection
model = st.sidebar.selectbox(
    "Model", list(MODELS.keys()), index=None, on_change=set_values, key="model"
)

# File Upload
file_upload = st.file_uploader(
    "Upload the config file here",
    type=".json",
    accept_multiple_files=False,
    on_change=set_values,
    key="file_upload",
    help="Please input the config file if the model is not listed.",
    label_visibility="visible",
)

# Parameters
model_size = st.sidebar.number_input(
    "Number of parameters (in billions)",
    min_value=0,
    step=1,
    value=None,
    key="model_size",
    help="Number of parameters in the model in billions",
)
precision = st.sidebar.selectbox(
    "Precision",
    DATA_TYPES,
    index=2,
    key="precision",
    help="Data type used (int 8 and int 4 are for quantization)",
)
batch_size = st.sidebar.number_input(
    "Batch Size",
    min_value=0,
    step=1,
    value=1,
    key="batch_size",
)
sequence_length = st.sidebar.number_input(
    "Sequence Length",
    min_value=0,
    step=1,
    value=2048,
    key="sequence_length",
    help="Number of tokens in the input sequence.",
)
hidden_size = st.sidebar.number_input(
    "Hidden Size",
    min_value=0,
    step=1,
    value=None,
    key="hidden_size",
    help="Size of the hidden layer (given by the model card).",
)
num_hidden_layers = st.sidebar.number_input(
    "Number of Layers",
    min_value=0,
    step=1,
    value=None,
    key="num_hidden_layers",
    help="Number of layers in the model (given by the model card).",
)
num_attention_heads = st.sidebar.number_input(
    "Number of Attention Heads",
    min_value=0,
    step=1,
    value=None,
    key="num_attention_heads",
    help="Number of attention heads in the model (given by the model card).",
)
num_key_value_heads = st.sidebar.number_input(
    "Number of KV Heads",
    min_value=0,
    step=1,
    value=None,
    key="num_key_value_heads",
    help="Number of KV heads in the model (given by the model card).",
)


# Main Screen UI
# Dividing the screen into two tabs
inference, training = st.tabs(["Inference", "Training"])

# Tab 2: Training
training1, training2 = training.columns(2)
optimizer = training2.selectbox("Optimizer", list(OPTIMIZERS.keys()), key="optimizer")
trainable_parameters = training2.slider(
    "Percentage of trainable parameters", 0, 100, 100, key="trainable_params"
)

# Inference Memory
inference_memory = calculate_inference_memory(
    model_size,
    precision,
    batch_size,
    sequence_length,
    hidden_size,
    num_hidden_layers,
    num_attention_heads,
)

inference.write(f"**Total Inference Memory**: {inference_memory['inference_memory']}")
inference.write(f"- **Model Weights**: {inference_memory['model_weights']}")
inference.write(f"- **KV Cache**: {inference_memory['kv_cache']}")
inference.write(f"- **Activation Memory**: {inference_memory['activation_memory']}")


# Training Memory
training_memory = calculate_training_memory(
    model_size,
    precision,
    batch_size,
    sequence_length,
    hidden_size,
    num_hidden_layers,
    num_attention_heads,
    optimizer,
    trainable_parameters,
)

training1.write(f"**Total Training Memory**: {training_memory['training_memory']}")
training1.write(f"- **Model Weights**: {training_memory['model_weights']}")
training1.write(f"- **KV Cache**: {training_memory['kv_cache']}")
training1.write(f"- **Activation Memory**: {training_memory['activation_memory']}")
training1.write(f"- **Optimizer Memory**: {training_memory['optimizer_memory']}")
training1.write(f"- **Gradients Memory**: {training_memory['gradients_memory']}")

# Error
if None in st.session_state.values():
    st.warning("Some information is missing.")