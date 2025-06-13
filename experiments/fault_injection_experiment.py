import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from datasets import load_dataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset (e.g. IMDb) and prepare tokenizer
dataset = load_dataset("imdb")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def encode_batch(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)

dataset = dataset.map(encode_batch, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(dataset["train"], batch_size=16, shuffle=True,num_workers=0)
test_loader = DataLoader(dataset["test"], batch_size=32,num_workers=0)

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
model.to(device)

model_save_path = "./distilbert_finetuned.pt"

# Training loop
def train(model, epochs=2):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss:.2f}")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Evaluation function
def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    return accuracy

# Hook to record neuron activations in Linear layers globally
activation_dict = {}
def get_activation(name):
    def hook(model, input, output):
        # sum activations over batch and seq dims if needed
        # output shape for classifier: (batch_size, num_classes)
        # For generality, flatten to 2D and sum abs values over batch dimension
        if output.dim() > 2:
            val = output.abs().sum(dim=[0,1]).cpu()
        else:
            val = output.abs().sum(dim=0).cpu()
        activation_dict[name] = activation_dict.get(name, torch.zeros_like(val)) + val
    return hook

# Register hooks on all nn.Linear layers in the model
linear_layers = []
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        linear_layers.append((name, module))
        module.register_forward_hook(get_activation(name))

# Fault injection functions
def inject_faults(model, layer_neuron_pairs, original_weights):
    for layer_name, neuron_idx in layer_neuron_pairs:
        layer = dict(model.named_modules())[layer_name]
        with torch.no_grad():
            layer.weight[neuron_idx] = -layer.weight[neuron_idx]

def restore_weights(model, layer_neuron_pairs, original_weights):
    for (layer_name, neuron_idx), orig_w in zip(layer_neuron_pairs, original_weights):
        layer = dict(model.named_modules())[layer_name]
        with torch.no_grad():
            layer.weight[neuron_idx] = orig_w

# Main experiment flow
def main():
    # Load saved model or train if not available
    try:
        model.load_state_dict(torch.load(model_save_path))
        print("Loaded saved model.")
    except FileNotFoundError:
        print("No saved model found. Training now...")
        train(model)

    # Evaluate base accuracy
    base_acc = evaluate(model)
    print(f"Base Accuracy: {base_acc:.2f}%")

    # Clear activation dict before forward pass
    activation_dict.clear()

    # Run a full forward pass over test set to gather activations
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            _ = model(input_ids, attention_mask=attention_mask)

    # Aggregate activations globally and get top neurons
    # activation_dict keys: layer names, values: tensors with shape [out_features]
    neuron_activations = []
    for layer_name, acts in activation_dict.items():
        for neuron_idx, act_val in enumerate(acts):
            neuron_activations.append((layer_name, neuron_idx, act_val.item()))

    # Sort descending by activation magnitude
    neuron_activations.sort(key=lambda x: abs(x[2]), reverse=True)

    # Pick top N neurons globally (e.g., top 10)
    top_n = 10
    top_neurons = neuron_activations[:top_n]

    print("Top activated neurons globally:")
    for layer_name, neuron_idx, act_val in top_neurons:
        print(f"Layer: {layer_name}, Neuron: {neuron_idx}, Total Activation: {act_val:.2f}")

    # Save original weights of these neurons for restoration later
    layer_neuron_pairs = [(ln, ni) for ln, ni, _ in top_neurons]
    original_weights = []
    for layer_name, neuron_idx in layer_neuron_pairs:
        layer = dict(model.named_modules())[layer_name]
        original_weights.append(layer.weight[neuron_idx].clone())

    # Inject faults (flip weights) for all top neurons simultaneously
    print("\nInjecting faults into all top neurons simultaneously...")
    inject_faults(model, layer_neuron_pairs, original_weights)

    # Evaluate after fault injection
    fault_acc = evaluate(model)
    print(f"Accuracy after fault injection: {fault_acc:.2f}%")

    # Restore original weights
    restore_weights(model, layer_neuron_pairs, original_weights)

    # Evaluate again to confirm restoration
    restored_acc = evaluate(model)
    print(f"Accuracy after weight restoration: {restored_acc:.2f}%")

if __name__ == "__main__":
    main()
