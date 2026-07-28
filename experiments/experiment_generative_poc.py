import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from collections import defaultdict
import matplotlib.pyplot as plt
import sys
import os

# Ensure src is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.generators.spectral_pauli_generator import generate_pauli_strings
from src.utils.pauli_utils import get_pauli_tensor

from src.utils.seeds import set_seed
set_seed()

# ==========================================
# 1. Vocabulary & Data Preparation
# ==========================================
class CharVocab:
    def __init__(self, chars="ABCDEFGHIJKLMNOPQRSTUVWXYZ ."):
        self.chars = list(chars)
        self.vocab_size = len(self.chars)
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for i, c in enumerate(self.chars)}
        
        # We need n_qubits such that 2^n >= vocab_size
        self.n_qubits = int(np.ceil(np.log2(self.vocab_size)))
        print(f"Vocab Size: {self.vocab_size}, Qubits Required: {self.n_qubits}")

    def char_to_idx(self, char):
        return self.char2idx.get(char, 0) # default to first char if unknown

    def idx_to_char(self, idx):
        return self.idx2char.get(idx, "?")

def prepare_dataset(text, vocab, context_size=3):
    X, y = [], []
    for i in range(len(text) - context_size):
        context = text[i:i+context_size]
        target = text[i+context_size]
        
        # Convert context characters to their indices
        ctx_indices = [vocab.char_to_idx(c) for c in context]
        X.append(ctx_indices)
        y.append(vocab.char_to_idx(target))
        
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# ==========================================
# 2. Generalized Spectral Selection
# ==========================================
def get_global_spectral_paulis(X_emb, n_qubits, top_k=50):
    """
    Generalized Spectral Method:
    Instead of class difference, we use the global covariance of the embeddings
    to find interactions (Pauli strings) that capture the most variance in the sequence context.
    
    X_emb: (N_samples, context_dim)
    """
    # X_emb is typically (N, context_size * embed_dim)
    # We treat it as context vectors and compute R = X^T X
    R = (X_emb.T @ X_emb) / len(X_emb)
    dim = R.shape[0]
    
    # Generate all candidate strings for the given subsystem required by context.
    # For a PoC, if context is large, we might need a sub-selection, but let's assume
    # we map the entire flattened context to a small n_sys qubits.
    # To map efficiently, we assume dim = 2^n_sys for the Paulis.
    n_sys = int(np.ceil(np.log2(dim)))
    
    all_strings = generate_pauli_strings(n_sys)
    results = []
    factor = 1.0 / (2**n_sys)
    
    # Pad R if necessary to be 2^n_sys x 2^n_sys
    padded_R = np.zeros((2**n_sys, 2**n_sys))
    padded_R[:dim, :dim] = R
    
    for s in all_strings:
        P = get_pauli_tensor(s)
        val = np.trace(padded_R @ P) * factor
        mag = np.abs(val)
        results.append((s, mag))
        
    results.sort(key=lambda x: x[1], reverse=True)
    
    strings = [r[0] for r in results[:top_k]]
    return strings

# ==========================================
# 3. Model Definition
# ==========================================
class GenerativeSIM(nn.Module):
    def __init__(self, vocab_size, context_size, embed_dim, n_qubits, pauli_strings):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits # Size of output space
        self.pauli_strings = pauli_strings
        
        # Classical Embeddings for context
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Flattened context dim
        self.flat_dim = context_size * embed_dim
        
        # The Flipped Model part: We project the classical context into Pauli coefficients
        self.pauli_proj = nn.Linear(self.flat_dim, len(pauli_strings))
        
        # Quantum Device
        # Use simple default qubit. We don't use 'probs' directly in training to simulate "shots",
        # but PyTorch needs gradients, so we output probabilities for training.
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Learnable rotational params for the optimal subspace filter
        self.q_params = nn.Parameter(torch.randn(3, n_qubits, requires_grad=True))

        @qml.qnode(self.dev, interface="torch")
        def quantum_generator(params, features):
            """
            features: the selected Pauli coefficients mapped from the context.
            This circuit creates a superposition and applies rotations.
            """
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            
            # Simple Strongly Entangling layer for state prep
            for i in range(n_qubits):
                qml.Rot(*params[:, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[n_qubits-1, 0])
            
            # Here, instead of measuring Expectation, we measure probabilities of the basis states!
            # These probabilities act as the "Softmax" over the 2^N vocabulary.
            return qml.probs(wires=range(n_qubits))
            
        self.qnode = quantum_generator

    def forward(self, x):
        # x is (batch, context_size)
        emb = self.embedding(x) # (batch, context_size, embed_dim)
        emb_flat = emb.view(x.size(0), -1) # (batch, context_size * embed_dim)
        
        # Map context to Pauli coefficients
        pauli_coeffs = self.pauli_proj(emb_flat)
        pauli_coeffs = torch.relu(pauli_coeffs) # Introduce non-linearity
        
        # Quantum Forward Pass (batch processing via vmap in an ideal world, using list comp here)
        batch_probs = []
        for i in range(x.size(0)):
            probs = self.qnode(self.q_params, pauli_coeffs[i])
            batch_probs.append(probs)
            
        probs_tensor = torch.stack(batch_probs)
        
        # Truncate probabilities to actual vocab size if 2^N > vocab_size
        return probs_tensor[:, :self.vocab_size]

# ==========================================
# 4. Training & Generative Loop
# ==========================================
def train():
    # 1. Setup
    text = "HELLO QUANTUM WORLD. THIS IS A GENERATIVE TEST. IT USES SPECTRAL INTERACTIONS."
    # Use only required characters for this text to keep standard 2^5 = 32
    chars = sorted(list(set(text)))
    if len(chars) < 32: # Pad to ensure we have a good mapping conceptually
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ .":
            if c not in chars and len(chars) < 32:
                chars.append(c)
                
    vocab = CharVocab("".join(chars)) # vocab_size <= 32
    context_size = 4
    embed_dim = 4 # Small embedding
    
    X, y = prepare_dataset(text, vocab, context_size)
    print(f"Dataset Size: {len(X)} samples")

    # 2. Spectral Method (Global Covariance)
    print("Applying Generalized Spectral Method...")
    # Simulate embedding manually for Spectral selection
    sim_emb = nn.Embedding(vocab.vocab_size, embed_dim)
    with torch.no_grad():
        X_emb = sim_emb(X).view(len(X), -1).numpy()
        
    top_paulis = get_global_spectral_paulis(X_emb, n_qubits=vocab.n_qubits, top_k=64)
    print(f"Selected Top {len(top_paulis)} Contextual Paulis.")

    # 3. Model
    model = GenerativeSIM(vocab.vocab_size, context_size, embed_dim, vocab.n_qubits, top_paulis)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    
    # 4. Train Loop
    epochs = 150
    print("Beginning Training...")
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        # forward returns raw probabilities. Log them to simulate log_softmax for NLLLoss, 
        # or use raw probabilities with some scaling. CrossEntropy takes logits.
        probs = model(X)
        
        # Avoid log(0)
        log_probs = torch.log(probs + 1e-8)
        
        loss = nn.NLLLoss()(log_probs, y)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")

    # Plot Loss 
    plt.figure()
    plt.plot(loss_history)
    plt.title("Generative Quantum Flipped Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("NLL Loss")
    plt.savefig("results/generative_poc_loss.png")
    plt.close()

    # 5. Generation (Simulating "Shots")
    print("\n--- Quantum Text Generation ---")
    seed_text = "HELL"
    print(f"Seed Context: '{seed_text}'")
    
    generated_text = seed_text
    current_context = [vocab.char_to_idx(c) for c in seed_text]
    
    model.eval()
    with torch.no_grad():
        for _ in range(20): # Generate 20 chars
            ctx_tensor = torch.tensor([current_context], dtype=torch.long)
            probs = model(ctx_tensor)[0].numpy()
            
            # SIMULATE SHOTS: Instead of argmax, we sample from the probability distribution
            # This is exactly what a quantum computer does naturally when measured!
            # shots = 1 (we just take one sample from the distribution)
            next_idx = np.random.choice(len(probs), p=probs/np.sum(probs))
            
            next_char = vocab.idx_to_char(next_idx)
            generated_text += next_char
            
            # Update context
            current_context = current_context[1:] + [next_idx]

    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    import os
    if not os.path.exists("results"):
        os.makedirs("results")
    train()
