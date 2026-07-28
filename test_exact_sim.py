import torch
from src.models.exact_sim_classifier import ExactSIMClassifier

print("Instantiating ExactSIMClassifier for n_qubits=2 (binary)")
n_qubits = 2
model_bin = ExactSIMClassifier(n_qubits=n_qubits, num_classes=1, pauli_strings=["II", "IZ", "ZI", "ZZ"])

print("Testing (batch, dim)")
x_flat = torch.randn(4, 2**n_qubits)
out_flat = model_bin(x_flat)
print(f"Shape: {out_flat.shape}")
assert out_flat.shape == (4,)

print("Testing (batch, seq_len, dim)")
x_seq = torch.randn(4, 3, 2**n_qubits)
out_seq = model_bin(x_seq)
print(f"Shape: {out_seq.shape}")
assert out_seq.shape == (4,)

print("Instantiating ExactSIMClassifier for n_qubits=2 (multi-class=3)")
model_multi = ExactSIMClassifier(n_qubits=n_qubits, num_classes=3, pauli_strings=["II", "IZ", "ZI", "ZZ"])

out_multi_flat = model_multi(x_flat)
print(f"Multi Shape: {out_multi_flat.shape}")
assert out_multi_flat.shape == (4, 3)

out_multi_seq = model_multi(x_seq)
print(f"Multi Seq Shape: {out_multi_seq.shape}")
assert out_multi_seq.shape == (4, 3)

print("All tests passed!")
