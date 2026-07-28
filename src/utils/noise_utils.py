import pennylane as qml

def get_depolarizing_noise_model(p, wires=None):
    """
    Creates a PennyLane NoiseModel for depolarizing noise, compatible with latest versions (0.42+).
    
    Args:
        p (float): Probability of depolarization for each gate.
        wires (list, optional): The wires to apply noise to. If None, it attempts 
                                to apply to a broad set of common gates.
        
    Returns:
        qml.noise.NoiseModel: The unified noise model.
    """
    if p <= 0:
        return None
        
    # Standard: Apply depolarization after every gate on specified wires
    def noise_rule(op, **kwargs):
        # Apply to all operations that aren't channels themselves to avoid infinite recursion
        if isinstance(op, (qml.operation.Channel, qml.operation.Observable)):
            return None
        return [qml.DepolarizingChannel(p, wires=w) for w in op.wires]
        
    if wires is not None:
        # Match any operation whose wires are a subset of provided wires
        cond = qml.noise.wires_in(wires)
    else:
        # Fallback: Match most common gates if wires are unknown
        cond = qml.noise.op_in([
            "RX", "RY", "RZ", "PauliRot", "CNOT", "CZ", "Hadamard", 
            "PauliX", "PauliY", "PauliZ", "Rot", "PhaseShift", "SWAP"
        ])

    return qml.noise.NoiseModel({cond: noise_rule})

def apply_layer_noise(p, wires, noise_type='depolarizing'):
    """
    Legacy functional approach (for immediate backward compatibility if needed).
    Better to use qml.add_noise(qnode, noise_model).
    """
    if p <= 0:
        return
    for i in wires:
        if noise_type == 'depolarizing':
            qml.DepolarizingChannel(p, wires=i)

