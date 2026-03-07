import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
from torchvision import models

N_QUBITS = 4
N_LAYERS = 2

# ============================================================
# QUANTUM CIRCUITS
# ============================================================
DEV_SINGLE = qml.device("lightning.qubit", wires=1)

@qml.qnode(DEV_SINGLE, interface="torch", diff_method="adjoint")
def single_qubit_circuit(inputs, weights, noise_type=None, noise_prob=0.0):
    qml.RY(inputs[0], wires=0)
    for layer in range(weights.shape[0]):
        qml.RX(weights[layer, 0], wires=0)
        qml.RY(weights[layer, 1], wires=0)
        qml.RZ(weights[layer, 2], wires=0)
    return qml.expval(qml.PauliZ(0))


DEV_ENTANGLE = qml.device("lightning.qubit", wires=N_QUBITS)

@qml.qnode(DEV_ENTANGLE, interface="torch", diff_method="adjoint")
def entanglement_circuit(inputs, noise_type=None, noise_prob=0.0):
    for q in range(N_QUBITS):
        qml.Hadamard(wires=q)
        qml.PhaseShift(inputs[q], wires=q)
    for _ in range(N_LAYERS):
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
        for q in range(0, N_QUBITS - 2, 2):
            qml.CNOT(wires=[q, q + 2])
    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]


DEV_FULL = qml.device("lightning.qubit", wires=N_QUBITS)

@qml.qnode(DEV_FULL, interface="torch", diff_method="adjoint")
def full_variational_circuit(inputs, weights, noise_type=None, noise_prob=0.0):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    for layer in range(weights.shape[0]):
        for q in range(N_QUBITS):
            qml.RX(weights[layer, q, 0], wires=q)
            qml.RY(weights[layer, q, 1], wires=q)
            qml.RZ(weights[layer, q, 2], wires=q)
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]


# ============================================================
# QUANTUM LAYERS
# ============================================================
class SingleQubitLayer(nn.Module):
    def __init__(self, n_layers=N_LAYERS, noise_type=None, noise_prob=0.0):
        super().__init__()
        self.noise_type = noise_type
        self.noise_prob = noise_prob
        w = torch.empty(n_layers, 3)
        nn.init.uniform_(w, -np.pi/4, np.pi/4)
        self.weights = nn.Parameter(w)

    def forward(self, x):
        results = []
        for sample in x:
            out = single_qubit_circuit(sample, self.weights, self.noise_type, self.noise_prob)
            results.append(out.to(x.device).unsqueeze(0))
        return torch.stack(results)


class EntanglementLayer(nn.Module):
    def __init__(self, noise_type=None, noise_prob=0.0):
        super().__init__()
        self.noise_type = noise_type
        self.noise_prob = noise_prob
        self.scale = nn.Parameter(torch.ones(N_QUBITS))
        self.bias  = nn.Parameter(torch.zeros(N_QUBITS))

    def forward(self, x):
        results = []
        for sample in x:
            out = entanglement_circuit(sample, self.noise_type, self.noise_prob)
            stacked = torch.stack(out).to(x.device)
            results.append(stacked)
        out = torch.stack(results).to(x.device)
        return out * self.scale.to(x.device) + self.bias.to(x.device)


class FullVariationalLayer(nn.Module):
    def __init__(self, n_layers=N_LAYERS, noise_type=None, noise_prob=0.0):
        super().__init__()
        self.noise_type = noise_type
        self.noise_prob = noise_prob
        w = torch.empty(n_layers, N_QUBITS, 3)
        nn.init.uniform_(w, -np.pi/4, np.pi/4)
        self.weights = nn.Parameter(w)

    def forward(self, x):
        results = []
        for sample in x:
            out = full_variational_circuit(sample, self.weights, self.noise_type, self.noise_prob)
            stacked = torch.stack(out).to(x.device)
            results.append(stacked)
        return torch.stack(results)


# ============================================================
# HYBRID MODEL
# ============================================================
class HybridModel(nn.Module):
    def __init__(self, config="full_variational", noise_type=None, noise_prob=0.0):
        super().__init__()
        self.config     = config
        self.noise_type = noise_type
        self.noise_prob = noise_prob

        densenet           = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.features      = densenet.features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        for param in self.features.parameters():
            param.requires_grad = False

        if config == "single_qubit":
            self.pre_quantum = nn.Sequential(
                nn.Linear(1024, 128), nn.BatchNorm1d(128),
                nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 1), nn.Tanh()
            )
            self.quantum   = SingleQubitLayer(N_LAYERS, noise_type, noise_prob)
            q_out_size = 1

        elif config == "entanglement":
            self.pre_quantum = nn.Sequential(
                nn.Linear(1024, 128), nn.BatchNorm1d(128),
                nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, N_QUBITS), nn.Tanh()
            )
            self.quantum   = EntanglementLayer(noise_type, noise_prob)
            q_out_size = N_QUBITS

        elif config == "full_variational":
            self.pre_quantum = nn.Sequential(
                nn.Linear(1024, 128), nn.BatchNorm1d(128),
                nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, N_QUBITS), nn.Tanh()
            )
            self.quantum   = FullVariationalLayer(N_LAYERS, noise_type, noise_prob)
            q_out_size = N_QUBITS

        self.post_quantum = nn.Sequential(
            nn.Linear(q_out_size, 32), nn.BatchNorm1d(32),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        self.bypass = nn.Linear(1024, 1)
        self.alpha  = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        feat   = self.features(x)
        feat   = self.adaptive_pool(feat)
        feat   = feat.view(feat.size(0), -1)
        q_in   = self.pre_quantum(feat) * np.pi
        q_out  = self.quantum(q_in)
        q_out  = q_out.float()
        pq_out = self.post_quantum(q_out)
        bypass = self.bypass(feat)
        alpha  = torch.sigmoid(self.alpha)
        return alpha * pq_out + (1 - alpha) * bypass

    def unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True