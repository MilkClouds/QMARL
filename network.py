import gym
import numpy as np
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torch.optim.lr_scheduler import CosineAnnealingLR
from  torch.distributions import Categorical
from Environment import ENV
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import pennylane as qml
import pennylane.numpy as np


# q_device = qml.device("lightning.gpu", wires=4, batch_obs=True)
q_device = qml.device("default.qubit", wires=4)
@qml.qnode(q_device)
def circuit_actor(inputs, weights):
    x = inputs
    qml.AngleEmbedding(x, wires=q_device.wires, rotation='Y')
    qml.RandomLayers(weights, wires=q_device.wires)
    qml.PauliX(wires=0)
    qml.PauliY(wires=1)
    qml.PauliZ(wires=3)
    qml.CNOT(wires=(0, 2))
    return [qml.expval(qml.PauliZ(i)) for i in q_device.wires]

@qml.qnode(q_device)
def circuit_critic(inputs, weights):
    x = inputs
    qml.AngleEmbedding(x[:4], wires=q_device.wires, rotation='Y')
    qml.AngleEmbedding(x[4:8], wires=q_device.wires, rotation='Z')
    qml.AngleEmbedding(x[8:12], wires=q_device.wires, rotation='X')
    qml.AngleEmbedding(x[12:], wires=q_device.wires, rotation='Y')
    qml.RandomLayers(weights, wires=q_device.wires)
    qml.PauliX(wires=0)
    qml.PauliY(wires=1)
    qml.PauliZ(wires=3)
    qml.CNOT(wires=(0, 2))
    return [qml.expval(qml.PauliZ(i)) for i in q_device.wires]
class QActor(nn.Module):

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # self.q_layer = QLayer_Actor(self.n_wires, self.q_device)
        weight_shapes = {"weights": qml.RandomLayers.shape(n_layers=10, n_rotations=4)}
        self.q_layer = qml.qnn.TorchLayer(circuit_actor, weight_shapes)

    def forward(self, x, use_qiskit=False):
        # bsz = x.shape[0]
        # x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            # self.encoder(self.q_device, x)
            # self.q_layer(self.q_device)
            # x = self.measure(self.q_device)
            # x = self.q_layer(self.q_device)
            x = self.q_layer(x)

        # x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
        x = F.softmax(x*4, dim=1)

        return x
    

class QCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        weight_shapes = {"weights": qml.RandomLayers.shape(n_layers=10, n_rotations=4)}
        self.q_layer = qml.qnn.TorchLayer(circuit_critic, weight_shapes)

    def forward(self, x, use_qiskit=False):
        # bsz = x.shape[0]
        # x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            x = self.q_layer(x)
            # self.encoder(self.q_device, x)
            # self.q_layer(self.q_device)
            # x = self.measure(self.q_device)
        x = x.sum(-1) * 20
        return x
    
class ReplayBuffer:
    def __init__(self,device):
        self.data = []
        self.device = device
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst = [], [], [], []
        for transition in self.data:
            s, a, r, s_prime = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            
        s,a,r,s_prime = torch.tensor(s_lst, dtype=torch.float).to(self.device), torch.tensor(a_lst, dtype=torch.int64).to(self.device), \
                        torch.tensor(r_lst, dtype=torch.float).to(self.device), torch.tensor(s_prime_lst, dtype=torch.float).to(self.device)
        self.data = []
        return s, a, r, s_prime
        
        
if __name__ == "__main__":
    x = torch.Tensor([[0, 0.1, 0.2, 0.3], [0, 0.1, 0.2, 0.4]])
    act = QActor()
    print(act(x))
    print(act.q_layer)
    print([*act.named_parameters()])