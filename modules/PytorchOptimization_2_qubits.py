import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Configuration
method = 'Adam'  # Optimizer choice
N = 99              # Number of time steps
dim = 4             # Hilbert space dimension
Omega_max = 1.0     # Maximum pulse amplitude
T_list = [0.1, 1.0, 5.0, 10.0, 100.0, 1000.0]  # List of T values to test


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Basis states for fidelity projection on the selected device
basis_state = torch.eye(dim, dtype=torch.complex128, device=device)
sqrt2 = np.sqrt(2)


class Qubits2Model(nn.Module):
    def __init__(self, N, dim, Omega_max):
        super().__init__()
        self.N = N
        self.dim = dim
        self.Omega_max = Omega_max
        # Learnable parameters
        self.theta = nn.Parameter(torch.rand(1, device=device) * 2 * np.pi)
        self.phi = nn.Parameter(torch.rand(N, device=device) * 2 * np.pi)

    def forward(self, T, psi_in):
        # Constrain parameters to [0, 2π] range to avoid runaway phases
        theta = torch.remainder(self.theta, 2*np.pi)
        phi = torch.remainder(self.phi, 2*np.pi)

        dt = T / self.N
        # Construct Hamiltonian batch
        Omega = self.Omega_max * torch.exp(1j * phi)
        H = torch.zeros((self.N, self.dim, self.dim), dtype=torch.complex128, device=device)
        H[:, 0,1] = Omega / 2
        H[:, 1,0] = Omega.conj() / 2
        H[:, 2,3] = Omega / sqrt2
        H[:, 3,2] = Omega.conj() / sqrt2

        # Accelerated propagation using batched matrix exponentials
        H_scaled = -1j * H * dt  # Scale Hamiltonians
        U_steps = torch.linalg.matrix_exp(H_scaled)  # Compute exponentials in batch (torch>=2.0)

        # Sequentially multiply using torch.matmul and cumprod-like reduction
        U = torch.eye(self.dim, dtype=torch.complex128, device=device)
        for U_step in U_steps:
            U = U_step @ U

        psi_f = U @ psi_in

        # Compute fidelity amplitudes
        a01 = torch.exp(-1j * theta) * torch.dot(basis_state[0].conj(), psi_f)
        a11 = torch.exp(-1j * (2 * theta + np.pi)) * torch.dot(basis_state[2].conj(), psi_f)
        # Average fidelity and clamp to [0,1]
        F = 0.05 * (torch.abs(1 + 2 * a01 + a11)**2 + 1 + 2 * torch.abs(a01)**2 + torch.abs(a11)**2)
        F = torch.clamp(F, 0.0, 1.0)
        return F

    def loss(self, T, psi_in):
        return 1 - self.forward(T, psi_in)


def optimize_for_T(T, model, psi_in, lr=1e-1, max_iter=2000):
    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initial diagnostics
    init_loss = model.loss(T, psi_in).item()
    print(f"  Initial Loss: {init_loss:.4f}, Theta start: {model.theta.item():.4f}")

    # Optimization loop
    for i in range(max_iter):
        optimizer.zero_grad()
        loss = model.loss(T, psi_in)
        loss.backward()
        optimizer.step()
        if i % 500 == 0:
            print(f"    Iter {i}, loss={loss.item():.4f}, theta={model.theta.item():.4f}")

    final_loss = loss.item()
    print(f"  Final Loss: {final_loss:.4f}, Theta end: {model.theta.item():.4f}")
    return final_loss, model.theta.item(), model.phi.detach().cpu().numpy()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using device: {device} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
    else:
        print(f"Using device: {device}")
    # Prepare initial state on device
    psi_in = torch.tensor([1, 0, 1, 0], dtype=torch.complex128, device=device)
    for T in T_list:
        print(f"Optimizing for T = {T:.2f}")
        model = Qubits2Model(N, dim, Omega_max).to(device)
        loss_val, theta_opt, phi_opt = optimize_for_T(T, model, psi_in)
        print(f"Result: Loss={loss_val:.4f}, Theta={theta_opt:.4f}\n")
