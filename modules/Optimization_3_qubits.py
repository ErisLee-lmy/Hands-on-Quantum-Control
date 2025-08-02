import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Configuration
# method = 'Adam'   
# Optimizer will be chosen later
N = 399              # Number of time steps
dim = 6             # Hilbert space dimension
Omega_max = 1.0     # Maximum pulse amplitude
T_list = [0.1, 1.0, 5.0, 10.0, 100.0]  # List of T values to test


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Basis states for fidelity projection on the selected device
basis_state = torch.eye(dim, dtype=torch.complex128, device=device)
# |001>,|00r〉,|011〉,|0〉⊗|W〉,|111〉,|W1〉
sqrt2 = np.sqrt(2)
sqrt3 = np.sqrt(3)
fidelity_factor = 1/72

class Qubits3Model(nn.Module):
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
        H[:, 0,1] = Omega *0.5
        H[:, 1,0] = Omega.conj() *0.5
        H[:, 2,3] = Omega * sqrt2  * 0.5 
        H[:, 3,2] = Omega.conj() * sqrt2 * 0.5
        H[:, 4,5] = Omega * sqrt3 * 0.5
        H[:, 5,4] = Omega.conj() * sqrt3 * 0.5

        # Accelerated propagation using batched matrix exponentials
        H_scaled = -1j * H * dt  # Scale Hamiltonians
        U_steps = torch.linalg.matrix_exp(H_scaled)  # Compute exponentials in batch (torch>=2.0)

        # Sequentially multiply using torch.matmul and cumprod-like reduction
        U = torch.eye(self.dim, dtype=torch.complex128, device=device)
        for U_step in U_steps:
            U = U_step @ U

        psi_f = U @ psi_in

        # Compute fidelity amplitudes
        a001 = torch.exp(-1j * theta) * torch.dot(basis_state[0].conj(), psi_f)
        a011 = torch.exp(-1j * 2 * theta) * torch.dot(basis_state[2].conj(), psi_f)
        a111 = torch.exp(-1j * (3 * theta + np.pi)) * torch.dot(basis_state[2].conj(), psi_f)
        # Average fidelity and clamp to [0,1]
        F = fidelity_factor * (torch.abs(1 + 3 * a001 + 3 * a011 + a111)**2 + 1 + 3 * torch.abs(a001)**2 + 3 * torch.abs(a011)**2 + torch.abs(a111)**2)
        F = torch.clamp(F, 0.0, 1.0)
        return F

    def loss(self, T, psi_in):
        return 1 - self.forward(T, psi_in)


def optimize_for_T(T, model, psi_in,tol_loss=1e-6, tol_grad=1e-6,
                   method = 'Adam', lr_adam=1e-1, max_iter_adam=10000, 
                   lr_LBFGS=1, inner_iter_LBFGS=20, max_outer_iter_LBFGS=100):
    
    prev_loss = 1.0
    convergence = 0
    
    if method == 'Adam':
        lr = lr_adam
        max_iter = max_iter_adam
        # Use Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        print(f"Optimizing with Adam: T={T:.2f}, lr = {lr}, max_iter = {max_iter}")
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
            grad_norm = torch.norm(torch.cat([p.grad.flatten() for p in model.parameters()]))
            
            if loss.item()< tol_loss or grad_norm < tol_grad:
                print(f"Converged at outer_iter {i}, loss={loss.item():.6e}, grad_norm={grad_norm:.3e}")
                convergence = 1
                break
            
            prev_loss = loss.item()

        if not convergence:
            print(f"Warning: Did not converge within {max_iter} iterations")

        
        print(f"  Final Loss: {final_loss:.4f}, Theta end: {model.theta.item():.4f}")
        return final_loss, model.theta.item(), model.phi.detach().cpu().numpy(),convergence
    
    elif method == 'LBFGS':
        max_iter = max_outer_iter_LBFGS
        lr = lr_LBFGS
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=inner_iter_LBFGS, line_search_fn="strong_wolfe")

        print(f"Optimizing with LBFGS: T={T:.2f}, max_iter = {max_iter}")
        
        print(f"  Initial Loss: {model.loss(T, psi_in).item():.6f}, Theta start: {model.theta.item():.4f}")

        for i in range(max_iter):
            def closure():
                optimizer.zero_grad()
                loss = model.loss(T, psi_in)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
        
        
            grad_norm = torch.norm(torch.cat([p.grad.flatten() for p in model.parameters()]))
                
            if abs(loss.item()) < tol_loss or grad_norm < tol_grad:
                print(f"Converged at outer_iter {i}, loss={loss.item():.6e}, grad_norm={grad_norm:.3e}")
                convergence = 1
                break

            prev_loss = loss.item()


            if i % 5 == 0:
                print(f"    Iter {i}, loss={loss.item():.6e}, theta={model.theta.item():.4f}")
        if not convergence:
            print(f"Warning: Did not converge within {max_iter} iterations")
        
        final_loss = model.loss(T, psi_in).item()
        print(f"  Final Loss: {final_loss:.6f}, Theta end: {model.theta.item():.4f}")
        return final_loss, model.theta.item(), model.phi.detach().cpu().numpy(),convergence


    final_loss = loss.item()
    print(f"  Final Loss: {final_loss:.4f}, Theta end: {model.theta.item():.4f}")
    return final_loss, model.theta.item(), model.phi.detach().cpu().numpy(),convergence,convergence

def deep_optimize_for_T(T, model, psi_in, lr=1e-3, max_iter=20000, tol=1e-10):
    """
    高精度优化：AdamW + CosineAnnealingLR
    - 自动调整学习率
    - 精度目标：loss <= 1e-10
    """
    # 转换为高精度
    model.theta = nn.Parameter(model.theta.data.to(torch.float64))
    model.phi = nn.Parameter(model.phi.data.to(torch.float64))

    # AdamW 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine Annealing 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=1e-6)

    # 初始 loss
    init_loss = model.loss(T, psi_in).item()
    print(f"  [AdamW + CosineAnnealing] Init Loss: {init_loss:.6e}, Theta: {model.theta.item():.4f}")

    # 训练循环
    for i in range(max_iter):
        optimizer.zero_grad()
        loss = model.loss(T, psi_in)
        loss.backward()

        # 梯度范数（收敛判定用）
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e6).item()

        optimizer.step()
        scheduler.step()

        # 打印进度
        if i % 1000 == 0 or loss.item() < tol:
            print(f"    Iter {i}, loss={loss.item():.3e}, grad_norm={grad_norm:.3e}, lr={scheduler.get_last_lr()[0]:.2e}")

        # 收敛判定
        if loss.item() < tol or grad_norm < 1e-12:
            print(f"Converged at iter {i} with loss={loss.item():.3e}")
            return loss.item(), model.theta.item(), model.phi.detach().cpu().numpy()

    # 未收敛情况
    print(f"Warning: Did not reach target precision (final loss={loss.item():.3e})")
    return loss.item(), model.theta.item(), model.phi.detach().cpu().numpy()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using device: {device} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
    else:
        print(f"Using device: {device}")
    # Prepare initial state on device
    psi_in = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.complex128, device=device)
    for T in T_list:
        print(f"Optimizing for T = {T:.2f}")
        model = Qubits3Model(N, dim, Omega_max).to(device)
        loss_val, theta_opt, phi_opt = optimize_for_T(T, model, psi_in)
        print(f"Result: Loss={loss_val:.4f}, Theta={theta_opt:.4f}\n")
