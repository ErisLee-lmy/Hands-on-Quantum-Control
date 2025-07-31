from modules.PytorchOptimization_2_qubits import Qubits2Model,deep_optimize_for_T
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using device: {device} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
else:
    print(f"Using device: {device}")

script_dir = os.path.dirname(os.path.abspath(__file__)) 

time_start = time.time()

T_list = np.linspace(7.5, 7.7, 100)  # T values from 7.5 to 7.7
theta = []
phi = []
loss = []


psi_in = torch.tensor([1, 0, 1, 0], dtype=torch.complex128, device=device)
count = 0
for T in T_list:
    count = count + 1
    time_current = time.time()
    print(f"Optimizing for T = {T:.4f}, proccess:{count}/{len(T_list)} , Time elapsed: {time_current - time_start:.2f} seconds")
    model = Qubits2Model(N=99, dim=4, Omega_max=1.0)
    loss_val, theta_opt, phi_opt = deep_optimize_for_T(T, model, psi_in)
    print(f"Result: Loss={loss_val}, Theta={theta_opt:.4f}\n")
    
    theta.append(theta_opt)
    phi.append(phi_opt)
    loss.append(loss_val)

lg_loss = np.log10(np.array(loss))

plt.figure(figsize=(10, 6))
plt.plot(T_list, lg_loss, label='Lg Loss')
plt.scatter(T_list, lg_loss, color='red', marker='o', label='Lg Loss Points')
plt.xlabel('T (Time)')
plt.ylabel('Lg Loss Value')
plt.title('Lg Loss vs T')
plt.grid()
plt.savefig(os.path.join(script_dir,"Output", "T=75to77_torch.png"))





data = np.hstack([T_list.reshape(-1, 1), np.array(loss).reshape(-1, 1), np.array(theta).reshape(-1, 1), np.array(phi)])
columns = ['T', 'loss','theta'] + [f'phi{i+1}' for i in range(model.N)]
df = pd.DataFrame(data, columns=columns)
df.to_csv(os.path.join(script_dir,"Output", "T=75to77_torch.csv"), index=False)

print(f"Optimization completed, Total time taken: {time.time() - time_start:.2f} seconds")

