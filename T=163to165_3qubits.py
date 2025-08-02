from modules.Optimization_3_qubits import Qubits3Model, deep_optimize_for_T
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
T_list = np.linspace(16.3, 30.0, 3)  # T values from 0.01 to 10.0
psi_in = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.complex128, device=device)


count = 0
theta = []
phi = []
loss = []
convergence_list = []


for T in T_list:
    count = count + 1
    time_current = time.time()
    
    print(f"Optimizing for T = {T:.2f} ,proccess:{count}/{len(T_list)} , Time elapsed: {time_current - time_start:.2f} seconds")
    model = Qubits3Model(N=399, dim=6, Omega_max=1.0)
    loss_val, theta_opt, phi_opt, convergence = deep_optimize_for_T(T, model, psi_in)
    print(f"Result: Loss={loss_val:.4f}, Theta={theta_opt:.4f}\n")
    
    theta.append(theta_opt)
    phi.append(phi_opt)
    loss.append(loss_val)
    convergence_list.append(convergence)
        
    # plt.figure(figsize=(10, 6))
    # plt.plot(T_list, loss, label='Loss')
    # plt.scatter(T_list, loss, color='red', marker='o', label='Loss Points')
    # plt.xlabel('T (Time)')
    # plt.ylabel('Loss Value')
    # plt.title('Loss vs T')
    # plt.grid()
    
    # plt.savefig(os.path.join(script_dir,"Output",method+ "T=1to20_3qubits.png"))


data = np.hstack([T_list.reshape(-1, 1), np.array(loss).reshape(-1, 1), np.array(theta).reshape(-1, 1), np.array(convergence_list).reshape(-1,1),np.array(phi)])
columns = ['T', 'loss','theta','convergence'] + [f'phi{i+1}' for i in range(model.N)]
df = pd.DataFrame(data, columns=columns)
df.to_csv(os.path.join(script_dir,"Output", "T=163to165_3qubits.csv"))



# dram a graph of loss vs T of 2 methods

df = pd.read_csv(os.path.join(script_dir,"Output", "T=163to165_3qubits.csv"))
loss = df['loss'].to_numpy()
convergence_list = df['convergence'].to_numpy()


plt.figure(figsize=(10, 6))
plt.plot(T_list, loss, label='Minimum Loss', color='blue')

colors = np.where(convergence_list == 1, 'red', 'blue')
plt.scatter(T_list, loss, c=colors, marker='o', label='Loss Points')

plt.xlabel('T (Time)')
plt.ylabel('Loss Value')
plt.title('Loss vs T for 3 Qubits Optimization')
plt.legend()
plt.grid()

plt.savefig(os.path.join(script_dir,"Output","T=163to165_3qubitsn.png"))


print(f"Optimization completed, Total time taken: {time.time() - time_start:.2f} seconds\n")