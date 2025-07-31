from modules.Optimization_3_qubits import Qubits3Model, optimize_for_T
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
T_list = np.linspace(0, 20.0, 200)  # T values from 0.01 to 10.0
psi_in = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.complex128, device=device)

for method in ['Adam', 'LBFGS']:
    count = 0
    theta = []
    phi = []
    loss = []
    convergence_list = []
    
    
    for T in T_list:
        count = count + 1
        time_current = time.time()
        
        print(f"Optimizing for T = {T:.2f},method = {method} ,proccess:{count}/{len(T_list)} , Time elapsed: {time_current - time_start:.2f} seconds")
        model = Qubits3Model(N=399, dim=6, Omega_max=1.0)
        loss_val, theta_opt, phi_opt, convergence = optimize_for_T(T, model, psi_in, method=method)
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
    df.to_csv(os.path.join(script_dir,"Output", method+"T=1to20_3qubits.csv"))



# dram a graph of loss vs T of 2 methods

df_Adam = pd.read_csv(os.path.join(script_dir,"Output", "AdamT=1to20_3qubits.csv"))
df_LBFGS = pd.read_csv(os.path.join(script_dir,"Output", "LBFGST=1to20_3qubits.csv"))



loss_Adam = df_Adam['loss'].to_numpy()
loss_LBFGS = df_LBFGS['loss'].to_numpy()
loss = np.minimum(loss_Adam, loss_LBFGS)

convergence_Adam = df_Adam['convergence'].to_numpy()
convergence_LBFGS = df_LBFGS['convergence'].to_numpy()
convergence = np.maximum(convergence_Adam, convergence_LBFGS)

plt.figure(figsize=(10, 6))
plt.plot(T_list, loss_Adam, label='Adam Loss', color='green')
plt.plot(T_list, loss_LBFGS, label='LBFGS Loss', color='orange')
plt.plot(T_list, loss, label='Minimum Loss', color='blue')

colors = np.where(convergence == 1, 'red', 'blue')
plt.scatter(T_list, loss, c=colors, marker='o', label='Loss Points')

plt.xlabel('T (Time)')
plt.ylabel('Loss Value')
plt.title('Loss vs T for Adam and LBFGS Methods')
plt.legend()
plt.grid()

plt.savefig(os.path.join(script_dir,"Output","T=1to20_3qubits_MethodComparision.png"))


print(f"Optimization completed, Total time taken: {time.time() - time_start:.2f} seconds\n")