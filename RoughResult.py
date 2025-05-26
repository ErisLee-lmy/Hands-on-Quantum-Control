from modules.Optimization_2_qubits import qubits2
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
import time

T_list = np.linspace(0, 10, 100)  # Time steps
loss_list = []  # List to store losses
count = 0 
start_time = time.time()  # Start timer

for T in T_list:
    count += 1
    print(f"Processing T = {T:.2f} ({count}/{len(T_list)})")
    
    # Create an instance of the qubits2 class
    qubit_instance = qubits2(T)
    
    result = qubit_instance.optimize()  # Perform optimization
    loss = result[0]
    curret_time = time.time() - start_time
    print(f"Loss for T = {T:.2f}: {loss:.4f}, Time taken: {curret_time:.2f} seconds")
    loss_list.append(loss)
    
    
plt.figure(figsize=(10, 6))
plt.plot(T_list, loss_list, marker='o', linestyle='-', color='b')
plt.title('1-Fidelity vs Time T')
plt.xlabel('T')
plt.ylabel('1-Fidelity')
plt.grid()
plt.savefig(os.path.join('Output', 'RoughResult.png'))
plt.show()