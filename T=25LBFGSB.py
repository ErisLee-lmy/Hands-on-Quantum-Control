# try to use L-BFGS-B optimization method instead of BFGS for T=25

from modules.Optimization_2_qubits import qubits2
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

import time

T_sample = 25.0
repeat = 20 # Number of repetitions for optimization
loss_list = []  # List to store losses
sample = qubits2(T_sample)
sample.method = 'L-BFGS-B'  # Set optimization method

start_time = time.time()

for i in range(repeat):
    print(f"Iteration {i+1}/{repeat}")
    sample.parameter = np.random.uniform(0, 2*np.pi, sample.N+1)  # Random initial parameters for each iteration
    
    result = sample.optimize()  # Perform optimization
    loss = result[0]
    loss_list.append(loss)
    print(f"Loss for iteration {i+1}: {loss:.4f}, Time taken: {time.time() - start_time:.2f} seconds")
    
plt.figure(figsize=(10, 6))
plt.hist(loss_list, bins=20, color='blue', alpha=0.7)
plt.title('Distribution of 1-Fidelity Losses')
plt.xlabel('1-Fidelity Loss')
plt.ylabel('Frequency')
plt.grid()

plt.savefig(os.path.join('Output', 'T25_Distribution_LBFGSB.png'))

plt.show()