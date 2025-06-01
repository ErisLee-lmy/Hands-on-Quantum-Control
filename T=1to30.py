from modules.Optimization_2_qubits import qubits2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import warnings
warnings.filterwarnings("ignore")

import time

T_list = np.linspace(0, 30, 30)  # Time steps from 1 to 30
loss_list = []  # List to store losses

sample = qubits2(T_list[0])  # Create an instance with the first T value

start_time = time.time()  # Start timer

for T in T_list:
    print(f"Processing T = {T:.2f}")
    
    sample.T = T  # Update the time T in the instance
    result = sample.optimize()  # Perform optimization
    loss = result[0]
    current_time = time.time() - start_time
    
    print(f"Loss for T = {T:.2f}: {loss:.4f}, Time taken: {current_time:.2f} seconds")
    loss_list.append(loss)
    
plt.figure(figsize=(10, 6))
plt.plot(T_list, loss_list, marker='o', linestyle='-', color='b')
plt.title('1-Fidelity vs Time T (1 to 30)')
plt.xlabel('T')
plt.ylabel('1-Fidelity')
plt.grid()
plt.savefig(os.path.join('Output', 'T1to30.png'))

# Save the results to a CSV file
output_file = os.path.join('Output', 'T1to30.csv')
os.makedirs('Output', exist_ok=True)  # Ensure the output directory exists

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['T', 'Loss'])  # Write header
    for T, loss in zip(T_list, loss_list):
        writer.writerow([f"{T:.3f}", f"{loss:.5f}"])

print(f"Results saved to {output_file}")

plt.show()
