from modules.Optimization_2_qubits import qubits2
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
import time
import csv

T_list = np.linspace(0, 10, 100)  # Time steps
loss_list = []  # List to store losses
satrt_time = time.time()  # Start time for performance measurement

sample = qubits2(T_list[0])  # Number of samples
sample.method = 'L-BFGS-B'  # Optimization method
repeat = 10


for T in T_list:
    
    sample.T = T  # Update time
    loss_best = 1.0  # Initialize best loss
    parameter_best = np.zeros(sample.N + 1)  # Initialize best parameters
    
    
    for _ in range(repeat):
        print(f"Starting optimization for T={T} with repeat {_ + 1}/{repeat}")
        # Reset parameters for each repeat
        sample.parameter = np.random.uniform(0, 2 * np.pi, sample.N + 1)
        # Run optimization
        result = sample.optimize()
        if result[0] < loss_best:
            loss_best = result[0]
            parameter_best = result[1]
            print(f"New best loss for T={T}: {loss_best:.4f}, time: {time.time() - satrt_time:.2f}s")
        else:
            print(f"Loss did not improve for T={T}, current best loss: {loss_best:.4f}, time: {time.time() - satrt_time:.2f}s")
    
    loss_list.append(loss_best)  # Append best loss for this T
    sample.parameter = parameter_best  # Update sample parameters to best found
    print(f"Best loss for T={T}: {loss_best:.4f}, total time: {(time.time() - satrt_time):.2f}s")
    
    
# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(T_list, loss_list, marker='o', linestyle='-', color='b')
plt.title('1-F vs Time (T)')
plt.xlabel('Time (T)')
plt.ylabel('1-Fidelity')
plt.grid()

plt.savefig(os.path.join('Output', 'T1to10.png'))

# Save the results to a CSV file
output_file = os.path.join('Output', 'T1to10.csv')
os.makedirs('Output', exist_ok=True)  # Ensure the output directory exists

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['T', 'Loss'])  # Write header
    for T, loss in zip(T_list, loss_list):
        writer.writerow([f"{T:.3f}", f"{loss:.5f}"])

print(f"Results saved to {output_file}")

plt.show()
