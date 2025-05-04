from modules.Optimization_2_qubits import Optimalization_2_qubits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

start_time = time.time()

# try to find the critical point of the Fidelity
T_min = 25.0
T_max = 28.0
num_T = 60
# based on the simulation before the critical point is between 25 and 28

repeat = 30 # number of times to repeat the optimization
tolerance = 1e-4 # tolerance for the decreasing

T_list = np.linspace(T_min, T_max, num_T) # Time list




results = []

for T in T_list:
    current_time = time.time()
    print(f"Starting optimization for T = {T:.2f}, program have been running for {current_time-start_time:.2f} seconds, progress: {T_list.tolist().index(T)+1}/{num_T}")
    sample = Optimalization_2_qubits(T) # Create an instance of the class
    if results:
        result_min = np.min(results)
        result_best = 1.0
        count = 0
        while result_best > result_min+tolerance:
            # force the optimization to find the decreasing result
            # if the result is not decreasing, repeat the optimization
            count += 1
            result = sample.repeat_optimize(num = repeat)
            result_temp = result[0]
            print(f"Try to find the decreasing result,best 1-F:{result_best:.4f}, 1-F for last T: {result_min}, Interations: {count}")
            if result_temp < result_best:
                result_best = result_temp
        else:
            results.append(result_best)
            print(f"Find the decreasing result,1-F:{result_best:.4f}, Interations: {count}")
    else:
        result = sample.repeat_optimize(num=repeat) # Optimize the phase angles
        results.append(result[0]) # Append the minimum fidelity to the results list
        print(f"Find the decreasing result,1-F:{result[0]:.4f}")


plt.figure(figsize=(10, 6))
plt.plot(T_list, results, linestyle='-', color='b')
plt.scatter(T_list, results, color='r', label='Results')
plt.xlabel('Control Time (T)')
plt.ylabel('1-Fidelity')
plt.title('1-Fidelity vs Control Time')
plt.grid(True)
plt.legend()

png_name = f"1-F_vs_ControlTime_T_from{int(T_min)}_to_{int(T_max)}_pointnum{int(num_T)}.png"
file_name = os.path.join('Output', png_name)
plt.savefig(file_name, dpi=300, bbox_inches='tight')
print(f"Graph saved as {png_name}")


# Save the results to a CSV file
results_df = pd.DataFrame({'Control Time (T)': T_list, '1-Fidelity': results})
csv_name = f"1-F_vs_ControlTime_T_from{int(T_min)}_to_{int(T_max)}_pointnum{int(num_T)}.csv"
file_name = os.path.join('Output', csv_name)
results_df.to_csv(file_name, index=False)
print(f"Results saved as {csv_name}")
