import time
from modules.Optimization_2_qubits import Optimization_2_qubits
import numpy as np
import matplotlib.pyplot as plt

repeat =  10
start_time1 = time.time()
T_sample = 25.0 
sample = Optimization_2_qubits(T_sample) # Create an instance of the class
results1 = sample.repeat_optimize(repeat)
end_time1 = time.time()
print(f"Time taken for repeat_optimize: {end_time1 - start_time1:.2f} seconds")
print(f"Results from repeat_optimize: {results1[0]:.4f}")

start_time2 = time.time()
results2 = sample.parallel_repeat_optimize(rrepeat)
end_time2 = time.time()
print(f"Time taken for parallel_repeat_optimize: {end_time2 - start_time2:.2f} seconds")
print(f"Results from parallel_repeat_optimize: {results2[0]:.4f}")


