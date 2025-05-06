from modules.Optimization_2_qubits import Optimization_2_qubits
import numpy as np
import matplotlib.pyplot as plt


T_min = 20.0 
T_max = 30,0
T_sample = 25.0
sample = Optimization_2_qubits(T_sample) # Create an instance of the class
sample.accurate_optimize(precision=1e-4,repeat = 3,disp=True)