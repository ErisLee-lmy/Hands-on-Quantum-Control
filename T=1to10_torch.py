from modules.PytorchOptimization_2_qubits import Qubits2Model
import numpy as np
import matplotlib.pyplot as plt
import time
import os

time_start = time.time()

T_list = np.linspace(0.01, 10.0, 100)  # T values from 0.01 to 10.0
