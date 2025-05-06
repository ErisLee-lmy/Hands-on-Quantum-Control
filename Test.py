from modules.Optimization_2_qubits import Optimization_2_qubits

T_test = 25.0
sample = Optimization_2_qubits(T_test) # Create an instance of the class
result = sample.optimize_test(total  = 100 , hist = True, save= True) 