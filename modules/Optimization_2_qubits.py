# from qutip import * # qutip == 5.0.4
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")
import os
# test

method = 'L-BFGS-B' # Optimization method
N = 99 # number of time steps
dim = 4 
# basis_state = [basis(dim,i) for i in range(dim)] # basis states

basis_state = np.zeros((dim, dim), dtype=complex) # Initialize basis states as a 2D array
# in the basis of |10>, |0r>, |11>, |W>=(|1r>+|r1>)/sqrt2
for i in range(dim): 
    basis_state[i, i] = 1 # Set diagonal elements to 1 for basis states


sqrt2 = np.sqrt(2)

Omega_max = 1.0   # fix maximum Omega
T_max = 1000.0 # maximum time
T_list = np.linspace(0, T_max, 100)

# Initialize the state
# psi_in = Qobj([1,0,1,0]) # |psi(0)> = |1>|0> + |1>|1> 
psi_in = np.array([1, 0, 1, 0]) # Convert to numpy array for consistency


class qubits2:
    # input T get optimal 1-F and optimized phi
    def __init__(self, T, N =99, psi_in = psi_in, dim = 4, Omega_max = Omega_max, method ='BFGS'):
        self.T = T
        self.N = N
        
        self.Omega_max = Omega_max
        self.dim = dim
        self.method = method
        self.psi_in = psi_in
        
        self.parameter = np.random.uniform(0, 2*np.pi, N+1) # Random initial parameters

        
    def get_H(self):
        self.phi = self.parameter[1:]
        phi = self.phi
        
        Omega = self.Omega_max * np.exp(1j*phi)# Global Pulse so Omega_1 = Omega_2 = Omega

        H = np.zeros((4,4)) # Hamiltonian matrix
        # in the basis of |01>, |0r>,|11>,|W>=(|1r>+|r1>)/sqr2
        H_list = [] # List to store Hamiltonian matrices
        
        for i in range(N):
            H[0,1] = Omega[i]/2 
            H[1,0] = Omega[i].conj()/2
            H[2,3] = Omega[i]/sqrt2
            H[3,2] = Omega[i].conj()/sqrt2
            
            # H_operator = Qobj(H) # Convert to Qobj
            # H_list.append(H_operator) # Append to the list
            H_list.append(H) # Append the Hamiltonian matrix 
        return H_list # Return the list of Hamiltonian matrices

    def get_U(self):
        dt = self.T / self.N # Time step size

        # Define the unitary operator
        H_list = self.get_H() # Get the Hamiltonian list
        U_total = np.eye(self.dim) # Initialize the total unitary operator
        
        for i in range(len(H_list)):
            H_i = H_list[i]               # Calculate the Hamiltonian for the current time step
            U_i = expm(-1j * H_i * dt) 
            U_total = U_i @ U_total
            
        
        return U_total # Return the total unitary operator

    def get_fidelity(self):
        
        self.theta = self.parameter[0] 
        
        # use the average fidelity formula
        U = self.get_U()
        
        psi_in = self.psi_in
        theta = self.theta
        psi_f = U @ psi_in # Final state after applying the unitary operator
        
        a01 = np.exp(-1j*theta) * basis_state[0].conj() @ psi_f
        a11 = np.exp(-1j * (2*theta + np.pi) ) * basis_state[2].conj() @ psi_f
        


        F = (0.05) * ( np.abs(1 + 2*a01 + a11)**2 + 1 + 2*np.abs(a01)**2 + np.abs(a11)**2 )
        return F # Return the fidelity

    def loss(self, parameter):
        self.parameter = parameter # Update the parameters
        self.theta = parameter[0] # Update theta
        self.phi = parameter[1:] # Update phi
        return 1-self.get_fidelity()
    
    def optimize(self):
        # Optimize the phase angles
        result = minimize(self.loss, self.parameter, method = self.method, options={'disp': False})
        return result.fun, result.x
    
    


if __name__ == "__main__":
    # Test the optimization
    import time
    start_time = time.time()
    for T in [0.01, 1.0, 5.0, 10.0, 100.0, 1000.0]:
        print(f"Optimizing for T = {T:.2f}")
        T_sample = 0.01
        total = 10
        sample = qubits2(T_sample)
        reuslt = sample.optimize()
        print(f"Optimized Loss: {reuslt[0]:.4f}")
        print(f"Optimized Theta: {reuslt[1][0]:.4f}")
        print(f"Current time: {time.time() - start_time:.2f} seconds")
