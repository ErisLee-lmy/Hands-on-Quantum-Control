from qutip import * # qutip == 5.0.4
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

N = 99 # number of time steps
dim = 4 
basis_state = [basis(dim,i) for i in range(dim)] # basis states
# in the basis of |10>, |0r>, |11>, |W>=(|1r>+|r1>)/sqrt2

method = 'BFGS'
sqrt2 = np.sqrt(2)

Omega_max = 1 # fix maximum Omega
T_max = 1000.0 # maximum time
T_list = np.linspace(0, T_max, 100)

# Initialize the state
psi_in = Qobj([1,0,1,0]/sqrt2) # |psi(0)> = |1>|0> + |1>|1> 

class Optimalization_2_qubits:
    # input T get optimal 1-F and optimized phi
    def __init__(self, T):
        self.T = T
        self.dt = T/N
        self.phi = np.random.uniform(0, 2*np.pi, N) # Random initial phase angles
        
    def get_H(self):
        phi = self.phi
        
        Omega = Omega_max * np.exp(1j*phi)# Global Pulse so Omega_1 = Omega_2 = Omega

        H = np.zeros((4,4)) # Hamiltonian matrix
        # in the basis of |01>, |0r>,|11>,|W>=(|1r>+|r1>)/sqr2
        H_list = [] # List to store Hamiltonian matrices
        
        for i in range(N):
            H[0,1] = Omega[i]/2 
            H[1,0] = Omega[i].conj()/2
            H[2,3] = Omega[i]/sqrt2
            H[3,2] = Omega[i].conj()/sqrt2
            
            H_operator = Qobj(H) # Convert to Qobj
            H_list.append(H_operator) # Append to the list
        return H_list # Return the list of Hamiltonian matrices

    def get_U(self):
        dt = self.dt
        phi = self.phi
        # Define the unitary operator
        H_list = self.get_H() # Get the Hamiltonian list
        U_total = qeye(H_list[0].dims[0]) # Initialize the total unitary operator
        for i in range(len(H_list)):
            H_i = H_list[i]               # Calculate the Hamiltonian for the current time step
            U_i = (-1j * H_i * dt).expm() 
            U_total = U_i * U_total
        return U_total # Return the total unitary operator

    def get_fidelity(self):
        
        # use the average fidelity fomula
        U = self.get_U()
        
        # a01 = basis_state[0].dag() * U * psi_in 
        # a11 = -basis_state[2].dag() * U * psi_in 
        a01 = basis_state[0].dag() * U * basis_state[0]
        a11 = -basis_state[2].dag() * U * basis_state[2]

        F = (0.05) * ( np.abs(1 + 2*a01 + a11)**2 + 1 + 2*np.abs(a01)**2 + np.abs(a11)**2 )
        return F # Return the fidelity

    def objective_function(self, phi):
        self.phi = phi # Update phi
        return 1-self.get_fidelity()
    
    def optimize(self):
        # Optimize the phase angles
        result = minimize(self.objective_function, self.phi, method='BFGS', options={'disp': False})
        return result.fun, result.x
    
    def repeat_optimize(self,num=10):
        # Repeat the optimization process to find the best fidelity
        best_result = 1.0
        best_phi = np.zeros(N)
        for i in range(10):
            self.phi = np.random.uniform(0, 2*np.pi, N)
            result = self.optimize()
            if result[0] < best_result:
                best_result = result[0]
                best_phi = result[1]
        return best_result, best_phi
    
def optimize_test(T_sample,total):
    # try to find the optimal phi from several random phi and get the best result

    results = []
    count_optimal = 0
    count = 0

    for i in range(total):
        count += 1
        sample = Optimalization_2_qubits(T_sample) # Create an instance of the class
        result = sample.optimize() # Optimize the phase angles
        if result[0] < 0.1:
            count_optimal += 1
        results.append(result[0]) # Append the minimum fidelity to the results list
        print(f" progress:{count}/{total}, probility: {count_optimal/count},minimum fidelity: {np.min(results)}")
        

    plt.hist(results, bins=10, density=False, alpha=0.7,rwidth=0.8 ,color='blue',label = 'toltal = 10')
    plt.title('Frequency distribution histogramof 1-F')
    plt.ylabel('Frequency')
    plt.xlabel('1-F')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    T_sample = 25.0
    total = 10
    optimize_test(T_sample,total) # Test the optimization process
    