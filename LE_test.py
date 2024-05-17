import torch
import numpy as np
from lyapunov.LEs_1 import calculateLEs
from torch.nn.parameter import Parameter
"""
class SimpleLinearModel:
    def __init__(self, A):
        self.A = A.double()  # Use torch.float64 for better precision

    def init_states(self, batch_size=1):
        # Initial state with batch size and consistent data type
        return torch.zeros(self.A.shape[0], batch_size, dtype=torch.float64), None, None, None

    def get_weights(self):
        # This model uses the same weights for input and recurrence for simplicity
        return self.A, self.A

    def spike_fn(self, x):
        # Linear activation for simplicity
        return x

    def spike_fn_derivative(self, x):
        # Derivative is simply the weights matrix itself since the activation is linear
        return torch.eye(self.A.shape[0], dtype=torch.float64)

def analytical_LEs(A):
    eigenvalues = torch.linalg.eigvals(A)
    return torch.log(torch.abs(eigenvalues))

# Define a simple matrix A with known eigenvalues
A = torch.tensor([[0.5, 0.0], [0.0, 2.0]], dtype=torch.float64)

# Instantiate the model
model = SimpleLinearModel(A)

# Calculate Lyapunov exponents using the existing function
calculated_LEs, _, _ = calculateLEs(model, k=2, num_steps=100)


# Print results
print("Calculated Lyapunov Exponents:", calculated_LEs)
print("Analytical Lyapunov Exponents:", np.log(np.abs([0.5, 2.0])))
print("Analytical Lyapunov Exponents:", analytical_LEs(A))
"""

#(env) C:\Users\qiant\Desktop\Research\GLIFS_ASC-main\test\test>python LE_test.py
#Calculated Lyapunov Exponents: [ 0.69309103 -0.69309103]
#Analytical Lyapunov Exponents: [-0.69314718  0.69314718]


# Simple linear model for testing purposes
class SimpleLinearModel:
    def __init__(self, A):
        self.A = Parameter(A)  # Linear transformation matrix
        self.hidden_size = A.shape[0]  # Assuming square matrix for simplicity

    def init_states(self, batch_size):
        # Initialize states as zeros
        firing = torch.zeros(batch_size, self.hidden_size, dtype=torch.float64)
        voltage = torch.zeros_like(firing)
        ascurrent = torch.zeros_like(firing)
        syncurrent = torch.zeros_like(firing)
        return firing, voltage, ascurrent, syncurrent

    def forward(self, x, firing, voltage, ascurrent, syncurrent):
        # Linear transformation without any nonlinearity
        firing = self.A @ x.transpose(0, 1)
        return firing.transpose(0, 1), voltage, ascurrent, syncurrent

    def spike_fn_derivative(self, firing):
        # Derivative of the spike function, which is just the linear transformation matrix in this case
        return self.A

# Define a simple matrix A with known dynamics (for example, a diagonal matrix)
A = torch.tensor([[0.5, 0.0], [0.0, 2.0]], dtype=torch.float64)

# Instantiate the simple linear model
model = SimpleLinearModel(A)

# Run the test
lyapunov_exponents, jacobian_norms, max_eigenvalues = calculateLEs(model, num_steps=100)

# Analytical Lyapunov exponents
analytical_LEs = np.sort(np.log(np.abs(np.diag(A))))[::-1]

# Print the results
print("Calculated Lyapunov Exponents:", lyapunov_exponents)
print("Analytical Lyapunov Exponents:", analytical_LEs)
#print("Jacobian Frobenius Norms:", jacobian_norms)
#print("Jacobian Max Eigenvalues:", max_eigenvalues)

# Optionally, you can also add a check to see if the calculated exponents are close to the analytical ones
np.testing.assert_allclose(lyapunov_exponents, analytical_LEs, atol=1e-5)
print("The calculated Lyapunov exponents match the analytical ones within the tolerance.")
