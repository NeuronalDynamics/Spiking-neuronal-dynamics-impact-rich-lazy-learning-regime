import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg 
import math
import requests
import pickle
import gzip
import math
import torch
import pickle
import sys


# function to generate a random orthogonal matrix
def rvs(dim=3):
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim-n+1,))
        D[n-1] = np.sign(x[0])
        x[0] -= D[n-1]*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
    # Fix the last sign such that the determinant is 1
    D[-1] = (-1)**(1-(dim % 2))*D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D*H.T).T
    return H

def u_rvs(dim=3):
    N=dim
    G = np.random.normal(0,g/np.sqrt(N),[N,N])
    P= scipy.linalg.schur(G, output="complex")[1]
    return P

"""
def calculateLEs(model, k, num_steps=1000):
    h_0, _, _, _ = model.init_states(batch_size=1)
    print(h_0)
    h_0 += 0.01 * torch.randn(h_0.size()).to(h_0.device)  # Introduce small perturbations

    input_weights, recurrent_weights = model.get_weights()

    Q = torch.linalg.qr(torch.randn(252, k))[0].to(h_0.device)  # Initial orthogonal matrix

    initial_input = torch.randn(28, 1).to(h_0.device)  # Simulate initial input
    if input_weights.shape[1] != 28:
        input_weights = input_weights.T
    x = input_weights @ initial_input
    x = recurrent_weights @ x + h_0.view(-1, 1)  # Apply initial recurrent processing

    print("Shape of x after initial processing:", x.shape)

    rvals = []
    for t in range(num_steps):
        #print("Shape of x at step", t, ":", x.shape)

        # Apply recurrent effect and spike function
        recurrent_effect = recurrent_weights @ x
        next_x = model.spike_fn(recurrent_effect.squeeze()).view(-1, 1)  # Reshape after spike function

        # Derivative for Jacobian
        v = model.spike_fn_derivative(recurrent_effect.squeeze())
        A = torch.diag_embed(v) @ recurrent_weights

        # QR decomposition
        Z = A @ Q
        Q, R = torch.linalg.qr(Z)

        if R.dim() == 3:
            diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
        else:
            diag_R = torch.diag(R)

        # Handling Small Values: When calculating torch.log(torch.abs(diag_R)), consider adding a small epsilon inside the absolute to avoid taking log of zero, which can result in -inf, affecting the mean calculation:
        eps = 1e-10
        rvals.append(torch.log(torch.abs(diag_R) + eps))
        #rvals.append(torch.log(torch.abs(diag_R)))
        x = next_x  # Update x with correct reshaping


    rvals = torch.stack(rvals)
    LEs = torch.mean(rvals, dim=0)
    return LEs.detach().numpy()

def calculateLEs(model, k, num_steps=1000):
    h_0, _, _, _ = model.init_states(batch_size=1)
    h_0 += 0.01 * torch.randn(h_0.size()).to(h_0.device)  # Small perturbations

    input_weights, recurrent_weights = model.get_weights()

    Q = torch.linalg.qr(torch.randn(h_0.size(0), k))[0].to(h_0.device)  # Proper initial orthogonal matrix

    initial_input = torch.randn(h_0.size(0), 1).to(h_0.device)  # Adjusted initial input
    if input_weights.shape[1] != h_0.size(0):
        input_weights = input_weights.T
    x = input_weights @ initial_input + h_0  # Apply initial recurrent processing

    rvals = []
    for t in range(num_steps):
        recurrent_effect = recurrent_weights @ x
        next_x = model.spike_fn(recurrent_effect).view(-1, 1)

        v = model.spike_fn_derivative(recurrent_effect)
        A = torch.diag_embed(v) @ recurrent_weights

        Z = A @ Q
        Q, R = torch.linalg.qr(Z)

        eps = 0.01  # Increase the epsilon value to see if it helps
        diag_R = torch.diagonal(R, dim1=-2, dim2=-1).abs()

        rvals.append(torch.log(diag_R + eps))
        x = next_x

    rvals = torch.stack(rvals)
    LEs = torch.mean(rvals, dim=0)
    return LEs.numpy()

# Assume `calculateLEs` function call and model setup remains as previously defined






def calculateLEs(model, k, num_steps=1000):
    h_0, _, _, _ = model.init_states(batch_size=1)
    h_0 += 0.01 * torch.randn(h_0.size()).to(h_0.device)  # Ensure dtype is double
    h_0 = h_0.T
    input_weights, recurrent_weights = model.get_weights()
    Q = torch.linalg.qr(torch.randn(2, k))[0]  # Ensure dtype is double #252

    initial_input = torch.randn(2, 1)  # Ensure dtype is double #28
    if input_weights.shape[1] != 2: #28
        input_weights = input_weights.T
    x = input_weights @ initial_input
    x = recurrent_weights @ x + h_0.view(-1, 1)  # Apply initial recurrent processing

    rvals = []
    for t in range(num_steps):
        recurrent_effect = recurrent_weights @ x
        next_h = model.spike_fn(recurrent_effect)
        v = model.spike_fn_derivative(recurrent_effect)
        A = v @ recurrent_weights
        Z = A @ Q
        Q, R = torch.linalg.qr(Z)
        diag_R = torch.diag(R)
        eps = 1e-8
        rvals.append(torch.log(torch.abs(diag_R) + eps))

    rvals = torch.stack(rvals)
    LEs = torch.mean(rvals, dim=0)
    return LEs.detach().numpy()






def calculateLEs(model, k, num_steps=1000):
    h_0, _, _, _ = model.init_states(batch_size=1)
    h_0 += 0.01 * torch.randn(h_0.size(), dtype=torch.float64).to(h_0.device)  # Ensure dtype is double
    h_0 = h_0.T
    input_weights, recurrent_weights = model.get_weights()
    Q = torch.linalg.qr(torch.randn(2, k, dtype=torch.float64))[0]  # Ensure dtype is double #252

    initial_input = torch.randn(2, 1, dtype=torch.float64)  # Ensure dtype is double #28
    if input_weights.shape[1] != 2: #28
        input_weights = input_weights.T
    x = input_weights @ initial_input
    x = recurrent_weights @ x + h_0.view(-1, 1)  # Apply initial recurrent processing

    rvals = []
    for t in range(num_steps):
        recurrent_effect = recurrent_weights @ x
        next_h = model.spike_fn(recurrent_effect)
        v = model.spike_fn_derivative(recurrent_effect)
        A = v @ recurrent_weights
        Z = A @ Q
        Q, R = torch.linalg.qr(Z)
        diag_R = torch.diag(R)
        eps = 1e-8
        rvals.append(torch.log(torch.abs(diag_R) + eps))

    rvals = torch.stack(rvals)
    LEs = torch.mean(rvals, dim=0)
    return LEs.detach().numpy()




def calculateLEs(model, k, num_steps=1000):
    h_0, _, _, _ = model.init_states(batch_size=1)
    input_weights, recurrent_weights = model.get_weights()

    # Start with an orthogonal matrix Q
    Q = torch.linalg.qr(torch.randn(model.A.shape[0], k, dtype=torch.float64))[0]

    # Main loop for calculating Lyapunov exponents
    rvals = []
    for _ in range(num_steps):
        # Applying the model's spike function and updating the state
        x = model.spike_fn(h_0)
        h_0 = x
        
        # Getting the Jacobian using the derivative of the spike function
        A = model.spike_fn_derivative(x) @ recurrent_weights

        # QR decomposition of the product of A and Q
        Z = A @ Q
        Q, R = torch.linalg.qr(Z)
        
        # Logarithm of the absolute values of the diagonal of R
        rvals.append(torch.log(torch.abs(torch.diag(R))))

    # Calculating the mean of logarithms to find Lyapunov exponents
    rvals = torch.stack(rvals)
    LEs = torch.mean(rvals, dim=0)
    return LEs.numpy()


def calculateLEs(model, k, num_steps=1000):
    h_0, _, _, _ = model.init_states(batch_size=252)
    #h_0 += 0.1 * torch.randn(h_0.size()).to(h_0.device)
    input_weights, recurrent_weights = model.get_weights()

    # Start with an orthogonal matrix Q
    Q = torch.linalg.qr(torch.randn(252, k))[0]

    initial_input = torch.randn(28, 1)
    x = input_weights.T @ initial_input
    x = recurrent_weights @ x + h_0

    # Main loop for calculating Lyapunov exponents
    rvals = []
    for _ in range(num_steps):
        # Applying the model's spike function and updating the state
        x = model.spike_fn(x) #h_0
        #h_0 = x
        
        # Getting the Jacobian using the derivative of the spike function
        A = model.spike_fn_derivative(x) @ recurrent_weights

        # QR decomposition of the product of A and Q
        Z = A @ Q
        Q, R = torch.linalg.qr(Z)
        
        # Logarithm of the absolute values of the diagonal of R
        rvals.append(torch.log(torch.abs(torch.diag(R))))

    # Calculating the mean of logarithms to find Lyapunov exponents
    rvals = torch.stack(rvals)
    LEs = torch.mean(rvals, dim=0)
    return LEs.detach().numpy()

"""

def jacobian_eigenspectrum(model, k, num_steps=1000):
    h_0, _, _, _ = model.init_states(batch_size=252)
    input_weights, recurrent_weights = model.get_weights()

    # Initialize the input
    initial_input = torch.randn(28, 1)
    x = input_weights.T @ initial_input
    x = recurrent_weights @ x + h_0

    # Collect all eigenvalues
    all_eigenvalues = []

    for _ in range(num_steps):
        # Apply the model's spike function
        x = model.spike_fn(x)

        # Compute the Jacobian using the derivative of the spike function
        A = torch.diag(model.spike_fn_derivative(x.squeeze())) @ recurrent_weights

        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvals(A).real
        all_eigenvalues.extend(eigenvalues.detach().cpu().numpy())

    # Concatenate all eigenvalues into a NumPy array and sort
    jacobian_eigenspectrum = np.array(all_eigenvalues)
    
    # Sort the eigenvalues from largest to smallest
    sorted_jacobian_eigenspectrum = np.sort(jacobian_eigenspectrum)[::-1]

    return sorted_jacobian_eigenspectrum



'''
def calculateLEs(model, k, num_steps=1000):
    h_0, _, _, _ = model.init_states(batch_size=252) #252
    #h_0 += 0.1 * torch.randn(h_0.size()).to(h_0.device)
    input_weights, recurrent_weights = model.get_weights()

    # Start with an orthogonal matrix Q
    Q = torch.linalg.qr(torch.randn(2, k, dtype=torch.float64))[0] #252

    initial_input = torch.randn(2, 1, dtype=torch.float64) #252
    x = input_weights.T @ initial_input
    x = recurrent_weights @ x + h_0

    # Main loop for calculating Lyapunov exponents
    rvals = []
    for _ in range(num_steps):
        # Applying the model's spike function and updating the state
        x = model.spike_fn(x) #h_0
        #h_0 = x
        
        # Getting the Jacobian using the derivative of the spike function
        A = model.spike_fn_derivative(x) @ recurrent_weights

        # QR decomposition of the product of A and Q
        Z = A @ Q
        Q, R = torch.linalg.qr(Z)
        
        # Logarithm of the absolute values of the diagonal of R
        rvals.append(torch.log(torch.abs(torch.diag(R))))

    # Calculating the mean of logarithms to find Lyapunov exponents
    rvals = torch.stack(rvals)
    LEs = torch.mean(rvals, dim=0)
    sorted_LEs = np.sort(LEs.detach().numpy())[::-1]
    return sorted_LEs
'''

#(env) C:\Users\qiant\Desktop\Research\GLIFS_ASC-main\test\test>python LE_test.py
#Calculated Lyapunov Exponents: [ 0.69209719 -0.69209719]
#Analytical Lyapunov Exponents: [-0.69314718  0.69314718]
#Analytical Lyapunov Exponents: tensor([-0.6931,  0.6931], dtype=torch.float64)

"""
def calculateLEs(model, k, num_steps=1000):
    h_0, _, _, _ = model.init_states(batch_size=252) #252
    #h_0 += 0.1 * torch.randn(h_0.size()).to(h_0.device)
    input_weights, recurrent_weights = model.get_weights()
    #recurrent_weights = recurrent_weights*0.5

    # Start with an orthogonal matrix Q
    Q = torch.linalg.qr(torch.randn(252, k))[0] #252

    initial_input = torch.randn(28, 1) #28
    x = input_weights.T @ initial_input
    x = recurrent_weights @ x + h_0

    # Main loop for calculating Lyapunov exponents
    rvals = []
    for _ in range(num_steps):
        # Applying the model's spike function and updating the state
        x = model.spike_fn(x) #h_0
        #h_0 = x
        
        # Getting the Jacobian using the derivative of the spike function
        A = model.spike_fn_derivative(x) @ recurrent_weights

        # QR decomposition of the product of A and Q
        Z = A @ Q
        Q, R = torch.linalg.qr(Z)
        
        # Logarithm of the absolute values of the diagonal of R
        rvals.append(torch.log(torch.abs(torch.diag(R))))

    # Calculating the mean of logarithms to find Lyapunov exponents
    rvals = torch.stack(rvals)
    LEs = torch.mean(rvals, dim=0)
    sorted_LEs = np.sort(LEs.detach().numpy())[::-1]
    return sorted_LEs
"""
#Calculated Lyapunov Exponents: [ 0.69313688 -0.69313688]
#Analytical Lyapunov Exponents: [-0.69314718  0.69314718]
#Analytical Lyapunov Exponents: tensor([-0.6931,  0.6931], dtype=torch.float64)

"""
def calculateLEs(model, k, num_steps=1000):
    h_0, _, _, _ = model.init_states(batch_size=252) #252 #2
    #h_0, _, _, _ = model.init_states(batch_size=2) #252 #2
    h_0 += 1 * torch.randn(h_0.size()).to(h_0.device)
    input_weights, recurrent_weights = model.get_weights()

    # Start with an orthogonal matrix Q
    Q = torch.linalg.qr(torch.randn(252, k))[0] #252 #2
    #Q = torch.linalg.qr(torch.randn(2, k, dtype=torch.float64))[0] #252

    initial_input = torch.randn(28, 1) #28 #2
    #initial_input = torch.randn(2, 1, dtype=torch.float64) #252
    x = input_weights.T @ initial_input
    x = recurrent_weights @ x + h_0

    # Main loop for calculating Lyapunov exponents
    rvals = []
    jacobian_norms = []
    max_eigenvalues = []
    for _ in range(num_steps):
        # Applying the model's spike function and updating the state
        x = model.spike_fn(x) #h_0
        #h_0 = x
        
        # Getting the Jacobian using the derivative of the spike function
        A = model.spike_fn_derivative(x) @ recurrent_weights

        # Calculate Frobenius norm
        #frob_norm = torch.norm(A, p='fro').item()
        #jacobian_norms.append(frob_norm)
        #print(frob_norm)
        
        # Calculate maximum absolute eigenvalue
        #eigenvalues = torch.linalg.eigvals(A)
        #max_eigen = torch.max(torch.abs(eigenvalues)).item()
        #max_eigenvalues.append(max_eigen)
        #print(max_eigen)

        # QR decomposition of the product of A and Q
        Z = A @ Q
        Q, R = torch.linalg.qr(Z)
        
        # Logarithm of the absolute values of the diagonal of R
        rvals.append(torch.log(torch.abs(torch.diag(R))))

    # Calculating the mean of logarithms to find Lyapunov exponents
    rvals = torch.stack(rvals)
    LEs = torch.mean(rvals, dim=0)
    sorted_LEs = np.sort(LEs.detach().numpy())[::-1]
    return sorted_LEs, jacobian_norms, max_eigenvalues
"""

def calculateLEs(model, num_steps=1000):
    batch_size = 252
    input_sequence = torch.randn(batch_size, num_steps, 28)
    #############################################################################
    #model.reset_state(252)
    #############################################################################
    firing, voltage, ascurrent, syncurrent = model.init_states(batch_size) #252 #2
    #h_0, _, _, _ = model.init_states(batch_size=2) #252 #2
    #print("h_0", h_0.shape)
    #torch.Size([252, 252]) 
    #(batch_size, hidden_size)
    Q = torch.eye(firing.size(dim=1)) 
    #print("Q", Q)

    lyapunov_exponents_sum = torch.zeros(firing.size(dim=0), firing.size(dim=1))
    jacobian_norms = []
    max_eigenvalues = []
    derivative_history = []

    for t in range(num_steps):
        firing, voltage, ascurrent, syncurrent = model.neuron_layer.forward(input_sequence[:, t, :], firing, voltage, ascurrent, syncurrent)
        J = model.spike_fn_derivative(firing)
        derivative_history.append(J[0].detach().numpy())

        # Calculate Frobenius norm
        frob_norm = torch.norm(J[0], p='fro').item()
        jacobian_norms.append(frob_norm)
        #print(frob_norm)
        
        # Calculate maximum absolute eigenvalue
        eigenvalues = torch.linalg.eigvals(J)
        max_eigen = torch.max(torch.abs(eigenvalues)).item()
        max_eigenvalues.append(max_eigen)
        #print(max_eigen)

        Q = J @ Q

        # Perform QR decomposition
        Q, R = torch.linalg.qr(Q)
        lyapunov_exponents_sum += torch.log(torch.abs(torch.diag(R)))

    lyapunov_exponents = lyapunov_exponents_sum / num_steps
    sorted_LEs = np.sort(lyapunov_exponents.mean(dim=0).detach().numpy())[::-1]
    return sorted_LEs, jacobian_norms, max_eigenvalues, derivative_history