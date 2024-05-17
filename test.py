import torch
import torch.nn as nn
from functorch import make_functional, vmap, vjp, jvp, jacrev
device = 'cuda'

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 32, (3, 3))
        self.conv3 = nn.Conv2d(32, 32, (3, 3))
        self.fc = nn.Linear(21632, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.relu()
        x = self.conv2(x)
        x = x.relu()
        x = self.conv3(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

x_train = torch.randn(20, 3, 32, 32, device=device)
x_test = torch.randn(5, 3, 32, 32, device=device)

net = CNN().to(device)
fnet, params = make_functional(net)

def fnet_single(params, x):
    return fnet(params, x.unsqueeze(0)).squeeze(0)

def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2):
    print(fnet_single)
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    for tensor in jac1:
        print(tensor.shape)
    print("**************")
    jac1 = [j.flatten(2) for j in jac1]
    for tensor in jac1:
        print(tensor.shape)
    
    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = [j.flatten(2) for j in jac2]
    
    # Compute J(x1) @ J(x2).T
    result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result

result = empirical_ntk_jacobian_contraction(fnet_single, params, x_train, x_test)
print("*****************")
print(result.shape)



    net.zero_grad()
    
    # Assuming 'net' accepts an argument 'track' and returns the output you're interested in.
    # Use a lambda function to ensure that 'net' is called with 'x' and 'track=False' within the jacobian computation.
    func_to_differentiate = lambda x: net(x, track=False)
    
    # Compute the Jacobian of the output with respect to the input 'x'
    J = torch.autograd.functional.jacobian(func_to_differentiate, x, create_graph=False)

    # Compute the NTK as the product of the Jacobian with itself, transposed
    J_flattened = J.reshape(J.shape[0], -1)
    
    # Compute the NTK as the dot product of the flattened Jacobian with its transpose
    ntk = torch.matmul(J_flattened, J_flattened.transpose(0, 1))
    return ntk



    def jacobian(y, params):
    """ Compute the Jacobian matrix dy/dx
    Args:
        y: output Tensor with shape (..., M)
        params: iterator of Tensors, network parameters
    Returns:
        Jacobian matrix with shape (..., M, N)
    """
    jac = []  # List to hold the gradients
    # Select only the second parameter for gradient computation
    second_param = [params[1]]  # Assuming params is indexable, otherwise convert it to a list
    for i in range(y.shape[-1]):
        # Create a zero tensor with the same shape as y
        grad_output = torch.zeros_like(y)
        # Set the i-th element to 1
        grad_output[..., i] = 1
        # Compute the gradient with respect to the second parameter
        grad = torch.autograd.grad(y, second_param, grad_outputs=grad_output, retain_graph=True, create_graph=True, allow_unused=True)
        # grad is a tuple of gradients, one for each parameter tensor (only one in this case).
        # Check if the gradient is not None and flatten
        grad = grad[0].flatten() if grad[0] is not None else torch.tensor([])
        jac.append(grad)
    return torch.stack(jac, dim=-2)

def compute_ntk(net, x):    
    """ Compute the Neural Tangent Kernel (NTK) of a network
    Args:
        net: the network
        x: input Tensor
    Returns:
        NTK matrix with shape (..., M, M)
    """
    net.zero_grad()
    _, _, _, _, hidden_activity = net(x)

    # Extract the hidden activity of the last step
    hidden_activity_last_step = hidden_activity[:,-1,:]
    
    # Convert net.parameters() to a list to index it
    params_list = list(net.parameters())
    
    # Compute the Jacobian of the hidden activity of the last step with respect to the second network parameter
    J_hidden_last_step = jacobian(hidden_activity_last_step, params_list)
    
    # Compute the NTK as the product of the Jacobian with itself, transposed
    ntk = torch.einsum('ij,kj->ik', J_hidden_last_step, J_hidden_last_step)
    
    return ntk



def compute_ntk(net, x):
    """ Compute the Neural Tangent Kernel (NTK) of a network
    Args:
        net: the network
        x: input Tensor
    Returns:
        NTK matrix with shape (..., M, M)
    """
    net.zero_grad()
    
    
    #func_to_differentiate = lambda x: net(x, track=False)
    def func_to_differentiate(x):
        _, _, _, _, hidden_activity = net(x)
        # Extract the hidden activity of the last step
        hidden_activity_last_step = hidden_activity[:,-1,:]
        return hidden_activity_last_step

    #params_list = list(net.parameters())
    #second_param = (params_list[1],)
    
    J = torch.autograd.functional.jacobian(func_to_differentiate, x, create_graph=False)

    # Compute the NTK as the product of the Jacobian with itself, transposed
    J_flattened = J.reshape(J.shape[0], -1)
    
    # Compute the NTK as the dot product of the flattened Jacobian with its transpose
    ntk = torch.matmul(J_flattened, J_flattened.transpose(0, 1))
    return ntk

def compute_ntk(net, x):
    """ Compute the Neural Tangent Kernel (NTK) of a network
    Args:
        net: the network
        x: input Tensor
    Returns:
        NTK matrix with shape (..., M, M)
    """
    net.zero_grad()
    
    
    #func_to_differentiate = lambda x: net(x, track=False)
    def func_to_differentiate(param):
        _, _, _, _, hidden_activity = net(x)
        # Extract the hidden activity of the last step
        hidden_activity_last_step = hidden_activity[:,-1,:]
        return hidden_activity_last_step

    params_list = list(net.parameters())
    second_param =  torch.tensor(params_list[1])
    
    J = torch.autograd.functional.jacobian(func_to_differentiate, second_param, create_graph=False)

    # Compute the NTK as the product of the Jacobian with itself, transposed
    #print(J.shape)
    J_flattened = J.reshape(J.shape[0], -1)
    #print(J_flattened.shape)
    
    # Compute the NTK as the dot product of the flattened Jacobian with its transpose
    ntk = torch.matmul(J_flattened, J_flattened.transpose(0, 1))
    return ntk


def jacobian(y, params):
    """ Compute the Jacobian matrix dy/dx
    Args:
        y: output Tensor with shape (..., M)
        params: iterator of Tensors, network parameters
    Returns:
        Jacobian matrix with shape (..., M, N)
    """
    jac = []  # List to hold the gradients
    # Select only the second parameter for gradient computation
    second_param = [params[1]]  # Assuming params is indexable, otherwise convert it to a list
    #param = tuple(params)
    for i in range(y.shape[-1]):
        # Create a zero tensor with the same shape as y
        grad_output = torch.zeros_like(y)
        # Set the i-th element to 1
        grad_output[..., i] = 1
        # Compute the gradient with respect to the second parameter
        grad = torch.autograd.grad(y, second_param, grad_outputs=grad_output, retain_graph=True, create_graph=True, allow_unused=True)
        # grad is a tuple of gradients, one for each parameter tensor (only one in this case).
        # Check if the gradient is not None and flatten
        grad = grad[0].flatten() if grad[0] is not None else torch.tensor([])
        jac.append(grad)
    return torch.stack(jac, dim=-2)

def compute_ntk(net, x):    
    """ Compute the Neural Tangent Kernel (NTK) of a network
    Args:
        net: the network
        x: input Tensor
    Returns:
        NTK matrix with shape (..., M, M)
    """
    net.zero_grad()
    _, _, _, _, hidden_activity = net(x)

    # Extract the hidden activity of the last step
    hidden_activity_last_step = hidden_activity[:,-1,:]
    
    # Convert net.parameters() to a list to index it
    params_list = list(net.parameters())
    
    # Compute the Jacobian of the hidden activity of the last step with respect to the second network parameter
    J_hidden_last_step = jacobian(hidden_activity_last_step, params_list)
    
    # Compute the NTK as the product of the Jacobian with itself, transposed
    ntk = torch.einsum('ij,kj->ik', J_hidden_last_step, J_hidden_last_step)
    
    return ntk





def compute_ntk(model, inputs):
    """
    Compute the Neural Tangent Kernel for models with 3D outputs.
    The function computes the NTK by considering the gradient outer product,
    aggregated over timesteps.
    
    Args:
        model (torch.nn.Module): The neural network model.
        inputs (torch.Tensor): Input tensor of shape [batch_size, nsteps, input_size].

    Returns:
        torch.Tensor: The computed NTK matrix.
    """
    # Ensure model is in eval mode to prevent training updates
    model.eval()
    
    # Placeholder for aggregated NTK
    ntk_aggregate = None

    # Loop through timesteps
    nsteps = inputs.size(1)
    for step in range(nsteps):
        # Extract timestep-specific input
        input_t = inputs[:, step, :]

        # Forward pass for specific timestep (requires adjustment in model's forward method)
        output_t = model(input_t.unsqueeze(1), track=False)  # Unsqueezing to keep the second dim for compatibility
        
        # Assuming a single output at each timestep for simplicity
        # You might need to adapt this if your model or objective is different
        output_t = output_t.squeeze(1)  # Adjust based on your model's output shape
        
        # Compute gradients w.r.t. all model parameters
        grads = torch.autograd.grad(outputs=output_t, inputs=model.parameters(),
                                    grad_outputs=torch.ones_like(output_t),
                                    create_graph=True, retain_graph=True, only_inputs=True)
        
        # Flatten gradients to compute outer product more easily
        grads_flattened = torch.cat([g.flatten() for g in grads])
        
        # Compute outer product of gradients
        ntk_t = torch.outer(grads_flattened, grads_flattened)
        
        # Aggregate NTK across timesteps
        if ntk_aggregate is None:
            ntk_aggregate = ntk_t
        else:
            ntk_aggregate += ntk_t
    
    # Normalize aggregated NTK by the number of timesteps
    ntk_aggregate /= nsteps

    return ntk_aggregate





    # Normalize the NTK matrix.
    # Here, we'll use a simple normalization by the Frobenius norm of the NTK.
    # Other normalization schemes can be applied based on the specifics of the network or the representations.
    norm_factor = torch.norm(ntk, p='fro')
    print(norm_factor)
    #normalized_ntk = ntk / norm_factor

    # Compute eigenvalues of the NTK.
    eigenvalues = torch.linalg.eigvals(ntk)
    
    # Use the sum of absolute eigenvalues as the normalization factor.
    norm_factor = torch.sum(torch.abs(eigenvalues))
    print(norm_factor)
    # Alternatively, use the maximum absolute eigenvalue for normalization.
    norm_factor = torch.max(torch.abs(eigenvalues))
    print(norm_factor)
    
    #normalized_ntk = ntk / norm_factor
    # Compute variance of the activations.
    activation_variance = torch.var(hidden_activity_last_step)
    print(norm_factor)
    
    normalized_ntk = ntk / activation_variance