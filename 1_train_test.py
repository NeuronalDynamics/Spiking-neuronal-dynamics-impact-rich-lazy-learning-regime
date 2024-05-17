from argparse import ArgumentParser
import yaml

import pytorch_lightning as pl
import torch

from datasets.sine import SineDataModule
from datasets.mnist import MNISTDataModule
from datasets.nmnist import NMNISTDataModule
from datasets.utils import add_data_args
from models.pl_modules_2 import add_structure_args, add_general_model_args, add_training_args, GLIFRN, RNN, LSTMN
from training.callbacks import *
from utils.types import NeuronParameters
from pytorch_lightning.callbacks import EarlyStopping

# Must specify default_root_dir
data_modules = {
    "sine": SineDataModule,
    "pmnist": MNISTDataModule,
    "lmnist": MNISTDataModule,
    "nmnist": NMNISTDataModule
}

def read_yaml_into_args(args, filename, modeltype):
    with open(filename) as f:
        data = yaml.load(f, Loader=yaml.loader.SafeLoader)
    # data = yaml.load(filename)
    arg_dict = args.__dict__
    for key, value in data.items():
        if isinstance(value, dict):
            arg_dict[key] = value[modeltype]
            # print(value[modeltype])
        # if key in arg_dict:
        #     continue
        elif isinstance(value, list):
            arg_dict[key] = []
            for v in value:
                arg_dict[key].append(v)
        else:
            arg_dict[key] = value
    return args

def jacobian(y, params):
    """ Compute the Jacobian matrix dy/dx
    Args:
        y: output Tensor with shape (..., M)
        params: iterator of Tensors, network parameters
    Returns:
        Jacobian matrix with shape (..., M, N)
    """
    params = tuple(params)  # Convert parameters to a tuple of tensors
    jac = []  # List to hold the gradients
    for i in range(y.shape[-1]):
        # Create a zero tensor with the same shape as y
        grad_output = torch.zeros_like(y)
        # Set the i-th element to 1
        grad_output[..., i] = 1
        # Compute the gradient with respect to params
        grad = torch.autograd.grad(y, params, grad_outputs=grad_output, retain_graph=True, create_graph=True, allow_unused=True)
        # grad is a tuple of gradients, one for each parameter tensor. 
        # We'll need to concatenate these into a single tensor.
        grad = torch.cat([g.flatten() for g in grad if g is not None])  # Exclude None gradients
        jac.append(grad)
    return torch.stack(jac, dim=-2)
    
'''
def jacobian(hidden_activity_last_step, parameters):
    jacobians = []
    for i in range(hidden_activity_last_step.size(1)):  # Iterate over each unit of activity
        # Compute gradient of each unit w.r.t. parameters
        grad_list = torch.autograd.grad(outputs=hidden_activity_last_step[:, i].sum(), inputs=parameters,
                                        create_graph=True, retain_graph=True, allow_unused=True)
        
        # Filter out None gradients and reshape
        grad_tensors = [g.contiguous().view(-1) for g in grad_list if g is not None]
        
        # Handle case where all gradients are None (unlikely but safer to check)
        if not grad_tensors:
            # This block is executed if all gradients are None, which means there's an issue with the computation graph.
            # You might need to ensure that your model's forward pass actually uses the parameters.
            raise RuntimeError("All gradients are None. This likely indicates a disconnection in the computation graph.")
        
        grad_vec = torch.cat(grad_tensors)
        jacobians.append(grad_vec)

    # Stack to form the Jacobian matrix
    jacobian_matrix = torch.stack(jacobians)

    return jacobian_matrix
'''
    

def compute_ntk(net, x):    
    """ Compute the Neural Tangent Kernel (NTK) of a network
    Args:
        net: the network
        x: input Tensor
    Returns:
        NTK matrix with shape (..., M, M)
    """
    net.zero_grad()
    # Ensure that the network has a forward method that returns both the output and the hidden activity
    _, _, _, _, hidden_activity = net(x)

    # Extract the hidden activity of the last step
    hidden_activity_last_step = hidden_activity[:,-1,:]  # Get the last step's activity
    
    # Compute the Jacobian of the hidden activity of the last step with respect to the network parameters
    J_hidden_last_step = jacobian(hidden_activity_last_step, net.parameters())
    
    # Compute the NTK as the product of the Jacobian with itself, transposed
    ntk = torch.einsum('ij,kj->ik', J_hidden_last_step, J_hidden_last_step)
    
    return ntk
'''
def get_ntk(net, x):
    """ 
    Compute the Neural Tangent Kernel (NTK) of a network
    Args:
        net: the network
        x: input Tensor
    Returns:
        NTK matrix with shape (..., M, M)
    """
    x.requires_grad_(True)
    output, _, _, _, hidden_activity = net(x)
    jacobian = []
    for i in range(output.size(-1)):
        grad = torch.autograd.grad(outputs=output[:, i].sum(), inputs=x, retain_graph=True, create_graph=True)[0]
        jacobian.append(grad)
    
    ntk = torch.einsum('b...ij,b...kj->b...ik', torch.stack(jacobian, dim=-1), torch.stack(jacobian, dim=-1))

    return ntk.mean(dim=0)
'''
'''
def get_ntk(model, inputs):
    """
    Compute the Neural Tangent Kernel (NTK) for a given model and inputs.

    Args:
        model (torch.nn.Module): The model for which to compute the NTK.
        inputs (torch.Tensor): The inputs to the model, requires grad.
        use_last_layer (bool): Whether to compute the NTK using only the last layer's activations.
                                If False, computes the NTK based on the model's final output.

    Returns:
        torch.Tensor: The computed NTK matrix.
    """

    # Ensure model is in evaluation mode to avoid dropout effects, etc.
    model.eval()

    # Forward pass to compute the outputs/activations necessary for NTK
    outputs, _, _, _, hidden_activity = model(inputs)
'''
"""
    batch_size, nsteps, out_size = outputs.shape

    # Initialize an empty NTK matrix
    ntk = torch.zeros((out_size, out_size), device=inputs.device, dtype=inputs.dtype)

    # Iterate over each time step and accumulate the NTK from all steps
    for step in range(nsteps):
        # Select the outputs at the current step
        outputs_step = outputs[:, step, :]

        # Compute gradients for each output feature w.r.t. parameters at this step
        for i in range(out_size):
            model.zero_grad()
            output_grad = torch.autograd.grad(outputs_step[:, i].sum(), model.parameters(), create_graph=True)
            flat_grad = torch.cat([g.view(-1) for g in output_grad if g is not None])

            for j in range(i, out_size):
                model.zero_grad()
                output_grad_j = torch.autograd.grad(outputs_step[:, j].sum(), model.parameters(), create_graph=True)
                flat_grad_j = torch.cat([g.view(-1) for g in output_grad_j if g is not None])

                # Compute the dot product of gradients (i-th and j-th feature gradients at this step)
                ntk[i, j] += (flat_grad * flat_grad_j).sum()
                ntk[j, i] = ntk[i, j]  # NTK is symmetric

    # Normalize by the product of batch size and number of steps for batch- and time step-wise NTK approximation
    return ntk / (batch_size * nsteps)
"""
'''
    final_hidden_activity = hidden_activity[:,-1,:]  # Shape: (batch_size, hidden_size)

    batch_size, hidden_size = final_hidden_activity.shape

    # Initialize an empty NTK matrix
    ntk = torch.zeros((hidden_size, hidden_size), device=inputs.device, dtype=inputs.dtype)

    # Compute gradients for each hidden state feature w.r.t. parameters
    for i in range(hidden_size):
        model.zero_grad()
        output_grad = torch.autograd.grad(final_hidden_activity[:, i].sum(), model.parameters(), create_graph=True, allow_unused=True)
        flat_grad = torch.cat([g.view(-1) for g in output_grad if g is not None])

        for j in range(i, hidden_size):
            model.zero_grad()
            output_grad_j = torch.autograd.grad(final_hidden_activity[:, j].sum(), model.parameters(), create_graph=True, allow_unused=True)
            flat_grad_j = torch.cat([g.view(-1) for g in output_grad_j if g is not None])

            # Compute the dot product of gradients (i-th and j-th hidden state feature gradients)
            ntk[i, j] += (flat_grad * flat_grad_j).sum()
            ntk[j, i] = ntk[i, j]  # NTK is symmetric

    # Normalize by batch size for batch-wise NTK approximation
    return ntk / batch_size
'''

'''
def get_ntk(model, inputs):
    """
    Calculates the Neural Tangent Kernel for a given model and a set of inputs.

    Args:
        model (torch.nn.Module): The neural network model.
        inputs (torch.Tensor): The inputs to the network, shape (n_samples, n_features).

    Returns:
        torch.Tensor: The Neural Tangent Kernel matrix, shape (n_samples, n_samples).
    """
    n_samples = inputs.shape[0]
    ntk = torch.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):
            # Clear any previously calculated gradients
            model.zero_grad()

            # Compute output for input i
            output_i = model(inputs[i].unsqueeze(0))
            grad_i = torch.autograd.grad(outputs=output_i, inputs=model.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
            grad_i_vec = torch.cat([g.view(-1) for g in grad_i])

            # Compute output for input j
            output_j = model(inputs[j].unsqueeze(0))
            grad_j = torch.autograd.grad(outputs=output_j, inputs=model.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
            grad_j_vec = torch.cat([g.view(-1) for g in grad_j])

            # Calculate the dot product of the gradients
            ntk[i, j] = torch.dot(grad_i_vec, grad_j_vec)
            ntk[j, i] = ntk[i, j]  # NTK is symmetric

    return ntk
    '''

def train(args, version): 
    # The order of priority for the configurations are first
    # the task related yaml and then the model realted yaml.
    print(f"training {args.model} on {args.task} on trial {version}")

    args.__dict__["default_root_dir"] = os.path.abspath(f"results/{args.task}-agn/{args.model}/trial_{version}")
    # if not os.path.exists(args.default_root_dir):
    os.makedirs(args.default_root_dir, exist_ok=True)
    if args.model == "rnn":
        args = read_yaml_into_args(args, "./../config/rnn.yaml", args.model)
        model_class = RNN
    elif args.model == "lstmn":
        args = read_yaml_into_args(args, "./../config/lstmn.yaml", args.model)
        model_class = LSTMN
    elif args.model.startswith("glifr"):
        args = read_yaml_into_args(args, "./../config/glifr.yaml", args.model)
        model_class = GLIFRN


        # Learning params
        if "lhet" in args.model or "rhet" in args.model:
            args.__dict__["params_learned"] = True
        else:
            args.__dict__["params_learned"] = False
        # Number of ascs
        if not args.model.endswith("a"):
            args.__dict__["num_ascs"] = 0
        # Initialization
        if "fhet" in args.model or "rhet" in args.model:
            args.__dict__["initialization"] = "het"
        else:
            args.__dict__["initialization"] = "hom"
    
    args = read_yaml_into_args(args, f"./../config/{args.task}_task.yaml", args.model)

    # add all the available trainer options to argparse
    # ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
    args.__dict__["callbacks"] = [eval(c)() for c in args.__dict__["callbacks"]]

    '''
    early_stopping = EarlyStopping(
    monitor='train_loss',  # Adjust if your validation loss has a different key
    min_delta=0.001,  # This is to prevent stopping for very tiny improvements
    patience=5,  # Number of epochs with no improvement after which training will be stopped
    verbose=True,
    mode='min',
    check_finite=True,
    stopping_threshold=0.25,  # Stops training immediately once the metric is below this threshold
    divergence_threshold=None,
    check_on_train_epoch_end=False  # Check on validation epoch end by default
    )
    '''

    Tau_log = []
    sign_similarity_log_1 = [] 
    #sign_similarity_log_2 = [] 
    representation_alignment_log_1 = []
    tangent_kernel_alignment_log = []
    Tau_init = 1
    Tau_inc = 2
    Tau_end = Tau_init+(13*Tau_inc)
    
    for Tau in range(Tau_init, Tau_end, Tau_inc): #sigma_v R I_0
        # Analysis params 
        args.__dict__["tau"] = Tau / 10.0 #0.5
        args.__dict__["sigma_v"] = 1
        args.__dict__["R"] = 0.1
        args.__dict__["I_0"] = 0
        print("********************************************************")
        print("tau = ",args.tau)
        print("********************************************************")
        #args.__dict__["max_epochs"] = 2

        # tb_logger = TensorBoardLogger(str(version))
        # args.__dict__["logger"] = tb_logger
        trainer = pl.Trainer.from_argparse_args(args)
        #trainer = pl.Trainer.from_argparse_args(args,callbacks=[early_stopping] + args.__dict__["callbacks"])

        if Tau == Tau_init:
            delattr(args, 'callbacks')
            # delattr(args, 'logger')
        model = model_class(**vars(args))

        # if args.model in ["glifr_rhet", "glifr_rheta", "glifr_fhet", "glifr_fheta"]:
        #     model = model_class.load_from_checkpoint(args.model_ckpt)
        print(f"training {args.model} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
        data_module = data_modules[args.task](**vars(args))


        # Create global variables for data so it is the same inputs to calculate measurement
        data_module.setup()

        # Assuming data_module.train_dataloader() returns a DataLoader
        train_loader = data_module.train_dataloader()
        for inputs, _ in train_loader:
            inputs_0 = inputs.clone().detach().requires_grad_(True)
        #print(inputs_0.shape)  # To verify the shape
        #inputs_0 = inputs_0[:1,:,:]


        model.reset_state(inputs_0.size()[0])

        #print(args)
        # Initial Hidden State/Voltages
        #if args.model in ["rnn", "lstmn"]:
        #    _, voltages_0 = model(inputs_0)#.forward(inputs_0, track=True)
        #    voltages_0 = voltages_0.detach().numpy()
        #    np.savetxt(f"./analysis/{args.model}_{version}_voltage_0.csv", voltages_0[0,:,0], delimiter=",")
        if args.model in ["glifr_lheta", "glifr_fheta", "glifr_rheta", "glifr_homa", "glifr_lhet", "glifr_fhet", "glifr_rhet", "glifr_hom"]:
            _, _, _, _, activity_0 = model(inputs_0) #model.forward(inputs_0, track=True) #
            #np.savetxt(f"./analysis/{args.model}_{version}_tau={args.tau}_sigma_v={args.sigma_v}_R={args.R}_I0={args.I0}_syn_0.csv", syns_0[0,:,0], delimiter=",")
            #np.savetxt(f"./analysis/{args.model}_{version}_tau={args.tau}_sigma_v={args.sigma_v}_R={args.R}_I0={args.I0}_voltage_0.csv", voltages_0[0,:,0], delimiter=",")
            #np.savetxt(f"./analysis/{args.model}_{version}_tau={args.tau}_sigma_v={args.sigma_v}_R={args.R}_I0={args.I0}_asc_0.csv", ascs_0[:,0,:,0], delimiter=",")

        #print(activity_0.shape)
        ntk_0 = compute_ntk(model, inputs_0)

        trainer.fit(model, data_module)

        # Final Hidden State/Voltages
        #if args.model in ["rnn", "lstmn"]:
        #    _, voltages_f = model(inputs_0)#.forward(inputs_0, track=True)
        #    voltages_f = voltages_f.detach().numpy()
        #    np.savetxt(f"./analysis/{args.model}_{version}_voltages_f.csv", voltages[0,:,0], delimiter=",")
        if args.model in ["glifr_lheta", "glifr_fheta", "glifr_rheta", "glifr_homa", "glifr_lhet", "glifr_fhet", "glifr_rhet", "glifr_hom"]:
            _, _, _, _, activity_f = model(inputs_0)#model.forward(inputs_0, track=True) #
            #np.savetxt(f"./analysis/{args.model}_{version}_tau={args.tau}_sigma_v={args.sigma_v}_R={args.R}_I0={args.I0}_syn_f.csv", syns_f[0,:,0], delimiter=",")
            #np.savetxt(f"./analysis/{args.model}_{version}_tau={args.tau}_sigma_v={args.sigma_v}_R={args.R}_I0={args.I0}_voltage_f.csv", voltages_f[0,:,0], delimiter=",")
            #np.savetxt(f"./analysis/{args.model}_{version}_tau={args.tau}_sigma_v={args.sigma_v}_R={args.R}_I0={args.I0}_asc_f.csv", ascs_f[:,0,:,0], delimiter=",")

        # Calculate Tangent Kernel Alignment
        #print(activity_f.shape)
        ntk_f = compute_ntk(model, inputs_0)  
        tangent_kernel_alignment = torch.trace(ntk_0 @ ntk_f.T) / (torch.norm(ntk_0) * torch.norm(ntk_f))
        print("TKA", tangent_kernel_alignment)
                
        # Calculate Sign Similarity
        #sign_similarity_1 = torch.sum(torch.sign(activity_0[:,-1,:]) == torch.sign(activity_f[:,-1,:])) / (activity_0[:,-1,:].size()[0] * activity_f[:,-1,:].size()[1])
        ## Calculate the sign of the activations
        #sign_activity_0 = torch.sign(activity_0[:,-1,:])
        #sign_activity_f = torch.sign(activity_f[:,-1,:])
        ## Calculate the boolean tensor of matching signs
        #matching_signs = (sign_activity_0 == sign_activity_f)
        ## Cast boolean tensor to float for summation and division
        #matching_signs_float = matching_signs.float()
        ## Calculate sign similarity
        #num_elements = activity_0[:,-1,:].numel()  # Alternative to calculating size[0] * size[1]
        #sign_similarity = torch.sum(matching_signs_float) / num_elements
        ## Convert to Python scalar for readability, if needed
        #sign_similarity_2 = sign_similarity.item()
            
        # Calculate Representation Alignment Option #1
        #K_0 = torch.mm(torch.t(activity_0[:,-1,:]), activity_0[:,-1,:])
        #K_t = torch.mm(torch.t(activity_f[:,-1,:]), activity_f[:,-1,:])
        #representation_alignment_1 = torch.trace(torch.mm(K_t, K_0))/(torch.norm(K_t, p='fro')*torch.norm(K_0, p='fro'))
        # Calculate Gram matrices for activity_0 and activity_f at the last timestep
        K_0 = torch.mm(activity_0[:,-1,:].T, activity_0[:,-1,:])
        K_t = torch.mm(activity_f[:,-1,:].T, activity_f[:,-1,:])
        # Calculate representation alignment
        representation_alignment_1 = torch.trace(torch.mm(K_t, K_0)) / (torch.norm(K_t, p='fro') * torch.norm(K_0, p='fro'))
        print("RA1", representation_alignment_1)
        Tau_log.append(Tau)
        #sign_similarity_log_1.append(sign_similarity_1.numpy())
        #sign_similarity_log_2.append(sign_similarity_2)#(sign_similarity.numpy())#
        tangent_kernel_alignment_log.append(tangent_kernel_alignment.detach().numpy())
        representation_alignment_log_1.append(representation_alignment_1.detach().numpy())
        #np.savetxt(f"./analysis/tau/{args.model}_{version}_SS.csv", sign_similarity_log, delimiter=",")

        trainer.save_checkpoint(os.path.join(trainer.logger.log_dir, "checkpoints", "last.ckpt"))
        trainer.test(datamodule=data_module, ckpt_path=os.path.join(trainer.logger.log_dir, "checkpoints", "last.ckpt"))

    np.savetxt(f"./analysis/tau/{args.model}_{version}_TAU.csv", Tau_log, delimiter=",")
    np.savetxt(f"./analysis/tau/{args.model}_{version}_TKA.csv", tangent_kernel_alignment_log, delimiter=",")
    #np.savetxt(f"./analysis/tau/{args.model}_{version}_SS_1.csv", sign_similarity_log_1, delimiter=",")
    #np.savetxt(f"./analysis/tau/{args.model}_{version}_SS_2.csv", sign_similarity_log_2, delimiter=",")
    np.savetxt(f"./analysis/tau/{args.model}_{version}_RA.csv", representation_alignment_log_1, delimiter=",")

    #np.savetxt(f"./analysis/sigma/{args.model}_{version}_TAU.csv", Tau_log, delimiter=",")
    #np.savetxt(f"./analysis/sigma/{args.model}_{version}_TKA.csv", tangent_kernel_alignment_log, delimiter=",")
    #np.savetxt(f"./analysis/sigma/{args.model}_{version}_SS.csv", sign_similarity_log, delimiter=",")
    #np.savetxt(f"./analysis/sigma/{args.model}_{version}_RA.csv", representation_alignment_log_1, delimiter=",")
    return trainer.logger.log_dir

if __name__ == '__main__':
    print(f"found {torch.cuda.device_count()} devices")
    parser = ArgumentParser()
    pl.Trainer.add_argparse_args(parser)

    # Model-specific
    add_structure_args(parser)
    add_general_model_args(parser)
    add_training_args(parser)
    add_data_args(parser)
    RNN.add_model_specific_args(parser)
    LSTMN.add_model_specific_args(parser)
    GLIFRN.add_model_specific_args(parser)

    # Dataset-specific
    SineDataModule.add_sine_args(parser)
    NMNISTDataModule.add_nmnist_args(parser)
    MNISTDataModule.add_mnist_args(parser)

    # add program level args
    parser.add_argument("--task", type=str, choices=["sine", "nmnist", "pmnist", "lmnist"])
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ntrials", type=int, default=1)
    # parser.add_argument("--model", type=str, choices=["rnn", "lstmn", "glifr_lheta", "glifr_fheta", "glifr_rheta", "glifr_homa", "glifr_lhet", "glifr_fhet", "glifr_rhet", "glifr_hom"])

    # add model specific args
    args = parser.parse_args()

    for i in range(args.ntrials):
        model_names = ["rnn", "lstmn", "glifr_lheta", "glifr_fheta", "glifr_rheta", "glifr_homa", "glifr_lhet", "glifr_fhet", "glifr_rhet", "glifr_hom"]
        ckpt_dict = {}
        to_train = ["glifr_lheta", "glifr_rheta"] #glifr_fheta glifr_rheta glifr_homa
        for m in ["rnn", "lstmn", "glifr_lheta", "glifr_homa", "glifr_lhet", "glifr_hom"]:
        # for m in ["rnn"]:
            if m not in to_train:
                continue
            args.__dict__["model"] = m
            ckpt_dict[m] = train(args, version=i)

        for m in ["glifr_fhet", "glifr_rhet"]:
            if m not in to_train:
                continue
            args.__dict__["model"] = m
            args.__dict__["ckpt_path"] = os.path.join(ckpt_dict["glifr_lhet"], "checkpoints", "last.ckpt")
            train(args, version=i)

        for m in ["glifr_fheta", "glifr_rheta"]:
            if m not in to_train:
                continue
            args.__dict__["model"] = m
            args.__dict__["ckpt_path"] = os.path.join(ckpt_dict["glifr_lheta"], "checkpoints", "last.ckpt")
            train(args, version=i)
