from argparse import ArgumentParser
import yaml

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from datasets.sine import SineDataModule
from datasets.mnist import MNISTDataModule
from datasets.nmnist import NMNISTDataModule
from datasets.utils import add_data_args
from models.pl_modules_2_3 import add_structure_args, add_general_model_args, add_training_args, GLIFRN, RNN, LSTMN
from training.callbacks import *
from utils.types import NeuronParameters
from pytorch_lightning.callbacks import EarlyStopping
import sys
from torch.cuda.amp import autocast
from training.loss_fns import *

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
'''
def batched_jacobian(outputs, inputs, batch_size=10):
    jacobians = []
    for i in range(0, outputs.size(0), batch_size):
        end = min(i + batch_size, outputs.size(0))
        grad_outputs = torch.zeros_like(outputs)
        grad_outputs[i:end] = 1
        grads = torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs,
                                    retain_graph=True, create_graph=False, allow_unused=True)
        grads = torch.cat([g.contiguous().view(-1) if g is not None else torch.zeros_like(inputs[j]) for j, g in enumerate(grads)], dim=0)
        jacobians.append(grads)
    return torch.stack(jacobians)

def stochastic_ntk(input, model, num_samples=96, batch_size=1):
    model.eval()
    subset_indices = torch.randperm(input.size(0))[:num_samples]
    sampled_input = input[subset_indices].detach().requires_grad_(True)
    output = model(sampled_input)

    if isinstance(output, tuple):  # Assuming the actual output is the first element
        output = output[0]
    
    output_flat = output.view(-1)
    params = list(model.parameters())

    jacob = batched_jacobian(output_flat, params, batch_size=batch_size)
    ntk = torch.einsum('ij,ik->jk', jacob, jacob)
    return ntk

def finite_difference_jacobian(model, inputs, epsilon=1e-5):
    inputs.requires_grad_(False)
    outputs = model(inputs)

    # Handling possible tuple output:
    if isinstance(outputs, tuple):
        outputs = outputs[0]

    num_outputs = outputs.numel()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    jacobian = torch.zeros(num_outputs, num_params, device=outputs.device)

    index = 0
    for param in model.parameters():
        if param.requires_grad:
            original_values = param.data.clone()
            param_size = param.numel()

            for i in range(param_size):
                param.view(-1)[i] += epsilon
                model.zero_grad()
                
                if isinstance(model(inputs), tuple):
                    perturbed_output = model(inputs)[0]
                else:
                    perturbed_output = model(inputs)

                perturbed_output = perturbed_output.view(-1)
                diff = (perturbed_output - outputs.view(-1)) / epsilon
                jacobian[:, index + i] = diff
                param.data = original_values

            index += param_size

    return jacobian


def compute_ntk(inputs, model):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        
        # Check if outputs are in a tuple and unpack
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        outputs.requires_grad_(True)
        jacobian = finite_difference_jacobian(model, inputs)
        ntk = torch.mm(jacobian, jacobian.t())

    return ntk


def input_perturbation_sensitivity(model, input, epsilon=1e-5):
    # Ensure model is in evaluation mode
    model.eval()

    # Original prediction
    with torch.no_grad():
        original_output = model(input)[0]  # Assuming the output is the first element in the model's return tuple

    # Perturbed input
    perturbed_input = input + epsilon * torch.randn_like(input)

    # Output with perturbed input
    with torch.no_grad():
        perturbed_output = model(perturbed_input)[0]

    # Compute the sensitivity as the norm of the output difference
    sensitivity = torch.norm(perturbed_output - original_output)

    return sensitivity.item()


def spectral_analysis(weight_matrix):
    # Assuming weight_matrix is a 2D tensor
    eigenvalues = torch.linalg.eigvals(weight_matrix).abs()  # Get the magnitude of eigenvalues
    # Normalize eigenvalues for comparison
    #norm_factor = torch.sum(eigenvalues)
    #normalized_spectrum = eigenvalues / norm_factor
    return eigenvalues.detach().numpy() #normalized_spectrum

def random_direction(size):
    """ Generate a random direction in parameter space """
    vec = torch.randn(size)
    return vec / torch.norm(vec)

def loss_along_direction(model, criterion=cross_entropy_one_hot, data, targets, direction, steps, scale):
    #loss_fn: cross_entropy_one_hot
    """ Evaluate loss along a random direction in the parameter space """
    original_params = [p.clone() for p in model.parameters()]
    losses = []

    # Move in the positive and negative direction of the vector
    for alpha in np.linspace(-scale, scale, steps):
        # Perturb model parameters along the direction
        for i, p in enumerate(model.parameters()):
            perturbation = alpha * direction[i]
            p.data = original_params[i] + perturbation

        # Compute the loss
        model.eval()
        with torch.no_grad():
            outputs = model(data)
            loss = criterion(outputs, targets)
            losses.append(loss.item())

        # Reset parameters after computation
        for i, p in enumerate(model.parameters()):
            p.data = original_params[i]

    return losses

def plot_loss_landscape(losses, scale):
    plt.plot(np.linspace(-scale, scale, len(losses)), losses)
    plt.title("Loss Landscape")
    plt.xlabel("Scale of Perturbation")
    plt.ylabel("Loss")
    plt.show()


def compute_hessian(loss, parameters):
    """
    Compute the Hessian matrix for the loss function with respect to the parameters.
    """
    # First, compute gradients of the loss with respect to parameters
    first_grads = grad(loss, parameters, create_graph=True, allow_unused=True)
    first_grads = [g for g in first_grads if g is not None]  # Filter out None gradients
    
    hessian = []
    for i, grad_i in enumerate(first_grads):
        # Compute second derivatives of each gradient component
        grad_i_components = torch.autograd.grad(grad_i, parameters, retain_graph=True)
        hessian.append(grad_i_components)

    # Convert list of gradient tuples into a full Hessian matrix
    hessian_matrix = torch.cat([torch.cat([g.contiguous().view(-1) for g in grad_tuple]) for grad_tuple in hessian])
    return hessian_matrix.view(len(first_grads), -1)

def analyze_minima_flatness(model, data, target, criterion=cross_entropy_one_hot):
    """
    Analyze the flatness of minima by calculating the eigenvalues of the Hessian matrix.
    """
    model.eval()
    outputs = model(data)
    loss = criterion(outputs, target)
    
    parameters = [p for p in model.parameters() if p.requires_grad]
    hessian_matrix = compute_hessian(loss, parameters)

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvals(hessian_matrix).real  # Get the real parts of eigenvalues
    return eigenvalues
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
    
    args = read_yaml_into_args(args, f"./../config/lmnist_task.yaml", args.model)

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

    Sigma_log = []
    #sign_similarity_log_1 = [] 
    #sign_similarity_log_2 = [] 
    representation_alignment_log_1 = []
    #representation_alignment_log_2 = []
    #tangent_kernel_alignment_log = []
    #output_weight_alignment_log = []
    #weight_matrix_alignment_log = []
    effective_weight_change_norm_log = []
    #spectral_similarity_log = []
    #normalized_sensitivity_log = []
    #trans_k_j_alignment_log = []
    #trans_r_j_alignment_log = []
    #a_j_alignment_log = []
    #thresh_alignment_log = []
    #trans_k_m_alignment_log = []
    #transform_to_asc_r_alignment_log = []
    #input_weight_alignment_log = []
    #hidden_weight_change_log = []
    Jacobian_change_norm_log = []
    #eff_weight_cos_sim_log = []
    #Jacobian_cos_sim_log = []

    Sigma_init = 5
    Sigma_inc = 5 #2
    Sigma_end = Sigma_init+(8*Sigma_inc) #13
    
    for Sigma in range(Sigma_init, Sigma_end, Sigma_inc): #tau sigma_v R I_0
        # Analysis params 
        args.__dict__["tau"] = 0.5 #0.5
        args.__dict__["sigma_v"] = Sigma / 10.0 #1
        args.__dict__["R"] = 0.1
        args.__dict__["I_0"] = 0
        print("********************************************************")
        print("Sigma_v = ",args.sigma_v)
        print("********************************************************")
        #args.__dict__["max_epochs"] = 2

        # tb_logger = TensorBoardLogger(str(version))
        # args.__dict__["logger"] = tb_logger
        trainer = pl.Trainer.from_argparse_args(args)
        #trainer = pl.Trainer.from_argparse_args(args,callbacks=[early_stopping] + args.__dict__["callbacks"])

        if Sigma == Sigma_init:
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
            #inputs_0 = torch.tensor(inputs)
        #print(inputs_0.shape)  # To verify the shape
        #inputs_0 = inputs_0[:1,:,:]


        model.reset_state(inputs_0.size()[0])

        #thresh_0 = model.neuron_layer.thresh.clone()
        #trans_k_m_0 = model.neuron_layer.trans_k_m.clone()
        #trans_k_j_0 = model.neuron_layer.trans_k_j.clone()
        #trans_r_j_0 = model.neuron_layer.trans_r_j.clone()
        #transform_to_asc_r_0 = model.neuron_layer.transform_to_asc_r(trans_r_j_0)
        #a_j_0 = model.neuron_layer.a_j.clone()
        #output_weight_0 = model.output_linear.weight.clone()
        #input_matrix_0 = model.neuron_layer.weight_iv.clone()
        weight_matrix_0 = model.neuron_layer.weight_lat.clone()
        #initial_spectrum = spectral_analysis(weight_matrix_0)
        #sensitivity_0 = input_perturbation_sensitivity(model, inputs_0, epsilon=1e-5)


        # Initial Hidden State/Voltages
        #if args.model in ["rnn", "lstmn"]:
        #    _, voltages_0 = model(inputs_0)#.forward(inputs_0, track=True)
        #    voltages_0 = voltages_0.detach().numpy()
        #    np.savetxt(f"./analysis/{args.model}_{version}_voltage_0.csv", voltages_0[0,:,0], delimiter=",")
        if args.model in ["glifr_lheta", "glifr_fheta", "glifr_rheta", "glifr_homa", "glifr_lhet", "glifr_fhet", "glifr_rhet", "glifr_hom"]:
            outputs_0, voltages_0, _, _, activity_0 = model(inputs_0) #model.forward(inputs_0, track=True) #
            #np.savetxt(f"./analysis/{args.model}_{version}_tau={args.tau}_sigma_v={args.sigma_v}_R={args.R}_I0={args.I0}_syn_0.csv", syns_0[0,:,0], delimiter=",")
            #np.savetxt(f"./analysis/{args.model}_{version}_tau={args.tau}_sigma_v={args.sigma_v}_R={args.R}_I0={args.I0}_voltage_0.csv", voltages_0[0,:,0], delimiter=",")
            #np.savetxt(f"./analysis/{args.model}_{version}_tau={args.tau}_sigma_v={args.sigma_v}_R={args.R}_I0={args.I0}_asc_0.csv", ascs_0[:,0,:,0], delimiter=",")

        #ntk_0 = compute_ntk(inputs_0, model)
        #ntk_0 = stochastic_ntk(inputs_0, model)
        #eigenvalues_0 = spectral_analysis_of_weight_matrices(model)
        #sensitivity_0 = compute_sensitivity(model, inputs_0)
        

        trainer.fit(model, data_module)

        #thresh_f = model.neuron_layer.thresh.clone()
        #trans_k_m_f = model.neuron_layer.trans_k_m.clone()
        #trans_k_j_f = model.neuron_layer.trans_k_j.clone()
        #trans_r_j_f = model.neuron_layer.trans_r_j.clone()
        #transform_to_asc_r_f = model.neuron_layer.transform_to_asc_r(trans_r_j_f)
        #a_j_f = model.neuron_layer.a_j.clone()
        #output_weight_f = model.output_linear.weight.clone()
        #input_matrix_f = model.neuron_layer.weight_iv.clone()
        weight_matrix_f = model.neuron_layer.weight_lat.clone()
        #final_spectrum = spectral_analysis(weight_matrix_f)
        #sensitivity_f = input_perturbation_sensitivity(model, inputs_0, epsilon=1e-5)
        # Calculate Tangent Kernel Alignment
        #ntk_f = compute_ntk(inputs_0, model)
        #ntk_f = stochastic_ntk(inputs_0, model)

        # Final Hidden State/Voltages
        #if args.model in ["rnn", "lstmn"]:
        #    _, voltages_f = model(inputs_0)#.forward(inputs_0, track=True)
        #    voltages_f = voltages_f.detach().numpy()
        #    np.savetxt(f"./analysis/{args.model}_{version}_voltages_f.csv", voltages[0,:,0], delimiter=",")
        if args.model in ["glifr_lheta", "glifr_fheta", "glifr_rheta", "glifr_homa", "glifr_lhet", "glifr_fhet", "glifr_rhet", "glifr_hom"]:
            outputs_f, voltages_f, _, _, activity_f = model(inputs_0)#model.forward(inputs_0, track=True) #
            #np.savetxt(f"./analysis/{args.model}_{version}_tau={args.tau}_sigma_v={args.sigma_v}_R={args.R}_I0={args.I0}_syn_f.csv", syns_f[0,:,0], delimiter=",")
            #np.savetxt(f"./analysis/{args.model}_{version}_tau={args.tau}_sigma_v={args.sigma_v}_R={args.R}_I0={args.I0}_voltage_f.csv", voltages_f[0,:,0], delimiter=",")
            #np.savetxt(f"./analysis/{args.model}_{version}_tau={args.tau}_sigma_v={args.sigma_v}_R={args.R}_I0={args.I0}_asc_f.csv", ascs_f[:,0,:,0], delimiter=",")

        #print("#########################################################")
        #tangent_kernel_alignment = torch.trace(ntk_0 @ ntk_f.T) / (torch.norm(ntk_0) * torch.norm(ntk_f))
        #print("TKA", tangent_kernel_alignment)
        
        #print("#########################################################")
        #thresh_alignment = torch.norm(thresh_f - thresh_0)
        #print(thresh_alignment)
        #print("#########################################################")
        #trans_k_m_alignment = torch.norm(trans_k_m_f - trans_k_m_0)
        #print(trans_k_m_alignment)
        #print("#########################################################")
        #K_j_0 = torch.mm(trans_k_j_0[:,-1,:].T, trans_k_j_0[:,-1,:])
        #K_j_t = torch.mm(trans_k_j_f[:,-1,:].T, trans_k_j_f[:,-1,:])
        #trans_k_j_alignment = torch.trace(torch.mm(K_j_t, K_j_0)) / (torch.norm(K_j_t, p='fro') * torch.norm(K_j_0, p='fro'))
        #print(trans_k_j_alignment)
        #print("#########################################################")
        #R_j_0 = torch.mm(trans_r_j_0[:,-1,:].T, trans_r_j_0[:,-1,:])
        #R_j_t = torch.mm(trans_r_j_f[:,-1,:].T, trans_r_j_f[:,-1,:])
        #trans_r_j_alignment = torch.trace(torch.mm(R_j_t, R_j_0)) / (torch.norm(R_j_t, p='fro') * torch.norm(R_j_0, p='fro'))
        #print(trans_r_j_alignment)
        #print("#########################################################")
        #A_j_0 = torch.mm(a_j_0[:,-1,:].T, a_j_0[:,-1,:])
        #A_j_t = torch.mm(a_j_f[:,-1,:].T, a_j_f[:,-1,:])
        #a_j_alignment = torch.trace(torch.mm(A_j_t, A_j_0)) / (torch.norm(A_j_t, p='fro') * torch.norm(A_j_0, p='fro'))
        #print(a_j_alignment)
        #print("#########################################################")
        #spectral_similarity = np.dot(final_spectrum, initial_spectrum) / (np.linalg.norm(final_spectrum) * np.linalg.norm(initial_spectrum))
        #print(f"Spectral Similarity: {spectral_similarity}")
        #print("#########################################################")
        #normalized_sensitivity = (sensitivity_f - sensitivity_0)/(sensitivity_f + sensitivity_0)
        #print(f"Normalized Sensitivity: {normalized_sensitivity}")
        #print("#########################################################")
        #output_weight_alignment = torch.linalg.matrix_norm(output_weight_f - output_weight_0)
        #print(output_weight_alignment)
        #print("#########################################################")
        #input_weight_alignment = torch.linalg.matrix_norm(input_matrix_f - input_matrix_0)
        #print(input_weight_alignment)
        #print("#########################################################")
        effective_weight_matrix_0 = model.neuron_layer.spike_fn_derivative(voltages_0.reshape(voltages_0.shape[0] * voltages_0.shape[1], voltages_0.shape[2])) @ weight_matrix_0
        effective_weight_matrix_f = model.neuron_layer.spike_fn_derivative(voltages_f.reshape(voltages_f.shape[0] * voltages_f.shape[1], voltages_f.shape[2])) @ weight_matrix_f
        effective_weight_change_norm = torch.norm(effective_weight_matrix_f - effective_weight_matrix_0, p='fro')
        print(effective_weight_change_norm)
        print("#########################################################")
        #eff_weight_cos_sim = np.dot(effective_weight_matrix_f.flatten().detach().numpy(), effective_weight_matrix_0.flatten().detach().numpy()) / (np.linalg.norm(effective_weight_matrix_f.flatten().detach().numpy()) * np.linalg.norm(effective_weight_matrix_0.flatten().detach().numpy()))
        #print(eff_weight_cos_sim)
        #print("#########################################################")
        J_0 = model.neuron_layer.spike_fn_derivative(activity_0[:,-1,:])
        J_f = model.neuron_layer.spike_fn_derivative(activity_f[:,-1,:])
        Jacobian_change_norm = torch.norm(J_f - J_0)
        print(Jacobian_change_norm)
        print("#########################################################")
        #Jacobian_cos_sim = np.dot(J_f.flatten().detach().numpy(), J_0.flatten().detach().numpy()) / (np.linalg.norm(J_f.flatten().detach().numpy()) * np.linalg.norm(J_0.flatten().detach().numpy()))
        #print(Jacobian_cos_sim)
        #print("#########################################################")
        #Delta_W = weight_matrix_f - weight_matrix_0
        #singular_values = np.linalg.svd(Delta_W.detach().numpy(), compute_uv=False)
        #hidden_weight_change = np.linalg.norm(singular_values)
        #print(hidden_weight_change)
        #print("#########################################################")
        #T_a_j_r_0 = torch.mm(transform_to_asc_r_0[:,-1,:].T, transform_to_asc_r_0[:,-1,:])
        #T_a_j_r_t = torch.mm(transform_to_asc_r_f[:,-1,:].T, transform_to_asc_r_f[:,-1,:])
        #transform_to_asc_r_alignment = torch.trace(torch.mm(T_a_j_r_t, T_a_j_r_0)) / (torch.norm(T_a_j_r_t, p='fro') * torch.norm(T_a_j_r_0, p='fro'))
        #print(transform_to_asc_r_alignment)
        #print("#########################################################")
        K_0 = torch.mm(activity_0[:,-1,:].T, activity_0[:,-1,:])
        K_t = torch.mm(activity_f[:,-1,:].T, activity_f[:,-1,:])
        # Calculate representation alignment
        representation_alignment_1 = torch.trace(torch.mm(K_t, K_0)) / (torch.norm(K_t, p='fro') * torch.norm(K_0, p='fro'))
        print("RA1", representation_alignment_1)
        print("#########################################################")
        #input_matrix_similiarty = np.dot(input_matrix_f.flatten().detach().numpy(), input_matrix_0.flatten().detach().numpy()) / (np.linalg.norm(input_matrix_f.flatten().detach().numpy()) * np.linalg.norm(input_matrix_0.flatten().detach().numpy()))
        #weight_matrix_similiary = np.dot(weight_matrix_f.flatten().detach().numpy(), weight_matrix_0.flatten().detach().numpy()) / (np.linalg.norm(weight_matrix_f.flatten().detach().numpy()) * np.linalg.norm(weight_matrix_0.flatten().detach().numpy()))
        #representation_alignment_2 = weight_matrix_similiary - input_matrix_similiarty
        #print("RA2", representation_alignment_2)
        #print("#########################################################")
        Sigma_log.append(Sigma)
        #sign_similarity_log_1.append(sign_similarity_1.numpy())
        #sign_similarity_log_2.append(sign_similarity_2)#(sign_similarity.numpy())#
        #tangent_kernel_alignment_log.append(tangent_kernel_alignment.detach().numpy())

        #trans_k_j_alignment_log.append(trans_k_j_alignment.detach().numpy())
        #trans_r_j_alignment_log.append(trans_r_j_alignment.detach().numpy())
        #a_j_alignment_log.append(a_j_alignment.detach().numpy())
        representation_alignment_log_1.append(representation_alignment_1.detach().numpy())
        #representation_alignment_log_2.append(representation_alignment_2)
        #output_weight_alignment_log.append(output_weight_alignment.detach().numpy())
        effective_weight_change_norm_log.append(effective_weight_change_norm.detach().numpy())
        #transform_to_asc_r_alignment_log.append(transform_to_asc_r_alignment.detach().numpy())
        #np.savetxt(f"./analysis/tau/{args.model}_{version}_SS.csv", sign_similarity_log, delimiter=",")
        #spectral_similarity_log.append(spectral_similarity)
        #normalized_sensitivity_log.append(normalized_sensitivity)
        #thresh_alignment_log.append(thresh_alignment.detach().numpy())
        #trans_k_m_alignment_log.append(trans_k_m_alignment.detach().numpy())
        #input_weight_alignment_log.append(input_weight_alignment.detach().numpy())
        #hidden_weight_change_log.append(hidden_weight_change)
        Jacobian_change_norm_log.append(Jacobian_change_norm.detach().numpy())
        #eff_weight_cos_sim_log.append(eff_weight_cos_sim)
        #Jacobian_cos_sim_log.append(Jacobian_cos_sim)

        trainer.save_checkpoint(os.path.join(trainer.logger.log_dir, "checkpoints", "last.ckpt"))
        trainer.test(datamodule=data_module, ckpt_path=os.path.join(trainer.logger.log_dir, "checkpoints", "last.ckpt"))

    np.savetxt(f"./analysis/sigma_v/lheta_lmnist_30/{args.model}_{version}_Sigma.csv", Sigma_log, delimiter=",")
    #np.savetxt(f"./analysis/sigma_v/{args.model}_{version}_TKA.csv", tangent_kernel_alignment_log, delimiter=",")
    #np.savetxt(f"./analysis/sigma_v/{args.model}_{version}_SS_1.csv", sign_similarity_log_1, delimiter=",")
    #np.savetxt(f"./analysis/sigma_v/{args.model}_{version}_SS_2.csv", sign_similarity_log_2, delimiter=",")
    np.savetxt(f"./analysis/sigma_v/lheta_lmnist_30/{args.model}_{version}_RA_1.csv", representation_alignment_log_1, delimiter=",")
    #np.savetxt(f"./analysis/sigma_v/lheta_lmnist_30/{args.model}_{version}_RA_2.csv", representation_alignment_log_2, delimiter=",")
    #np.savetxt(f"./analysis/sigma_v/lheta_lmnist_30/{args.model}_{version}_WMA.csv", weight_matrix_alignment_log, delimiter=",")
    #np.savetxt(f"./analysis/sigma_v/lheta_lmnist_30/{args.model}_{version}_OWA.csv", output_weight_alignment_log, delimiter=",")
    #np.savetxt(f"./analysis/sigma_v/lheta_lmnist_30/{args.model}_{version}_IWA.csv", input_weight_alignment_log, delimiter=",")
    #np.savetxt(f"./analysis/sigma_v/lheta_lmnist_30/{args.model}_{version}_TTARA.csv", transform_to_asc_r_alignment_log, delimiter=",")
    #np.savetxt(f"./analysis/sigma_v/lheta_lmnist_30/{args.model}_{version}_SS.csv", spectral_similarity_log, delimiter=",")
    #np.savetxt(f"./analysis/sigma_v/lheta_lmnist_30/{args.model}_{version}_NS.csv", normalized_sensitivity_log, delimiter=",")
    #np.savetxt(f"./analysis/sigma_v/lheta_lmnist_30/{args.model}_{version}_TKJA.csv", trans_k_j_alignment_log, delimiter=",")
    #np.savetxt(f"./analysis/sigma_v/lheta_lmnist_30/{args.model}_{version}_TRJA.csv", trans_r_j_alignment_log, delimiter=",")
    #np.savetxt(f"./analysis/sigma_v/lheta_lmnist_30/{args.model}_{version}_AJA.csv", a_j_alignment_log, delimiter=",")
    #np.savetxt(f"./analysis/sigma_v/lheta_lmnist_30/{args.model}_{version}_TA.csv", thresh_alignment_log, delimiter=",")
    #np.savetxt(f"./analysis/sigma_v/lheta_lmnist_30/{args.model}_{version}_TKMA.csv", trans_k_m_alignment_log, delimiter=",")
    #np.savetxt(f"./analysis/sigma_v/lheta_lmnist_30/{args.model}_{version}_HWC.csv", hidden_weight_change_log, delimiter=",")
    np.savetxt(f"./analysis/sigma_v/lheta_lmnist_30/{args.model}_{version}_JCN.csv", Jacobian_change_norm_log, delimiter=",")
    #np.savetxt(f"./analysis/sigma_v/lheta_lmnist_30/{args.model}_{version}_EWCS.csv", eff_weight_cos_sim_log, delimiter=",")
    #np.savetxt(f"./analysis/sigma_v/lheta_lmnist_30/{args.model}_{version}_JCS.csv", Jacobian_cos_sim_log, delimiter=",")
    np.savetxt(f"./analysis/sigma_v/lheta_lmnist_30/{args.model}_{version}_EWCN.csv", effective_weight_change_norm_log, delimiter=",")

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
    parser.add_argument("--task", type=str, choices=["sine", "nmnist", "pmnist", "lmnist", "lmnist_1"])
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ntrials", type=int, default=1)
    # parser.add_argument("--model", type=str, choices=["rnn", "lstmn", "glifr_lheta", "glifr_fheta", "glifr_rheta", "glifr_homa", "glifr_lhet", "glifr_fhet", "glifr_rhet", "glifr_hom"])

    # add model specific args
    args = parser.parse_args()

    for i in range(args.ntrials):
        model_names = ["rnn", "lstmn", "glifr_lheta", "glifr_fheta", "glifr_rheta", "glifr_homa", "glifr_lhet", "glifr_fhet", "glifr_rhet", "glifr_hom"]
        ckpt_dict = {}
        to_train = ["glifr_lheta", "glifr_rheta"] #glifr_fheta glifr_rheta glifr_homa #glifr_lheta
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
            print(ckpt_dict["glifr_lheta"])
            args.__dict__["ckpt_path"] = os.path.join(ckpt_dict["glifr_lheta"], "checkpoints", "last.ckpt")
            train(args, version=i)
