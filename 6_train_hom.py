from argparse import ArgumentParser
import yaml

import pytorch_lightning as pl
import torch

from datasets.sine import SineDataModule
from datasets.mnist import MNISTDataModule
from datasets.nmnist import NMNISTDataModule
from datasets.utils import add_data_args
from models.pl_modules_1 import add_structure_args, add_general_model_args, add_training_args, GLIFRN, RNN, LSTMN
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
    params = [p for p in params if p.requires_grad]
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

def compute_ntk(net, x):    
    """ Compute the Neural Tangent Kernel (NTK) of a network
    Args:
        net: the network
        x: input Tensor
    Returns:
        NTK matrix with shape (..., M, M)
    """
    # Ensure that the network has a forward method that returns both the output and the hidden activity
    _, voltages, ascs, syns, hidden_activity = net(x)
    
    # Extract the hidden activity of the last step
    hidden_activity_last_step = hidden_activity[:,-1,:]  # Get the last step's activity
    
    # Compute the Jacobian of the hidden activity of the last step with respect to the network parameters
    J_hidden_last_step = jacobian(hidden_activity_last_step, net.parameters())
    
    # Compute the NTK as the product of the Jacobian with itself, transposed
    ntk = torch.einsum('ij,kj->ik', J_hidden_last_step, J_hidden_last_step)
    
    return ntk

def train(args, version): 
    # The order of priority for the configurations are first
    # the task related yaml and then the model realted yaml.
    print(f"training {args.model} on {args.task} on trial {version}")

    args.__dict__["default_root_dir"] = os.path.abspath(f"results/{args.task}-agn/{args.model}/trial_{version}")
    # if not os.path.exists(args.default_root_dir):
    os.makedirs(args.default_root_dir, exist_ok=True)
    if args.model == "rnn":
        args = read_yaml_into_args(args, "./config/rnn.yaml", args.model)
        model_class = RNN
    elif args.model == "lstmn":
        args = read_yaml_into_args(args, "./config/lstmn.yaml", args.model)
        model_class = LSTMN
    elif args.model.startswith("glifr"):
        args = read_yaml_into_args(args, "./config/glifr.yaml", args.model)
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
    
    args = read_yaml_into_args(args, f"./config/{args.task}_task.yaml", args.model)

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

    k_m_log = []
    sign_similarity_log_1 = [] 
    #sign_similarity_log_2 = [] 
    representation_alignment_log_1 = []
    tangent_kernel_alignment_log = []
    k_m_init = 1
    k_m_inc = 1
    k_m_end = k_m_init+(13*k_m_inc)
    
    for k_m in range(k_m_init, k_m_end, k_m_inc): #tau sigma_v R I_0
        # Analysis params 
        args.__dict__["tau"] = 0.5 #0.5
        args.__dict__["sigma_v"] = 1 #1
        args.__dict__["R"] = 0.1
        args.__dict__["I_0"] = 0
        args.__dict__["V_reset"] = 0
        args.__dict__["k_m"] = k_m / 100.0
        print("********************************************************")
        print("k_m = ",args.k_m)
        print("********************************************************")
        #args.__dict__["max_epochs"] = 2

        # tb_logger = TensorBoardLogger(str(version))
        # args.__dict__["logger"] = tb_logger
        trainer = pl.Trainer.from_argparse_args(args)
        #trainer = pl.Trainer.from_argparse_args(args,callbacks=[early_stopping] + args.__dict__["callbacks"])

        if k_m == k_m_init:
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
            inputs_0 = torch.tensor(inputs)
        #print(inputs_0.shape)  # To verify the shape
        #inputs_0 = inputs_0[:1,:,:]


        model.reset_state(inputs_0.size()[0])


        # Initial Hidden State/Voltages
        #if args.model in ["rnn", "lstmn"]:
        #    _, voltages_0 = model(inputs_0)#.forward(inputs_0, track=True)
        #    voltages_0 = voltages_0.detach().numpy()
        #    np.savetxt(f"./analysis/{args.model}_{version}_voltage_0.csv", voltages_0[0,:,0], delimiter=",")
        if args.model in ["glifr_lheta", "glifr_fheta", "glifr_rheta", "glifr_homa", "glifr_lhet", "glifr_fhet", "glifr_rhet", "glifr_hom"]:
            _, voltages_0, ascs_0, syns_0, activity_0 = model(inputs_0) #model.forward(inputs_0, track=True) #
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
            _, voltages_f, ascs_f, syns_f, activity_f = model(inputs_0)#model.forward(inputs_0, track=True) #
            #np.savetxt(f"./analysis/{args.model}_{version}_tau={args.tau}_sigma_v={args.sigma_v}_R={args.R}_I0={args.I0}_syn_f.csv", syns_f[0,:,0], delimiter=",")
            #np.savetxt(f"./analysis/{args.model}_{version}_tau={args.tau}_sigma_v={args.sigma_v}_R={args.R}_I0={args.I0}_voltage_f.csv", voltages_f[0,:,0], delimiter=",")
            #np.savetxt(f"./analysis/{args.model}_{version}_tau={args.tau}_sigma_v={args.sigma_v}_R={args.R}_I0={args.I0}_asc_f.csv", ascs_f[:,0,:,0], delimiter=",")

        # Calculate Tangent Kernel Alignment
        #print(activity_f.shape)
        ntk_f = compute_ntk(model, inputs_0)
        tangent_kernel_alignment = torch.trace(ntk_0 @ ntk_f.T) / (torch.norm(ntk_0) * torch.norm(ntk_f))
        
                
        # Calculate Sign Similarity
        sign_similarity_1 = torch.sum(torch.sign(activity_0[:,-1,:]) == torch.sign(activity_f[:,-1,:])) / (activity_0[:,-1,:].size()[0] * activity_f[:,-1,:].size()[1])
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
        
        k_m_log.append(k_m)
        sign_similarity_log_1.append(sign_similarity_1.numpy())
        #sign_similarity_log_2.append(sign_similarity_2)#(sign_similarity.numpy())#
        tangent_kernel_alignment_log.append(tangent_kernel_alignment.detach().numpy())
        representation_alignment_log_1.append(representation_alignment_1.detach().numpy())
        #np.savetxt(f"./analysis/tau/{args.model}_{version}_SS.csv", sign_similarity_log, delimiter=",")

        trainer.save_checkpoint(os.path.join(trainer.logger.log_dir, "checkpoints", "last.ckpt"))
        trainer.test(datamodule=data_module, ckpt_path=os.path.join(trainer.logger.log_dir, "checkpoints", "last.ckpt"))

    np.savetxt(f"./analysis/k_m/{args.model}_{version}_k_m.csv", k_m_log, delimiter=",")
    np.savetxt(f"./analysis/k_m/{args.model}_{version}_TKA.csv", tangent_kernel_alignment_log, delimiter=",")
    np.savetxt(f"./analysis/k_m/{args.model}_{version}_SS_1.csv", sign_similarity_log_1, delimiter=",")
    #np.savetxt(f"./analysis/I_0/{args.model}_{version}_SS_2.csv", sign_similarity_log_2, delimiter=",")
    np.savetxt(f"./analysis/k_m/{args.model}_{version}_RA.csv", representation_alignment_log_1, delimiter=",")

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
        to_train = ["glifr_hom"] #glifr_fheta glifr_rheta glifr_lheta
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
