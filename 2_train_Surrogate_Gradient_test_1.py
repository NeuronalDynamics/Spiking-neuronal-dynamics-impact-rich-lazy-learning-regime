from argparse import ArgumentParser
import yaml

import pytorch_lightning as pl
import torch

from datasets.sine import SineDataModule
from datasets.mnist import MNISTDataModule
from datasets.nmnist import NMNISTDataModule
from datasets.utils import add_data_args
from models.pl_modules_8 import add_structure_args, add_general_model_args, add_training_args, GLIFRN, RNN, LSTMN
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
    
    args = read_yaml_into_args(args, f"./../config/{args.task}_1_task.yaml", args.model)
    args.__dict__["callbacks"] = [eval(c)() for c in args.__dict__["callbacks"]]

    Sigma_log = []
    representation_alignment_log_1 = []
    weight_matrix_change_log = []
    effective_weight_change_norm_log = []
    Jacobian_change_norm_log = []

    Sigma_init = 100
    Sigma_inc = 100
    Sigma_end = Sigma_init+(20*Sigma_inc)
    
    for Sigma in range(Sigma_init, Sigma_end, Sigma_inc): #tau sigma_v R I_0
        args.__dict__["tau"] = 0.5 #0.5
        args.__dict__["sigma_v"] = Sigma / 1000.0 #1
        args.__dict__["R"] = 0.1
        args.__dict__["I_0"] = 0
        print("********************************************************")
        print("Sigma_v = ",args.sigma_v)
        print("********************************************************")

        trainer = pl.Trainer.from_argparse_args(args)
        
        if Sigma == Sigma_init:
            delattr(args, 'callbacks')
        model = model_class(**vars(args))

        print(f"training {args.model} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
        data_module = data_modules[args.task](**vars(args))

        data_module.setup()

        train_loader = data_module.train_dataloader()
        for inputs, _ in train_loader:
            inputs_0 = inputs.clone().detach().requires_grad_(True)

        model.reset_state(inputs_0.size()[0])

        weight_matrix_0 = model.neuron_layer.weight_lat.clone()

        if args.model in ["glifr_lheta", "glifr_fheta", "glifr_rheta", "glifr_homa", "glifr_lhet", "glifr_fhet", "glifr_rhet", "glifr_hom"]:
            _, voltages_0, _, _, activity_0 = model(inputs_0) #model.forward(inputs_0, track=True) #

        trainer.fit(model, data_module)

        weight_matrix_f = model.neuron_layer.weight_lat.clone()

        if args.model in ["glifr_lheta", "glifr_fheta", "glifr_rheta", "glifr_homa", "glifr_lhet", "glifr_fhet", "glifr_rhet", "glifr_hom"]:
            _, voltages_f, _, _, activity_f = model(inputs_0)#model.forward(inputs_0, track=True) #

        print("#########################################################")
        weight_matrix_change = torch.norm(weight_matrix_f - weight_matrix_0, p='fro')
        print(weight_matrix_change)
        print("#########################################################")
        effective_weight_matrix_0 = model.neuron_layer.spike_fn_derivative(voltages_0.reshape(voltages_0.shape[0] * voltages_0.shape[1], voltages_0.shape[2])) @ weight_matrix_0
        effective_weight_matrix_f = model.neuron_layer.spike_fn_derivative(voltages_f.reshape(voltages_f.shape[0] * voltages_f.shape[1], voltages_f.shape[2])) @ weight_matrix_f
        effective_weight_change_norm = torch.norm(effective_weight_matrix_f - effective_weight_matrix_0, p='fro')
        print(effective_weight_change_norm)
        print("#########################################################")
        J_0 = model.neuron_layer.spike_fn_derivative(activity_0[:,-1,:])
        J_f = model.neuron_layer.spike_fn_derivative(activity_f[:,-1,:])
        Jacobian_change_norm = torch.norm(J_f - J_0, p='fro')
        print(Jacobian_change_norm)
        print("#########################################################")
        K_0 = torch.mm(activity_0[:,-1,:].T, activity_0[:,-1,:])
        K_t = torch.mm(activity_f[:,-1,:].T, activity_f[:,-1,:])
        representation_alignment_1 = torch.trace(torch.mm(K_t, K_0)) / (torch.norm(K_t, p='fro') * torch.norm(K_0, p='fro'))
        print(representation_alignment_1)
        print("#########################################################")
        
        Sigma_log.append(Sigma)
        representation_alignment_log_1.append(representation_alignment_1.detach().numpy())
        weight_matrix_change_log.append(weight_matrix_change.detach().numpy())
        effective_weight_change_norm_log.append(effective_weight_change_norm.detach().numpy())
        Jacobian_change_norm_log.append(Jacobian_change_norm.detach().numpy())

        trainer.save_checkpoint(os.path.join(trainer.logger.log_dir, "checkpoints", "last.ckpt"))
        trainer.test(datamodule=data_module, ckpt_path=os.path.join(trainer.logger.log_dir, "checkpoints", "last.ckpt"))

    np.savetxt(f"./analysis/sigma_v/{args.model}_{version}_Sigma.csv", Sigma_log, delimiter=",")
    np.savetxt(f"./analysis/sigma_v/{args.model}_{version}_RA.csv", representation_alignment_log_1, delimiter=",")
    np.savetxt(f"./analysis/sigma_v/{args.model}_{version}_WMC.csv", weight_matrix_change_log, delimiter=",")
    np.savetxt(f"./analysis/sigma_v/{args.model}_{version}_EWCN.csv", effective_weight_change_norm_log, delimiter=",")
    np.savetxt(f"./analysis/sigma_v/{args.model}_{version}_JCN.csv", Jacobian_change_norm_log, delimiter=",")
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
