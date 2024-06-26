from argparse import ArgumentParser
import yaml

import pytorch_lightning as pl
import torch

from datasets.sine import SineDataModule
from datasets.mnist import MNISTDataModule
from datasets.nmnist import NMNISTDataModule
from datasets.utils import add_data_args
from models.pl_modules_6_3 import add_structure_args, add_general_model_args, add_training_args, GLIFRN, RNN, LSTMN
from training.callbacks import *
from utils.types import NeuronParameters
from pytorch_lightning.callbacks import EarlyStopping
import sys
from lyapunov.LEs_3 import calculateLEs, jacobian_eigenspectrum
from scipy import linalg
import matplotlib.pyplot as plt
import numpy

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
    
    args = read_yaml_into_args(args, f"./../config/{args.task}_task.yaml", args.model)

    # add all the available trainer options to argparse
    # ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
    args.__dict__["callbacks"] = [eval(c)() for c in args.__dict__["callbacks"]]

    
    #args.__dict__["max_epochs"] = 2

    # tb_logger = TensorBoardLogger(str(version))
    # args.__dict__["logger"] = tb_logger
    trainer = pl.Trainer.from_argparse_args(args)
    #trainer = pl.Trainer.from_argparse_args(args,callbacks=[early_stopping] + args.__dict__["callbacks"])

    delattr(args, 'callbacks')
    # delattr(args, 'logger')
    model = model_class(**vars(args))

    print(f"training {args.model} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    data_module = data_modules[args.task](**vars(args))

    data_module.setup()

    les, jacobian_norms, max_eigenvalues, derivative_history = calculateLEs(model, num_steps=28) #num_steps=100
    np.savetxt(f"./analysis/lyapunov/{args.model}_LEs.csv", les, delimiter=",")
    np.savetxt(f"./analysis/lyapunov/{args.model}_jacobian_norms.csv", jacobian_norms, delimiter=",")
    np.savetxt(f"./analysis/lyapunov/{args.model}_max_eigenvalues.csv", max_eigenvalues, delimiter=",")
    #np.savetxt(f"./analysis/lyapunov/{args.model}_sigma_v=_{args.sigma_v}_derivative_history.csv", derivative_history, delimiter=",")
    
    trainer.fit(model, data_module)

    les, jacobian_norms, max_eigenvalues, derivative_history = calculateLEs(model, num_steps=28) #num_steps=100
    np.savetxt(f"./analysis/lyapunov/{args.model}_after_LEs.csv", les, delimiter=",")
    np.savetxt(f"./analysis/lyapunov/{args.model}_after_jacobian_norms.csv", jacobian_norms, delimiter=",")
    np.savetxt(f"./analysis/lyapunov/{args.model}_after_max_eigenvalues.csv", max_eigenvalues, delimiter=",")

    #les, jacobian_norms, max_eigenvalues = calculateLEs(model, num_steps=100)
    #np.savetxt(f"./analysis/lyapunov/{args.model}_{version}_sigma_v=_{args.sigma_v}_LEs_after.csv", les, delimiter=",")
    #np.savetxt(f"./analysis/lyapunov/{args.model}_{version}_sigma_v=_{args.sigma_v}_jacobian_norms_after.csv", jacobian_norms, delimiter=",")
    #np.savetxt(f"./analysis/lyapunov/{args.model}_{version}_sigma_v=_{args.sigma_v}_max_eigenvalues_after.csv", max_eigenvalues, delimiter=",")
    
    trainer.save_checkpoint(os.path.join(trainer.logger.log_dir, "checkpoints", "last.ckpt"))
    trainer.test(datamodule=data_module, ckpt_path=os.path.join(trainer.logger.log_dir, "checkpoints", "last.ckpt"))
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
        to_train = ["rnn"] #glifr_fheta glifr_rheta glifr_homa #glifr_lheta
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
