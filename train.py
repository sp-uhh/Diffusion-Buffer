import argparse
from argparse import ArgumentParser
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, Callback
import wandb
import subprocess

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.model import ScoreModel


def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups


class GPUMonitorCallback(Callback):
     '''
     Prints nvidia-smi every `interval` training batches.
     '''
     def __init__(self, interval: int = 0):
          super().__init__()
          self.interval = interval

     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args):
          if self.interval and batch_idx % self.interval == 0:
               try:
                    smi = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT).decode()
                    print(f'\n[GPU usage at batch {batch_idx}]\n{smi}')
               except Exception as e:
                    print(f"Failed to run nvidia-smi: {e}")


if __name__ == '__main__':
     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--backbone_score", type=str, choices=["none"] + BackboneRegistry.get_all_names(), default="ncsnpp_causal")
          parser_.add_argument("--pretrained_score", default=None, help="checkpoint for score")
          parser_.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve")
          parser_.add_argument("--no_wandb", action='store_true', help="Turn off logging to W&B, using local default logger instead")
          parser_.add_argument("--nolog", action='store_true', help="Turn off logging (for development purposes)")
          parser_.add_argument("--wandb_name", type=str, default="random", help="Name for wandb logger, if set to random, will generate a random name")
          parser_.add_argument("--wandb_project_name", type=str, default="test", help="Name for wandb logger, if set to random, will generate a random name")
          parser_.add_argument("--ckpt_destination", type=str, default="")
          parser_.add_argument("--wandb_entity", type=str, default="bunlong")
          parser_.add_argument("--grad_clip_val", type=float, default=200.)
          parser_.add_argument("--run_id", type=str,  default='0', help="run id of wandb. If 0 then wandb runs will not be resume")
          parser_.add_argument('--nvidia_smi_interval', type=int, default=0, help='Print `nvidia-smi` every N batches during training (0 to disable)')


          
     temp_args, _ = base_parser.parse_known_args()

     model_cls = ScoreModel

     # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
     backbone_cls_score = BackboneRegistry.get_by_name(temp_args.backbone_score) if temp_args.backbone_score != "none" else None
     sde_class = SDERegistry.get_by_name(temp_args.sde)
     parser = pl.Trainer.add_argparse_args(parser)
     model_cls.add_argparse_args(
          parser.add_argument_group(model_cls.__name__, description=model_cls.__name__))
     sde_class.add_argparse_args(
          parser.add_argument_group("SDE", description=sde_class.__name__))
               
     if temp_args.backbone_score != "none":
          backbone_cls_score.add_argparse_args(
               parser.add_argument_group("BackboneScore", description=backbone_cls_score.__name__))
     else:
          parser.add_argument_group("BackboneScore", description="none")

     # Add data module args
     data_module_cls = SpecsDataModule
     data_module_cls.add_argparse_args(
          parser.add_argument_group("DataModule", description=data_module_cls.__name__))
     # Parse args and separate into groups
     args = parser.parse_args()
     arg_groups = get_argparse_groups(parser)

 #    if Repository('.').head.shorthand != args.git_branch:
 #         raise ValueError('Enter correct git branch (only used for logs)')

     model = model_cls(
          backbone=args.backbone_score, sde=args.sde, data_module_cls=data_module_cls,
          **{
               **vars(arg_groups['ScoreModel']),
               **vars(arg_groups['SDE']),
               **vars(arg_groups['BackboneScore']),
               **vars(arg_groups['DataModule'])
          },
          nolog=args.nolog
     )

     # Set up logger configuration
     #if args.no_wandb:
     #     logger = TensorBoardLogger(save_dir="logs", name="tensorboard")
     #elif args.nolog:
     #     logger = None
     #else:
     #     logger = WandbLogger(project="causal_ot", entity="makarov321-", name=args.wandb_name, log_model=True, save_dir="logs")
     #     logger.experiment.log_code(".")

     #
     #if logger != None:
     #     if os.path.isdir('/data1/causal_ot_logs'):
     #          savedir_ck = f'/data1/causal_ot_logs/logs/{logger.version}'
     #          if not os.path.isdir(savedir_ck):
     #                    os.makedirs(os.path.join(savedir_ck))
     #     else:
     #          savedir_ck = f'logs/{logger.version}'
          
          
          
     if args.pretrained_score is None:
          # Initialize logger, trainer, model, datamodule
          model = model_cls(
               backbone=args.backbone_score, sde=args.sde, data_module_cls=data_module_cls,
               **{
                    **vars(arg_groups['ScoreModel']),
                    **vars(arg_groups['SDE']),
                    **vars(arg_groups['BackboneScore']),
                    **vars(arg_groups['DataModule'])
               },
               nolog=args.nolog
          )
     else:
        checkpoint = torch.load(args.pretrained_score, weights_only=False)
        model = model_cls(
               backbone=args.backbone_score, sde=args.sde, data_module_cls=data_module_cls,
               **{
                    **vars(arg_groups['ScoreModel']),
                    **vars(arg_groups['SDE']),
                    **vars(arg_groups['BackboneScore']),
                    **vars(arg_groups['DataModule'])
               },
               nolog=args.nolog
          )
        model.load_state_dict(checkpoint['state_dict'])

     
     
     
 
     if not args.nolog:
          if args.wandb_name == "random" and args.run_id == '0':
               logger = pl.loggers.WandbLogger(project=args.wandb_project_name, log_model=False, save_dir="logs", entity=args.wandb_entity)
               logger.experiment.log_code(".")  # needed here because otherwise logger.version is None
               save_check_folder = logger.version
          elif args.run_id == '0':
               logger = pl.loggers.WandbLogger(project=args.wandb_project_name, log_model=False, save_dir="logs", name=args.wandb_name, entity=args.wandb_entity)
               logger.experiment.log_code(".")  # needed here because otherwise logger.version is None
               save_check_folder = logger.version
          elif args.run_id != '0':
               logger = pl.loggers.WandbLogger(project=args.wandb_project_name, log_model=False, save_dir="logs", id=args.run_id, resume="must", entity=args.wandb_entity)
               logger.experiment.log_code(".")  # needed here because otherwise logger.version is None
               save_check_folder = args.run_id

          if logger.version != None:
               if args.ckpt_destination == "":
                    savedir_ck = f'/checkpoints/logs_debugging/{save_check_folder}'
                    savedir_ck = os.path.join(os.getcwd() + savedir_ck)
                    if not os.path.isdir(savedir_ck):
                         os.makedirs(savedir_ck)
               else:
                    savedir_ck = os.path.join(args.ckpt_destination, save_check_folder)
                    print(f'saving checkpoints to {savedir_ck}')
                    if not os.path.isdir(savedir_ck):
                         os.makedirs(os.path.join(savedir_ck))
     else:
          logger = None
          
          
          
          
     # Set up callbacks for logger
     gpu_mon = GPUMonitorCallback(args.nvidia_smi_interval)
     if logger is not None:
          progress_bar = TQDMProgressBar(refresh_rate=50)
          callbacks = [progress_bar]
          callbacks += [ModelCheckpoint(dirpath=f"{savedir_ck}", save_last=True, filename='{epoch}-last')]
          callbacks += [ModelCheckpoint(dirpath=f"{savedir_ck}", filename='train_steps={step}', every_n_train_steps=100000, save_top_k=-1)]
          if args.num_eval_files:
               checkpoint_callback_pesq = ModelCheckpoint(dirpath=f"{savedir_ck}", 
                    save_top_k=1, monitor="pesq", mode="max", filename='{epoch}-pesq={pesq:.2f}')
               callbacks += [checkpoint_callback_pesq]
               checkpoint_callback_si_sdr = ModelCheckpoint(dirpath=f"{savedir_ck}", 
                    save_top_k=1, monitor="si_sdr", mode="max", filename='{epoch}-si_sdr={si_sdr:.2f}')
               callbacks += [checkpoint_callback_si_sdr]
     else:
          callbacks = []
     callbacks.append(gpu_mon)

     # Initialize the Trainer and the DataModule
     trainer = pl.Trainer.from_argparse_args(
          arg_groups['pl.Trainer'],
          accelerator="gpu",
          strategy=DDPStrategy(find_unused_parameters=True), 
          logger=logger,
          log_every_n_steps=10, num_sanity_val_steps=0,
          callbacks=callbacks,
          gradient_clip_val=args.grad_clip_val,
          gradient_clip_algorithm='value'
     )

     # Train model
     trainer.fit(model)
