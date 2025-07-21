import argparse
from argparse import ArgumentParser
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import platform

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

class Trainer(pl.Trainer):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--devices", type=int, default=1, help="number of devices (gpus)")
        parser.add_argument("--strategy", type=str, default="ddp", help="strategy name")
        parser.add_argument("--accumulate_grad_batches", type=int, default=1)
        parser.add_argument("--max_epochs", type=int, default=-1)
        return parser


if __name__ == '__main__':
     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
          parser_.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve")
          parser_.add_argument("--pre_ckpt", type=str, default=None, help="Load ckpt, should match with wandb run id to resume wandb plot correctly")
          parser_.add_argument("--wandb_entity", type=str, default="enter-your-wandb-name")      
          parser_.add_argument("--nolog", action='store_true', help="Turn off logging (for development purposes)")
          parser_.add_argument("--save_every_n_epochs", type=int, default=0, help="0 means its turned off") 
          parser_.add_argument("--wandb_name", type=str, default="random", help="Name for wandb logger, if set to random, will generate a random name")
          parser_.add_argument("--wandb_project_name", type=str, default="default", help="Wandb project name")
          parser_.add_argument("--ckpt_destination", type=str, default="")
          parser_.add_argument("--gradclip", type=float, default=0)
          parser_.add_argument("--detect_anomaly", action='store_true', default=False)
          parser_.add_argument("--run_id", type=str,  default='0', help="run id of wandb. If 0 then wandb runs will not be resume instead new training starts")
     temp_args, _ = base_parser.parse_known_args()


     # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
     backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
     sde_class = SDERegistry.get_by_name(temp_args.sde)
     #parser = pl.Trainer.add_argparse_args(parser)
     Trainer.add_argparse_args(parser.add_argument_group("Trainer", description=Trainer.__name__))
     ScoreModel.add_argparse_args(
          parser.add_argument_group("ScoreModel", description=ScoreModel.__name__))
     sde_class.add_argparse_args(
          parser.add_argument_group("SDE", description=sde_class.__name__))
     backbone_cls.add_argparse_args(
          parser.add_argument_group("Backbone", description=backbone_cls.__name__))
     # Add data module args
     data_module_cls = SpecsDataModule
     data_module_cls.add_argparse_args(
          parser.add_argument_group("DataModule", description=data_module_cls.__name__))
     # Parse args and separate into groups
     args = parser.parse_args()
     arg_groups = get_argparse_groups(parser)


     
     
     
     if args.pre_ckpt is None:
     # Initialize logger, trainer, model, datamodule
          model = ScoreModel(
               backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
               **{
                    **vars(arg_groups['ScoreModel']),
                    **vars(arg_groups['SDE']),
                    **vars(arg_groups['Backbone']),
                    **vars(arg_groups['DataModule'])
               }
          )
     else:
        model = ScoreModel.load_from_checkpoint(checkpoint_path=args.pre_ckpt, base_dir=args.base_dir)

 
     if not args.nolog:
          if args.wandb_name == "random" and args.run_id == '0':
               logger = pl.loggers.WandbLogger(project=args.wandb_project_name, log_model=False, save_dir="logs",  entity=args.wandb_entity)
               logger.experiment.log_code(".")  # needed here because otherwise logger.version is None
               save_check_folder = logger.version
          elif args.run_id == '0':
               logger = pl.loggers.WandbLogger(project=args.wandb_project_name, log_model=False, save_dir="logs", name=args.wandb_name,  entity=args.wandb_entity)
               logger.experiment.log_code(".")  # needed here because otherwise logger.version is None
               save_check_folder = logger.version
          elif args.run_id != '0':
               logger = pl.loggers.WandbLogger(project=args.wandb_project_name, log_model=False, save_dir="logs", id=args.run_id, resume="must",  entity=args.wandb_entity)
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



     torch.set_float32_matmul_precision('medium')


     # Set up callbacks for logger
     if args.num_eval_files and logger != None:
          callbacks = [ModelCheckpoint(dirpath=savedir_ck, save_last=True, filename='{epoch}-last')]
          checkpoint_callback_last = ModelCheckpoint(dirpath=savedir_ck,
               save_last=True, filename='{epoch}-last')
          checkpoint_callback_pesq = ModelCheckpoint(dirpath=savedir_ck,
               save_top_k=1, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}')

          
          if args.save_every_n_epochs > 0:
               checkpoint_n_epochs = ModelCheckpoint(dirpath=savedir_ck, filename='{epoch:02d}', 
                                                       every_n_epochs = args.save_every_n_epochs, 
                                                       save_top_k = -1)
               callbacks = [checkpoint_callback_last, checkpoint_callback_pesq,
               checkpoint_n_epochs]
          else:
               callbacks = [checkpoint_callback_last, checkpoint_callback_pesq]
     # Initialize the Trainer and the DataModule
     if logger != None:
          trainer = Trainer(**vars(arg_groups['Trainer']),
                          accelerator="gpu", gradient_clip_val=args.gradclip, detect_anomaly=args.detect_anomaly,
                          log_every_n_steps=100,
                          num_sanity_val_steps=0, callbacks=callbacks, logger=logger)
          
     else:
          trainer = Trainer(**vars(arg_groups['Trainer']),
                          accelerator="gpu", gradient_clip_val=args.gradclip, detect_anomaly=args.detect_anomaly,
                          log_every_n_steps=100,
                          num_sanity_val_steps=0)


     # Train model
     trainer.fit(model)
