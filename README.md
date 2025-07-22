# Diffusion Buffer: Online Diffusion-based Speech Enhancement with Sub-Second Latency

This repository contains the official PyTorch implementations for the papers:

- Bunlong Lay, Rostislav Makarov, Timo Gerkmann [*"Diffusion Buffer: Online Diffusion-based Speech Enhancement with Sub-Second Latency"*](https://arxiv.org/abs/2506.02908), ISCA Interspeech, Rotterdam, August 2025. [[bibtex]](#citations--references)


## Installation

-TODO: add requirements.txt


## Pretrained checkpoints

-Find here the ckpt of BBED with B = 30 reverse steps/buffer length from the paper when trained on filtered EARS-WHAM: https://drive.google.com/drive/folders/1NjNiPj42wZ6NyQT4ykZcwxNjhqkbnLD9?usp=drive_link. See under eval how to evaluate this.
      


## Training

Training is done by executing `train.py`. A minimal running example with default settings (as in our paper [1]) can be run with

```bash
python train.py --base_dir <your_base_dir>
```

where `your_base_dir` should be a path to a folder containing subdirectories `train/` and `valid/` (optionally `test/` as well). Each subdirectory must itself have two subdirectories `clean/` and `noisy/`, with the same filenames present in both. We currently only support training with `.wav` files.

To see all available training options, run `python train.py --help`. Note that the available options for the SDE and the backbone network change depending on which SDE and backbone you use. These can be set through the `--sde` and `--backbone` options.



More advanced example (this will reproduce BBED with B = 30 reverse steps/buffer length from the paper when trained on filtered EARS-WHAM)

```bash
python train.py --base_dir <enterpath> --batch_size 32 --backbone ncsnpp --sde bbed 
--t_eps 0.03 --num_eval_files 3 --spec_abs_exponent 0.5 --spec_factor 0.15 --loss_abs_exponent 1 --loss_type mse --theta 0.08 --k 2.6 --timestep_type_inf default 
--wandb_name <entername> --fs 16000 --audiologs_every_epoch 25 --speclogs_every_epoch 25 --save_every_n_epochs 0 --wandb_project_name <entername> --ch_mult 1 2 2 2
--hop_length 256 --n_fft 510 --num_frames 128 --normalize not --output_scale time --num_res_blocks 1 --format noise
--diff_gate_range 30 31 --nf 96 --wandb_entity <entername> --diff_gate_length_inference 30 --T_sampling 0.8
```

## Evaluation

To evaluate on a test set, run
```bash
python --test_dir <path_to_testdir>
            --experiments_folder <parent_folder>
            --destination_folder <subfolder>
            --reverse_starting_point <enter_rsp_from_ckpt> --N <enter_bufferlength_from_ckpt>
            --ckpt <path_to_ckpt.ckpt>
```

to generate the enhanced .wav files and compute metrics on the enhanced files (the test_dir must contain noisy and clean) The enhanced files are saved in parent_folder/subfolder given by experiments_folder and destination_folder. Note that this script evaluates in an online fashion and therefore takes more time to process than offline processing. For the provided ckpt above please set

```bash
--reverse_starting_point 0.8 --N 30
``




## Citations / References

We kindly ask you to cite our papers in your publication when using any of our research or code:
```bib
@inproceedings{lay24diffbuffer,
  author={Bunlong Lay and Rostislav Makarov and Timo Gerkmann},
  title={Diffusion Buffer: Online Diffusion-based Speech Enhancement with Sub-Second Latency},
  year={2025},
  booktitle={Proc. Interspeech 2025},
}
```


