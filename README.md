TODO...

Example how to train:
(this will retrain DB-BBED with buffer length 30 from the paper) 

python train.py --base_dir <enterpath> --batch_size 32 --backbone ncsnpp --sde bbed 
--t_eps 0.03 --num_eval_files 3 --spec_abs_exponent 0.5 --spec_factor 0.15 --loss_abs_exponent 1 --loss_type mse --theta 0.08 --k 2.6 --timestep_type_inf default 
--wandb_name <entername> --fs 16000 --audiologs_every_epoch 25 --speclogs_every_epoch 25 --save_every_n_epochs 0 --wandb_project_name <entername> --ch_mult 1 2 2 2
--hop_length 256 --n_fft 510 --num_frames 128 --normalize not --output_scale time --num_res_blocks 1 --format noise
--diff_gate_range 30 31 --nf 96 --wandb_entity <entername> --diff_gate_length_inference 30 --T_sampling 0.8



Example how to evaluate:
Load ckpt from here TBA and then:

python --test_dir /data/EARS-WHAM-16k-filtered
            --experiments_folder ./enhanced/test
            --destination_folder debug
            --reverse_starting_point 0.8 --N 30
            --ckpt TBA/last.ckpt
