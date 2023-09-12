#!/bin/bash
#SBATCH --job-name=medbert
#SBATCH --partition pgpu
#SBATCH --mem=0
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=96:00:00
#SBATCH --output=logs_medbert.o%j
#SBATCH --error=errors_medbert.e%j

echo $`nvidia-smi`

cd /sc-scratch/sc-scratch-cc06-medbert/medbert/scripts/pretraining

source /home/bressekk/miniconda3/etc/profile.d/conda.sh
conda activate medbert

python main.py fit \
		--seed_everything 42 \
		--data.datafiles "/sc-scratch/sc-scratch-cc06-medbert/medbert/datasets/mlm_pretraining_data.csv" \
		--data.num_proc 128 \
		--data.tokenizer_name "/sc-scratch/sc-scratch-cc06-medbert/medbert/bert-base-medical-german/" \
		--data.dont_overwrite_existing_dataset false \
		--data.batch_size 512 \
		--data.num_workers 16 \
		--data.max_seq_len 128 \
		--data.cache_path "/sc-scratch/sc-scratch-cc06-medbert/medbert/cache" \
		--model.warmup_steps 2000 \
		--model.decay_steps 5038 \
		--model.lr 6e-3 \
		--model.tokenizer_name "/sc-scratch/sc-scratch-cc06-medbert/medbert/bert-base-medical-german/" \
		--model.optimizer "lamb" \
		--model.huggingface_save_dir "/sc-scratch/sc-scratch-cc06-medbert/medbert/checkpoint" \
		--trainer.precision "bf16" \
		--trainer.accumulate_grad_batches 16 \
		--trainer.accelerator "gpu" \
		--trainer.devices 8 \
		--trainer.strategy "ddp_find_unused_parameters_false" \
		--trainer.benchmark true \
		--trainer.default_root_dir "/sc-scratch/sc-scratch-cc06-medbert/medbert/logs" \
		--trainer.callbacks LearningRateMonitor \
		--trainer.callbacks.logging_interval step \
		--trainer.val_check_interval 1000 \
		--trainer.log_every_n_steps 1 \
		--trainer.max_steps 7038
