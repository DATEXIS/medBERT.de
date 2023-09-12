#!/bin/bash
#SBATCH --job-name=medbert-gottbert-adaptation
#SBATCH --partition pgpu
#SBATCH --mem=0
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=96:00:00
#SBATCH --output=logs_gottbert.o%j
#SBATCH --error=errors_gottbert.e%j

echo $`nvidia-smi`

cd /sc-scratch/sc-scratch-cc06-medbert/medbert/scripts/pretraining
export TRANSFORMERS_CACHE=/sc-scratch/sc-scratch-cc06-medbert/transformers-cache
export HF_DATASETS_CACHE=/sc-scratch/sc-scratch-cc06-medbert/hf-datasets-cache
source /home/bressekk/miniconda3/etc/profile.d/conda.sh
conda activate medbert

python main.py fit \
	--seed_everything 42 \
	--data.datafiles "/sc-scratch/sc-scratch-cc06-medbert/medbert/datasets/mlm_pretraining_data.csv" \
	--data.num_proc 128 \
	--data.tokenizer_name "uklfr/gottbert-base" \
	--data.dont_overwrite_existing_dataset false \
	--data.batch_size 64 \
	--data.num_workers 16 \
	--data.max_seq_len 512 \
	--data.cache_path "/sc-scratch/sc-scratch-cc06-medbert/gottbert-domain-adapted/cache" \
	--model.hf_checkpoint "uklfr/gottbert-base" \
	--model.warmup_steps 200 \
	--model.decay_steps 1363 \
	--model.lr 4e-3 \
	--model.tokenizer_name "uklfr/gottbert-base" \
	--model.optimizer "lamb" \
	--model.huggingface_save_dir "/sc-scratch/sc-scratch-cc06-medbert/gottbert-domain-adapted/checkpoint" \
	--trainer.precision "bf16" \
	--trainer.accumulate_grad_batches 64 \
	--trainer.accelerator "gpu" \
	--trainer.devices 8 \
	--trainer.strategy "ddp_find_unused_parameters_false" \
	--trainer.benchmark true \
	--trainer.default_root_dir "/sc-scratch/sc-scratch-cc06-medbert/gottbert-domain-adapted/logs" \
	--trainer.callbacks LearningRateMonitor \
	--trainer.callbacks.logging_interval step \
	--trainer.val_check_interval 250 \
	--trainer.log_every_n_steps 1 \
	--trainer.max_steps 1563
