#!/bin/bash

#SBATCH --job-name=MT-base-finetune-passiv-en
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=30GB 
#SBATCH --time=40:00:00 
#SBATCH --gres=gpu:v100:1

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.2-cudnn7-devel-ubuntu18.04.sif /bin/bash -c "

source /ext3/env.sh
conda activate py38

python ../models/run_seq2seq.py \
    --model_name_or_path 'facebook/mbart-large-cc25' \
    --do_train \
    --task translation_src_to_tgt \
    --random_weights \
    --train_file ../../data/passiv_de_nps/passiv_de_nps.train.json \
    --validation_file ../../data/passiv_de_nps/passiv_de_nps.dev.json \
    --output_dir /scratch/am12057/mbart-cc-random-finetuning-passivization-de-bs128/  \
    --source_prefix de_DE \
    --target_prefix de_DE \
    --per_device_train_batch_size=16 \
    --gradient_accumulation_steps=8 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_steps 20000 \
    --num_train_epochs 10.0
"
