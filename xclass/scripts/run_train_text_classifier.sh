GPU=$1
dataset_name=$2
train_suffix=$3
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
model_name_or_path=bert-base-cased

# this is also defined in utils.py, make sure to change both when changing.
output_dir=../models/${model_name_or_path}_${train_suffix}
#  --max_seq_length 512 \
#  --per_gpu_train_batch_size 32 \
#  --per_gpu_eval_batch_size 32 \
#  --logging_steps 100000 \
#  --learning_rate 5e-5 \
#  --num_train_epochs 3.0 \

CUDA_VISIBLE_DEVICES=$GPU python train_text_classifier.py \
  --model_name_or_path ${model_name_or_path} \
  --task_name ${dataset_name} \
  --train_suffix ${train_suffix} \
  --test_suffix "" \
  --output_dir ${output_dir} \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --learning_rate 5e-5 \
  --num_train_epochs 1.0 \
  --max_seq_length 100 \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 8 \
  --logging_steps 100000 \
  --save_steps -1 
