#python ./transformers/examples/question-answering_mlu/run_squad.py \
python ./run_squad.py \
 --model_type bert \
 --model_name_or_path ./pretrain_bert_base_cased \
 --do_train \
 --fp16 \
 --do_lower_case \
 --train_file ./squad/train-v1.1.json \
 --predict_file ./squad/dev-v1.1.json \
 --per_gpu_train_batch_size 12 \
 --learning_rate 3e-5 \
 --num_train_epochs 1.0 \
 --max_seq_length 384 \
 --doc_stride 128 \
 --logging_steps 2000 \
 --save_steps 2000 \
 --overwrite_output_dir \
 --output_dir ./output
