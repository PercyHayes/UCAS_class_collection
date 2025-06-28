#python ./transformers/examples/question-answering_mlu/run_squad.py \
python ./run_squad.py \
  --model_type bert \
  --model_name_or_path ./output \
  --do_eval \
  --fp16 \
  --do_lower_case \
  --predict_file ./squad/dev-v1.1.json \
  --max_seq_length 384 \
  --doc_stride 128 \
  --overwrite_output_dir \
  --output_dir ./output
