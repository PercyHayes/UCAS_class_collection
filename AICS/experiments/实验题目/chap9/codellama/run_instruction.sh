torchrun --nproc_per_node 1 example_instructions_mlu.py \
    --ckpt_dir /workspace/model/favorite/large-scale-models/model-v1/CodeLlama-7b-Instruct/ \
    --tokenizer_path /workspace/model/favorite/large-scale-models/model-v1/CodeLlama-7b-Instruct/tokenizer.model \
    --max_seq_len 512 --max_batch_size 4