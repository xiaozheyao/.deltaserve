export OPENAI_API_KEY="test"

lm_eval --model openai-completions \
    --use_cache .local/cache/deltaserve.lmsys.vicuna-7b-v1.5 \
    --model_args model=lmsys/vicuna-7b-v1.5,base_url=http://localhost:8000/v1,tokenizer_backend=huggingface, \
    --tasks openbookqa,piqa,boolq,sciq,logiqa,truthfulqa \
    --output_path .local/eval_results/deltaserve.lmsys.vicuna-7b-v1.5