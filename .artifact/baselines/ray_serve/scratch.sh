singularity shell --nv \
    --bind $PWD:/workdir \
    --bind /mnt/scratch/xiayao/cache/HF:/hf \
    --env TRANSFORMERS_CACHE=/hf \
    /mnt/scratch/xiayao/cache/images/rayllm.sif