singularity shell --nv \
    --bind $PWD:/workdir \
    --bind /mnt/scratch/xiayao/cache/HF:/hf \
    --env HF_HOME=/hf \
    --workdir /workdir \
    /mnt/scratch/xiayao/cache/images/rayllm.sif

# cd /workdir && serve run 7b.yaml