singularity run --nv --bind $PWD/.artifact:/artifact --bind /scratch/xiayao/models/:/models deltaserve.dev.sif bash /artifact/scripts/sgs/nvme/7b/run_vllm_server_with_lora.sh