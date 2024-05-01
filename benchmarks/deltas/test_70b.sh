echo "--------- TP=8 (bitblas, 4bits) ---------"
CUDA_LAUNCH_BLOCKING=1 BITWIDTH=4 TP_SIZE=8 USE_BITBLAS=1 python3 benchmarks/deltas/benchmark_70b.py