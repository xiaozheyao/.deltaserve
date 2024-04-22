# echo "--------- TP=1 ---------"
# TP_SIZE=1 python benchmarks/deltas/benchmark_7b.py
# echo "--------- Not-split weights TP=2 ---------"
# TP_SIZE=2 UNOPTIMIZED_DELTA=1 python benchmarks/deltas/benchmark_7b.py
# echo "--------- Split weights ---------"
# TP_SIZE=2 python benchmarks/deltas/benchmark_7b.py

echo "--------- TP=1 (triteia) ---------"
TP_SIZE=2 CUDA_LAUNCH_BLOCKING=1 USE_TRITEIA=1 python benchmarks/deltas/benchmark_7b.py
# echo "--------- Split weights ---------"
# TP_SIZE=2 USE_TRITEIA=1 python benchmarks/deltas/benchmark_7b.py