# echo "--------- TP=1 ---------"
# TP_SIZE=1 python benchmarks/deltas/benchmark_7b.py
# echo "--------- Not-split weights TP=2 ---------"
# TP_SIZE=2 UNOPTIMIZED_DELTA=1 python benchmarks/deltas/benchmark_7b.py
# echo "--------- Split weights ---------"
# TP_SIZE=2 python benchmarks/deltas/benchmark_7b.py

# echo "--------- TP=1 (bitblas) ---------"
# TP_SIZE=1 USE_BITBLAS=1 python benchmarks/deltas/benchmark_7b.py
# echo "--------- TP=2 (bitblas) ---------"
# TP_SIZE=2 USE_BITBLAS=1 python benchmarks/deltas/benchmark_7b.py

# echo "--------- Split weights ---------"
# TP_SIZE=2 USE_TRITEIA=1 python benchmarks/deltas/benchmark_7b.py

# echo "--------- TP=1 (triton) ---------"
# TP_SIZE=1 python benchmarks/deltas/benchmark_7b.py

echo "--------- TP=2 (bitblas, 2bits) ---------"
BITWIDTH=2 TP_SIZE=2 USE_BITBLAS=1 python benchmarks/deltas/benchmark_7b.py