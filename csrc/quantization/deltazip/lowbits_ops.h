#pragma once

#include <torch/extension.h>

void pack_rows_4
(
    torch::Tensor input,
    torch::Tensor output
);
void pack_columns
(
    torch::Tensor input,
    torch::Tensor output,
    int bits
);
void quantize_err
(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scale,
    float qzero,
    float maxq,
    float err_norm,
    float min_p,
    float max_p,
    int p_grid
);
void quantize
(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scale,
    torch::Tensor out_q,
    float qzero,
    float maxq
);
uintptr_t make_q_matrix
(
    torch::Tensor q_weight,
    torch::Tensor q_perm,
    torch::Tensor q_invperm,
    torch::Tensor q_scale,
    torch::Tensor q_scale_max,
    torch::Tensor q_groups,
    torch::Tensor q_group_map,
    torch::Tensor gptq_qzeros,
    torch::Tensor gptq_scales,
    torch::Tensor gptq_g_idx,
    torch::Tensor temp_dq
);
void free_q_matrix
(
    uintptr_t handle
);
void reconstruct
(
    uintptr_t q_handle,
    torch::Tensor output
);
void gemm_half_q_half
(
    torch::Tensor a,
    uintptr_t b,
    torch::Tensor c,
    bool force_cuda
);
uintptr_t make_q_attn
(
    torch::Tensor layernorm,
    float norm_epsilon,
    uintptr_t q_q_proj,
    uintptr_t q_k_proj,
    uintptr_t q_v_proj,
    uintptr_t q_o_proj,
    torch::Tensor temp_state,
//    torch::Tensor temp_q,
//    torch::Tensor temp_k,
//    torch::Tensor temp_v,
    torch::Tensor temp_dq,
    int max_rows,
    int hidden_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len
);
void free_q_attn
(
    uintptr_t handle
);
void q_attn_forward_1
(
    uintptr_t q_attn,
    torch::Tensor x,
    int batch_size,
    int q_len,
    int past_len,
    torch::Tensor past_lens,
    torch::Tensor q_temp,
    torch::Tensor k_temp,
    torch::Tensor v_temp,
    torch::Tensor sin,
    torch::Tensor cos,
    const std::vector<uintptr_t>& loras,
    torch::Tensor loras_temp
);
void q_attn_forward_2
(
    uintptr_t q_attn,
    torch::Tensor x,
    torch::Tensor attn_output,
    int batch_size,
    int q_len,
    const std::vector<uintptr_t>& loras,
    torch::Tensor loras_temp
);
int q_attn_set_loras
(
    uintptr_t q_attn,
    std::unordered_map<uintptr_t, torch::Tensor>& q_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& q_proj_lora_b,
    std::unordered_map<uintptr_t, torch::Tensor>& k_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& k_proj_lora_b,
    std::unordered_map<uintptr_t, torch::Tensor>& v_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& v_proj_lora_b,
    std::unordered_map<uintptr_t, torch::Tensor>& o_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& o_proj_lora_b
);
uintptr_t make_q_mlp
(
    torch::Tensor layernorm,
    float norm_epsilon,
    uintptr_t q_gate,
    uintptr_t q_up,
    uintptr_t q_down,
    torch::Tensor temp_state,
    torch::Tensor temp_a,
    torch::Tensor temp_b,
    torch::Tensor temp_dq,
    int max_rows
);
void free_q_mlp
(
   uintptr_t handle
);
void q_mlp_forward_
(
    uintptr_t q_mlp,
    torch::Tensor x,
    const std::vector<uintptr_t>& loras,
    torch::Tensor loras_temp
);
int q_mlp_set_loras
(
    uintptr_t q_mlp,
    std::unordered_map<uintptr_t, torch::Tensor>& gate_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& gate_proj_lora_b,
    std::unordered_map<uintptr_t, torch::Tensor>& up_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& up_proj_lora_b,
    std::unordered_map<uintptr_t, torch::Tensor>& down_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& down_proj_lora_b
);
uintptr_t make_q_moe_mlp
(
    torch::Tensor layernorm,
    float norm_epsilon,
    torch::Tensor gate,
    int num_experts,
    int num_experts_per_token,
    const std::vector<uintptr_t>& w1,
    const std::vector<uintptr_t>& w2,
    const std::vector<uintptr_t>& w3,
    torch::Tensor temp_state,
    torch::Tensor temp_gathered_state,
    torch::Tensor temp_a,
    torch::Tensor temp_b,
    torch::Tensor temp_logits,
    torch::Tensor temp_dq,
    int max_rows
);
void free_q_moe_mlp
(
   uintptr_t handle
);
void q_moe_mlp_forward_
(
    uintptr_t q_moe_mlp,
    torch::Tensor x
//    const std::vector<uintptr_t>& loras,
//    torch::Tensor loras_temp
);
void rope_
(
    torch::Tensor x,
    torch::Tensor sin,
    torch::Tensor cos,
    int past_len,
    int num_heads,
    int head_dim,
    torch::Tensor offsets
);
void lowbits_rms_norm
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor y,
    float epsilon
);
void lowbits_rms_norm_
(
    torch::Tensor x,
    torch::Tensor w,
    float epsilon
);
void apply_rep_penalty
(
    torch::Tensor sequence,
    float penalty_max,
    int sustain,
    int decay,
    torch::Tensor logits
);
std::vector<float> sample_basic
(
    torch::Tensor logits,           // shape [bsz, vocab_size]
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    float tfs,
    float typical,
    float random,
    torch::Tensor output_tokens,    // shape [bsz, 1]
    torch::Tensor output_probs,     // shape [bsz, 1]
    torch::Tensor logit_filter,     // shape [bsz, vocab_size]
    bool mirostat,
    std::vector<float>& mirostat_mu,
    float mirostat_tau,
    float mirostat_eta,
    float post_temperature
);
void logit_filter_exclusive
(
    torch::Tensor filter,                                       // shape [bsz, vocab_size]
    const std::vector<std::vector<int>> &exclusive_lists
);
void fp16_to_fp8(torch::Tensor in_tensor, torch::Tensor out_tensor, int batch_size, int offset, int width);
void fp8_to_fp16(torch::Tensor in_tensor, torch::Tensor out_tensor, int batch_size, int offset, int width);
void gemm_half_half_half
(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    const float alpha,
    const float beta,
    bool force_cublas
);
