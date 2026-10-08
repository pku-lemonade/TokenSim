#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>
namespace cute { template<class T> using numeric_limits=std::numeric_limits<T>; using std::min; using std::max; }
namespace math { template<class T> T ceil_div(T a,T b) { return (a+b-1)/b; } }
// KV range one Q block must visit, cover all [ks, ke)
// precomputed by metadata kernel, consumed by (Q TMA, KV TMA, UMMA, math)
struct MQALogitsKVSpan {
    uint32_t kv_token_base; // min k_start over rows, rounded down to 4 tokens
    uint32_t num_kv_splits; // ceil((max k_end - kv_token_base) / SPLIT_KV)
};

// NOTE: Out-of-range rows are clamped to the last valid row
template <uint32_t BLOCK_Q, uint32_t SPLIT_KV>
 MQALogitsKVSpan get_mqa_logits_kv_span(const uint32_t& q_block_idx,
                                                      const uint32_t& num_q_tokens, const uint32_t& num_kv_tokens,
                                                      const uint32_t* cu_seq_len_k_start,
                                                      const uint32_t* cu_seq_len_k_end) {
    uint32_t start = cute::numeric_limits<uint32_t>::max();
    uint32_t end = cute::numeric_limits<uint32_t>::min();
    #pragma unroll 8
    for (uint32_t token_idx = 0; token_idx < BLOCK_Q; ++ token_idx) {
        const auto row_idx = cute::min(q_block_idx * BLOCK_Q + token_idx, num_q_tokens - 1);
        const auto k_start = cute::min(cu_seq_len_k_start[row_idx], num_kv_tokens);
        const auto k_end = cute::min(cu_seq_len_k_end[row_idx], num_kv_tokens);
        start = cute::min(start, k_start);
        end = cute::max(end, k_end);
    }
    const uint32_t kv_token_base = start / 4 * 4;
    return {kv_token_base, math::ceil_div(end - kv_token_base, SPLIT_KV)};
}
int main() {
 uint32_t b,q,p;
 while(std::cin>>b>>q>>p) {
  uint32_t len=(p+q)/4,n=b*q,nkv=b*len;
  std::vector<uint32_t> ks(n),ke(n);
  for(uint32_t i=0;i<n;i++) { ks[i]=(i/q)*len; ke[i]=ks[i]+(p+i%q+1)/4; }
  uint64_t splits=0;
  for(uint32_t block=0;block<(n+1)/2;block++)
   splits+=get_mqa_logits_kv_span<2,256>(block,n,nkv,ks.data(),ke.data()).num_kv_splits;
  std::cout<<splits*256*2<<"\n";
 }
}
