// Standalone benchmark adapter using llama.cpp's public API. No upstream edits.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc,char** argv) {
    if(argc!=3) {std::cerr<<"usage: bench_t5_llama model.gguf whitespace-token-file\n";return 2;}
    std::ifstream input(argv[2]);std::vector<llama_token> prompt;
    for(llama_token token;input>>token;)prompt.push_back(token);
    if(prompt.empty() || prompt.size()>512)return 2;
    llama_backend_init();
    auto mp=llama_model_default_params();mp.n_gpu_layers=99;
    auto* model=llama_model_load_from_file(argv[1],mp);
    if(!model)return 1;
    auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=512;cp.n_ubatch=512;cp.n_seq_max=1;
    cp.type_k=GGML_TYPE_F32;cp.type_v=GGML_TYPE_F32;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto* ctx=llama_init_from_model(model,cp);
    if(!ctx)return 1;
    const auto* vocab=llama_model_get_vocab(model);
    if(!llama_model_has_encoder(model))return 2;
    std::cout<<"{\"engine\":\"llama.cpp public API\",\"context\":512,\"trials\":[";
    for(int trial=0;trial<6;++trial) {
        llama_memory_clear(llama_get_memory(ctx),true);
        auto start=std::chrono::steady_clock::now();
        auto batch=llama_batch_get_one(prompt.data(),prompt.size());
        if(llama_encode(ctx,batch))return 1;
        llama_token token=llama_model_decoder_start_token(model);
        std::vector<llama_token> output;double first=0;
        for(int step=0;step<8;++step) {
            batch=llama_batch_get_one(&token,1);
            if(llama_decode(ctx,batch))return 1;
            const auto* logits=llama_get_logits_ith(ctx,-1);
            token=std::max_element(logits,logits+llama_vocab_n_tokens(vocab))-logits;
            output.push_back(token);
            if(step==0)first=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            if(llama_vocab_is_eog(vocab,token))break;
        }
        auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        if(trial)std::cout<<",";
        std::cout<<"{\"warmup\":"<<(trial==0?"true":"false")<<",\"first_token_seconds\":"<<first<<",\"total_seconds\":"<<elapsed<<",\"tokens\":[";
        for(size_t i=0;i<output.size();++i) {if(i)std::cout<<",";std::cout<<output[i];}
        std::cout<<"]}";
    }
    std::cout<<"]}\n";
    llama_free(ctx);llama_model_free(model);llama_backend_free();
}
