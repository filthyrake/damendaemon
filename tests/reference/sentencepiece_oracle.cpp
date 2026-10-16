#include "llama.h"
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
static void hex(const char *name,const std::string &value) {
    std::cout << name << ':';
    static const char *digits="0123456789abcdef";
    for(unsigned char c:value)std::cout << digits[c>>4] << digits[c&15];
    std::cout << '\n';
}
int main(int argc,char **argv) {
    if(argc<3)return 2;
    llama_log_set([](ggml_log_level,const char *,void *){},nullptr);
    llama_backend_init();
    auto params=llama_model_default_params();params.vocab_only=true;params.n_gpu_layers=0;
    auto model=llama_model_load_from_file(argv[1],params);if(!model)return 3;
    auto vocab=llama_model_get_vocab(model);
    std::string prompt((std::istreambuf_iterator<char>(std::cin)),{});
    std::vector<llama_token> ids(4096);
    int count=llama_tokenize(vocab,prompt.data(),prompt.size(),ids.data(),ids.size(),std::string(argv[2])=="1",true);
    if(count<0)return 4;ids.resize(count);
    std::cout << "ids:";for(size_t i=0;i<ids.size();i++){if(i)std::cout<<',';std::cout<<ids[i];}std::cout<<'\n';
    if(argc>3){ids.clear();std::istringstream values(argv[3]);std::string item;while(std::getline(values,item,','))ids.push_back(std::stoi(item));}
    for(bool skip:{true,false}) {
        std::vector<char> buffer(1024*1024);
        int size=llama_detokenize(vocab,ids.data(),ids.size(),buffer.data(),buffer.size(),skip,!skip);
        if(size<0)return 5;
        hex(skip?"decode_skip":"decode_all",std::string(buffer.data(),size));
        std::string generated;
        for(auto id:ids){size=llama_token_to_piece(vocab,id,buffer.data(),buffer.size(),0,!skip);if(size<0)return 6;generated.append(buffer.data(),size);}
        hex(skip?"generated_skip":"generated_all",generated);
    }
    llama_model_free(model);llama_backend_free();
}
