// Validation-only adapter to the pinned llama.cpp public vocabulary API.
// Link against an unchanged reference build; this is never used by dllm runtime.
#include "llama.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    llama_backend_init();
    auto params = llama_model_default_params();
    params.vocab_only = true;
    auto * model = llama_model_load_from_file(argv[1], params);
    if (!model) return 3;
    const auto * vocab = llama_model_get_vocab(model);
    std::string line;
    while (std::getline(std::cin, line)) {
        const auto request = nlohmann::json::parse(line);
        const auto text = request.at("text").get<std::string>();
        const bool special = request.at("add_special_tokens").get<bool>();
        std::vector<llama_token> ids(text.size() + 16);
        int count = llama_tokenize(vocab, text.data(), text.size(), ids.data(), ids.size(), special, true);
        if (count < 0) {
            ids.resize(-count);
            count = llama_tokenize(vocab, text.data(), text.size(), ids.data(), ids.size(), special, true);
        }
        if (count < 0) return 4;
        ids.resize(count);
        nlohmann::json response{{"ids", ids}};
        for (bool skip : {false, true}) {
            std::vector<char> output(text.size() * 4 + 128);
            int bytes = llama_detokenize(vocab, ids.data(), ids.size(), output.data(), output.size(), skip, !skip);
            if (bytes < 0) {
                output.resize(-bytes);
                bytes = llama_detokenize(vocab, ids.data(), ids.size(), output.data(), output.size(), skip, !skip);
            }
            if (bytes < 0) return 5;
            response[skip ? "decoded_skip_special" : "decoded"] = std::string(output.data(), bytes);
        }
        std::cout << response.dump() << '\n';
    }
    llama_model_free(model);
    llama_backend_free();
}
