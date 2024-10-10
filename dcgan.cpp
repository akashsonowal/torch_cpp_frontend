#include <torch.h>

struct Net : torch::nn::Module {
    Net(int64_t N, int64_t M){
        W = register_buffer("W", torch::randn({N , M}));
        B = register_buffer("B", torch::randn(M));
    }

    torch::Tensor forward(torch::Tensor input){
        return torch::addmm(b, input, W);
    }
    torch::Tensor W, b;
};

struct Net : torch::nn::Module {
    Net(int64_t N, int64_t M) : linear(register_module("linear", torch::nn::Linear(N, M))) {
        another_bias = register_parameter("b", torch::randn(M));
    }

    torch::Tensor forward(torch::Tensor input){
        return linear(input) + another_bias;
    }
    torch::nn:Linear linear;
    torch::Tensor another_bias;
};