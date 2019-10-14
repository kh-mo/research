import torch
import torch.nn as nn
import torch.nn.functional as f

class Gaussian(object):
    def __init__(self, mu, rho):
        super(Gaussian, self).__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.normal.Normal(loc=0, scale=1)

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(device)
        sigma = torch.log(1 + torch.exp(self.rho))
        return self.mu + sigma * epsilon

class BayesianLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BayesianLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weight_mu = nn.Parameter(torch.Tensor(self.out_dim, self.in_dim).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(self.out_dim, self.in_dim).uniform_(-0.2, 0.2))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        self.bias_mu = nn.Parameter(torch.Tensor(self.out_dim).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(self.out_dim).uniform_(-0.2, 0.2))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

    def forward(self, input):
        print(input) #3
        print(input.shape) #4
        weight = self.weight.sample()
        bias = self.bias.sample()
        print(weight.shape)
        print(bias.shape)
        return f.linear(input, weight, bias)

class bayesian(nn.Module):
    def __init__(self, num_classes=10):
        super(bayesian, self).__init__()
        self.classifier = nn.Sequential(
            BayesianLinear(784, 10),
        )
    def forward(self, x):
        x = torch.flatten(x, 1) #1
        x = self.classifier(x) #2
        return x

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = torch.randn(2,1,28,28)
    target = torch.tensor([0,1])
    model = bayesian().to(device)
    output = model(input.to(device))

    negative_log_likehood = f.nll_loss(output, target.to(device)).to(device)
    log_prior = 0
    log_variational_posterior = 0
    loss = log_variational_posterior - log_prior + negative_log_likehood
# output.mean(0)
#
# output
# tmp.mean(0)
# tmp
# (5.2919 + 12.7009)/2
# # loss 구하기
#
# weight_mu = nn.Parameter(torch.Tensor(10, 784).uniform_(-0.2, 0.2))
# f.linear(torch.flatten(input, 1), weight_mu)
#
#
# torch.log(torch.tensor([-10.0459]))
