import math

import torch
import torch.nn as nn
import torch.nn.functional as f

class Gaussian(object):
    def __init__(self, mu, rho):
        super(Gaussian, self).__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.normal.Normal(loc=0, scale=1)
        self.sigma = 0

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(device)
        self.sigma = torch.log(1 + torch.exp(self.rho))
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - (((input - self.mu)/self.sigma)**2)/2).sum()

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
        weight = self.weight.sample()
        bias = self.bias.sample()
        return f.linear(input, weight, bias)

class bayesian(nn.Module):
    def __init__(self, num_classes=10):
        super(bayesian, self).__init__()
        self.linear1 = BayesianLinear(784, 10),

    def forward(self, x):
        x = torch.flatten(x, 1) #1
        x = self.linear1(x) #2
        return x

    def log_variational_posterior(self, input):
        return self.linear1.weight.log_prior(input) + self.linear1.bias.log_prior(input)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = torch.randn(2,1,28,28)
    target = torch.tensor([0,1])
    model = bayesian().to(device)
    output = model(input.to(device))

    negative_log_likehood = f.nll_loss(f.log_softmax(output, dim=1), target.to(device)).to(device)
    log_variational_posterior = model.log_variational_posterior()
    log_prior = 0
    loss = log_variational_posterior - log_prior + negative_log_likehood
