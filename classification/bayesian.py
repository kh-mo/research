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

class ScaleMixturePrior(nn.Module):
    def __init__(self, pi, sigma1, sigma2):
        super(ScaleMixturePrior, self).__init__()
        self.pi = pi
        self.sigma1 = torch.FloatTensor([math.exp(sigma1)]).to(device)
        self.sigma2 = torch.FloatTensor([math.exp(sigma2)]).to(device)
        self.gaussian1 = torch.distributions.normal.Normal(0, self.sigma1)
        self.gaussian2 = torch.distributions.normal.Normal(0, self.sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()

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

        self.log_variational_posterior = 0
        self.log_prior = 0

        self.pi = 0.5
        self.sigma1 = 0
        self.sigma2 = -6
        self.weight_prior = ScaleMixturePrior(self.pi, self.sigma1, self.sigma2)
        self.bias_prior = ScaleMixturePrior(self.pi, self.sigma1, self.sigma2)

    def forward(self, input):
        weight = self.weight.sample()
        bias = self.bias.sample()
        self.log_variational_prior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)

        return f.linear(input, weight, bias)

class bayesian(nn.Module):
    def __init__(self, num_classes=10):
        super(bayesian, self).__init__()
        self.linear1 = BayesianLinear(784, 10)

    def forward(self, x):
        x = torch.flatten(x, 1) #1
        x = self.linear1(x) #2
        return x

    def log_variational_posterior(self):
        return self.linear1.log_variational_posterior

    def log_prior(self):
        return self.linear1.log_prior

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = torch.randn(2,1,28,28)
    target = torch.tensor([0,1])
    model = bayesian().to(device)
    output = model(input.to(device))

    negative_log_likehood = f.nll_loss(f.log_softmax(output, dim=1), target.to(device)).to(device)
    log_variational_posterior = model.log_variational_posterior()
    log_prior = model.log_prior()
    loss = log_variational_posterior - log_prior + negative_log_likehood

