import os
os.chdir(os.getcwd()+"/bayesian")
import numpy as np
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize

count_data = np.loadtxt("datasets/txtdata.csv")
n_count_data = len(count_data)
figsize(12.5, 3.5)
plt.bar(np.arange(n_count_data), count_data, color="#348ABD")
plt.xlabel("시간(일수)", fontsize=13)
plt.ylabel("수신한 문자 메시지 개수", fontsize=13)
plt.title("사용자의 메시지 습관이 시간에 따라 변하는가?")
plt.xlim(0, n_count_data)




import pymc3 as pm

alpha = 1.0 / count_data.mean()
lambda_1 = pm.Exponential.dist(alpha, "lambda_1") # lambda_1 = pm.Exponential("lambda_1", alpha)
lambda_2 = pm.Exponential.dist(alpha, "lambda_2") # lambda_2 = pm.Exponential("lambda_2", alpha)
tau = pm.DiscreteUniform.dist(lower=0, upper=n_count_data) # tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data)
print("Random output:", tau.random(), tau.random(), tau.random())

@pm.Deterministic
def lambda_(tau=tau, lambda_1=lambda_1, lambda_2=lambda_2):
    out = np.zeros(n_count_data)
    out[:tau] = lambda_1
    out[tau:] = lambda_2
    return out

out = np.zeros(n_count_data)
threshold = tau.random()
out[:threshold] = lambda_1.random()
out[threshold:] = lambda_2.random()

observation = pm.Poisson.dist(mu=out) # observation = pm.Poisson("obs", lambda_, value=count_data, observed=True)
model = pm.Model([observation, lambda_1, lambda_2, tau])

mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000)

