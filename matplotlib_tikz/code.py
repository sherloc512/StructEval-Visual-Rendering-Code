from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt
def poisson_pmf(k, lambda_val):
    '''
    Calculates and plots the Poisson probability mass function (PMF
    :param k: Number of events to occur at the same time
    :param lambda_val: Lambda value. rate at which the events occur
    :return:
    - Print out the probability that k events occur at the same time with a rate lambda value
    - Plot the PMF from 0 to k occurrences
    '''
    x = np.arange(0, step=0.1, stop=k + 1)
    pmf = poisson.pmf(k=x, mu=lambda_val)
    print("Poisson:: Probability of having 10 customers at the shop")
    print(np.round(poisson.pmf(k=k, mu=lambda_val), 3))
    # plotting the PMF
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    # removing all borders except bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.bar(x, pmf * 100, color='#0F72AC')
    plt.xlabel('Number of customers at the shop', fontsize=12, labelpad=20)
    plt.ylabel('P(X=k) | Probability of k occurrences', fontsize=12, labelpad=20)
    plt.show()
poisson_pmf(k=10, lambda_val=5)
