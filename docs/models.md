# Statistical models

Here, we explain the statistical models used.

The models are composed of two components:

  - The growth model used to describe how the abundances increase or decrease, basing on competition between different variants. This model links some interpretable parameters (such as the growth advantages) with idealised (unobserved) abundances, $p(t)$.
  - The noise model connecting idealised abundances $p(t)$ with deconvolved values $y(t)$, which are subject to some noise.

## Growth model

### Single-location model

We consider V competing variants, numbered between 1, ..., V present in the population. At time $t$ the relative abundances of the variants are represented by a vector $p (t) = (p_1(t), p_2(t), ..., p_V(t))$.
We assume survival-of-the fittest type of [selection dynamics](https://en.wikipedia.org/wiki/Replicator_equation#Equation), where the $i$-th variant has a fitness value $f_i$, which is fixed in time, and the relative abundances change at a rate proportional to their fitness advantage over the average value:

$$ \frac{\mathrm d}{\mathrm dt}p_i(t)= p_i(t)\left(f_i - \sum_{j=1}^V p_j(t) f_j \right).$$

In this model, the selection advantage of variant $X_i$ over variant $X_j$ is given by $s_{ij}=f_i-f_j$.
Note that the model dynamics is determined only by the relative selection advantages, rather than the fitness values $f_i$, making the problem non-identifiable: adding a constant to all fitness values does not change the dynamics of the model.
Hence, without loss of generality, we set $f_1=0$, to avoid this identifiability issue.

This set of ordinary differential equations has the analytical solution 

$$ p_i(t)=\frac{\exp(f_i\cdot t+b_i)}{ \sum_{j=1}^V \exp(f_j\cdot t + b_j)} $$

Where $b_1, b_2, ..., b_V$ are constants given by the initial conditions. Similarly as with fitness values, adding a constant to all $b_i$ does not change the functions $p_i(t)$. We fix $b_1 = 0$.

### Multiple-location model

The above model can be used to describe a collection of location-specific relative abundance vectors $p_k(t)$, where the index $k\in \{1,..., K\}$ represents a spatial location (e.g., the city or a district connected to one data collection system).

We expect that the introduction times of different variants to different locations may be different, which we accommodate by defining location-specific parameters $b_{vk}$. However, if the wastewater sampling locations are located at the same country subject to the same vaccine program and immunization, we suspect that the processes $p_k(t)$ are not entirely independent.
We therefore *make an assumption that the fitness value does not change across the locations*.

Note that while we find this assumption plausible in the analysis of the data from the same country, it may not hold when analysing locations subject to different vaccine programs.

To summarize, we infer parameters $b_{vk}$ for all variants $v\in \{1, ..., V\}$ and locations $k\in \{1,..., K\}$  together with fitness values $f_1, ..., f_V$, which are shared between the sampling locations.
We use the identifiability constraints $f_1 = 0$ and $b_{1k} = 0$ for all $k$.

## Noise model

We deconvolute a wastewater sample collected at time point $t$ and location $k$ to obtain the observed relative abundances vector

$y_k(t) =(y_{1k}(t), ..., y_{Vk}(t))$, where $y_{vk}(t)$ represents the relative abundance of variant $v\in \{1, ..., V\}$ as obtained in the deconvolution procedure.

Due to a small load of viral signal, amplification through next generation sequencing method, and deconvolution procedure, we do not have an explicit generative model linking the ideal abundance value $p_k(t)$, to the deconvolved value $y_k(t)$.
Instead, we use the quasi-likelihood approach, where we assume that $\mathbb E[y_k(t)] = p_k(t),$
and use a covariance function in a generalized linear model corresponding to the scaled multinomial distribution.
In this manner, the quasi-likelihood inference allows one to correct the obtained confidence intervals by using the dispersion parameter, which is adapted to capture the variance observed in the data.

As both $y_k(t)$ and $p_k(t)$ are probability vectors, we use the quasi-multinomial model, in which the quasi-loglikelihood function is given by

$$ q(f,b)= \sum_{k=1}^K \sum_{t=1}^T\sum_{v=1}^V y_{vk}(t) \log p_{vk}(t). $$ 

We numerically optimize it to find the maximum quasi-likelihood estimate $\hat \theta = (\hat f, \hat b)$.
