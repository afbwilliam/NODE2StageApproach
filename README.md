# Nueral ODE 2-Stage Approach
The examples in this repository illustrate the Neural ODE (NODE) 2-Stage Approach to parameter estimation of Ordinary Differential Equations (ODEs) (see reference [1]). Examples also illustrate tips for hyperparameter tuning of Neural ODEs to better enable them to learn system dynamics from training data (see reference [2]).

## Requirements

Code was tested using the following Python modules:

* Python 3.10.11
* Pytorch 1.12.1

For examples using pyomo for parameter estimation, installation of pyomo w/ipopt is required.  This [website](https://ndcbe.github.io/CBE60499/01.00-Pyomo-Introduction.html) offers a good walkthrough for installing these.

## Background
Neural Ordinary Differential Equations (Neural ODEs or NODEs) are a deep learning data-driven model with the ability to learn the trajectories of highly nonlinear spatio-temporal dynamics (see reference [3]).  However, like other machine learning frameworks, their ability to extrapolate to conditions beyond the range of training data is limited.  ODEs based on first-principles or domain knowledge (i.e., mechanistic ODEs) have the potential to extrapolate beyond measured conditions, but they can be computationally onerous to train when directly fit to data.  Merging the benefits of both modeling approaches, the NODE 2-Stage (i.e., Indirect) Approach consists of 1) fitting a Neural ODE to time-series or spatial data and 2) using the Neural ODE’s derivative estimates to estimate the parameters of a mechanistic ODE.  This enables a modeler to exploit the rapid and flexible training of Neural ODEs to fit the parameters of a mechanistic ODE, resulting in a final model that better generalizes to conditions of interest.

![alt text](https://github.com/afbwilliam/NODE2StageApproach/blob/main/KineticRxn/visuals/2-stage-approach.png)

## The Advantage of Hybridizing First-Principles and Machine Learning
A simple example of using the Neural ODE 2-Stage Approach to fit ODE model parameters can be found in the [`KineticRxn/AG_Rxn+Deg.py`](./KineticRxn/AG_Rxn+Deg.py) file.

In this example, the Neural ODE easily fits the measured time-series data.  However, when the fitted Neural ODE later simulates the same system for conditions outside the range of training data, as shown in the below graphic, the predictions of the fitted Neural ODE diverge from the true dyanmics.  In contrast, if the Neural ODE derivative estimates are used via the 2-stage approach to train the parameters of a correctly formulated (i.e., first-principles or mechanistically-inspired) ODE model, this model can be more accurate, even when predicting the system dynamics for conditions outside the range of the original training data.

![alt text](https://github.com/afbwilliam/NODE2StageApproach/blob/main/KineticRxn/visuals/Extrap.png)
*Simulations of fitted Neural ODE (left) and fitted mechanistic ODE (right) on extrapolating conditions*

Additional examples of applying Neural ODEs to estimate the parameters of mechanistic ODEs can be found in the [`2StagePaperExamples`](./2StagePaperExamples) folder.  Examples of interpolating sparse, multi-experiment data via Neural ODEs can be found in the [`SparsityPaperExamples`](./SparsityPaperExamples) folder.

## Multiple Overlapping Integration Intervals – Effect on Over-fitting and Over-Smoothing Data
To optimally train a Neural ODE, Neural ODE hyperparameters must be tuned to avoid issues of both overfitting and over-smoothing.  A key hyperparameter for this is the length of the interval of integration used during training.  To help visualize how breaking integration into multiple intervals prevents over-smoothing, Neural ODE training on data from the Lotka-Volterra system is illustrated below using two different lengths of integration.  Clearly, simulating the Neural ODE across the longer interval of integration at every iteration of training ultimately converges to a model that “over-smooths” the dynamics.  In contrast, by breaking the Neural ODE predictions to cover shorter, overlapping intervals, the NODE captures the true dynamics and can avoid the local minimum wherein the model over-smooths the system response.

Longer Intervals           |  Shorter Intervals
:-------------------------:|:-------------------------:
![alt text](https://github.com/afbwilliam/NODE2StageApproach/blob/main/GIFs/LoVoIC.gif) | ![alt text](https://github.com/afbwilliam/NODE2StageApproach/blob/main/GIFs/LoVoICs.gif)

*Progression of Neural ODE training by simulating the NODE over a single time interval (left) vs simulating the NODE from multiple initial conditions (right).  NODE predictions in green.  Dots represent measured points and dashed red and blue lines the true dynamics.*

Nevertheless, intervals of integration should not be so short as to encourage Neural ODE overfitting, especially problematic when data is noisy.  Shown below is the process of fitting a Neural ODE to noisy data for different lengths of integration.  Although the Neural ODE captures the dynamics for both cases during training, only the Neural ODE trained with longer intervals is able to simulate (as shown with black lines) the long-range dynamics after training.

Longer  Intervals          |  Shorter Intervals
:-------------------------:|:-------------------------:
![alt text](https://github.com/afbwilliam/NODE2StageApproach/blob/main/GIFs/8stepsFHN.gif) | ![alt text](https://github.com/afbwilliam/NODE2StageApproach/blob/main/GIFs/2stepsFHN.gif)

*Progression of Neural ODE training by simulating the NODE over 9 datapoints (left) or 2 datapoints (right) during training.  NODE predictions in green.  Dots represent measured points and dashed red and blue lines the true dynamics.  Simulation of trained NODE over full 10 datapoints represented by black lines.*

A useful heuristic to follow to minimize Neural ODE overfitting is to choose the longest interval of integration during Neural ODE training that does not cause the fitted Neural ODE to over-smooth the data.  

## References
Further reading on the advantages of Neural ODEs when applying the 2-stage approach to ODE parameter estimation can be found in the papers:

1) Bradley, W. and F. Boukouvala, *Two-Stage Approach to Parameter Estimation of Differential Equations Using Neural ODEs.* Industrial & Engineering Chemistry Research, 2021. [[paper]](https://pubs.acs.org/doi/10.1021/acs.iecr.1c00552)

2) Bradley, W., Volkovinsky, R., & Boukouvala, F. (2024). *Enabling global interpolation, derivative estimation and model identification from sparse multi-experiment time series data via neural ODEs.* Engineering Applications of Artificial Intelligence, 130, 107611. [[paper]](https://doi.org/https://doi.org/10.1016/j.engappai.2023.107611)

Code in this repository is modified from earlier work described in the following reference:

3) Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud. *Neural Ordinary Differential Equations.* Advances in Neural Information Processing Systems, 2018. [[arxiv]](https://arxiv.org/abs/1806.07366)

