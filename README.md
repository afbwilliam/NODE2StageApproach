# NODE 2-Stage Approach
 Examples to illustrate NODE 2-Stage Approach to parameter estimation of Ordinary Differential Equations (ODEs) (see reference [1]). Examples also illustrate tips for successful estimation of system dynamics from training data.  

## Requirements

Code was tested using the following Python modules:

* Python 3.10.11
* Pytorch 1.12.1

# The Advantage of Hybridizing First-Principles and Machine Learning
A simple example of how a Neural ODE can fit time-series data can be found in the [`KineticRxn/AG_Rxn+Deg.py`](./KineticRxn/AG_Rxn+Deg.py.py) file.

In this example, the Neural ODE readily fits the measured state data.  However, when the Neural ODE then simulates the same system for conditions outside the range of training data, as shown in the below graphic, the predictions of the fitted Neural ODE diverge from the true dyanmics.  In contrast, if the Neural ODE derivative estimates are used to train the parameters of a correctly formulated (i.e., first-principles or mechanistically-inspired) ODE model, this model can be more accurate, even when predicting the dynamics of the underlying system for conditions outside the range of the original training data.

![alt text](https://github.com/afbwilliam/NODE2StageApproach/blob/main/KineticRxn/visuals/Extrap.png)

# Multiple Overlapping Integration Intervals – Effect on Over-fitting and Over-Smoothing Data
To optimally train a Neural ODE, Neural ODE hyperparameters must be tuned to avoid issues of both overfitting and “over-smoothing”.  A key hyperparameter for this is the length of the interval of integration used during training.  To help visualize how breaking integration into multiple intervals prevents over-smoothing, Neural ODE training on data from the Lotka-Volterra system is illustrated below using two different lengths of integration.  Clearly, simulating the Neural ODE across the longer interval of integration at every iteration of training ultimately converges to a model that “over-smooths” the dynamics.  In contrast, by breaking the Neural ODE predictions to cover shorter, overlapping intervals, the NODE captures the true dynamics and can avoid the local minimum wherein the model over-smooths the system response.

Longer Intervals           |  Shorter Intervals
:-------------------------:|:-------------------------:
![alt text](https://github.com/afbwilliam/NODE2StageApproach/blob/main/GIFs/LoVoIC.gif) | ![alt text](https://github.com/afbwilliam/NODE2StageApproach/blob/main/GIFs/LoVoICs.gif)

*Progression of Neural ODE training by simulating the NODE over a single time interval (left) vs simulating the NODE from multiple initial conditions (right).  NODE predictions in green.  Dots represent measured points and dashed red and blue lines the true dynamics.*

Nevertheless, intervals of integration should not be so short as to encourage Neural ODE overfitting, especially problematic when data is noisy.  Shown below is the process of fitting a Neural ODE to noisy data for different lengths of integration.  Although the Neural ODE captures the dynamics when trained for both cases, only the Neural ODE trained with longer intervals is able to simulate (as shown with black lines) the long-range dynamics after training.

Longer  Intervals          |  Shorter Intervals
:-------------------------:|:-------------------------:
![alt text](https://github.com/afbwilliam/NODE2StageApproach/blob/main/GIFs/8stepsFHN.gif) | ![alt text](https://github.com/afbwilliam/NODE2StageApproach/blob/main/GIFs/2stepsFHN.gif)

*Progression of Neural ODE training by simulating the NODE over 2 datapoints (left) or 9 datapoints (right) during training.  NODE predictions in green.  Dots represent measured points and dashed red and blue lines the true dynamics.  Simulation of trained NODE over full 10 datapoints represented by black lines.*

Ultimately, a useful heuristic to follow to minimize Neural ODE overfitting is to choose the longest interval of integration during Neural ODE training that does cause the fitted Neural ODE to over-smooth the data.  

# References
Further reading on the advantages of Neural ODEs when applying the 2-stage approach to ODE parameter estimation can be found in the paper:

Bradley, W. and F. Boukouvala, *Two-Stage Approach to Parameter Estimation of Differential Equations Using Neural ODEs.* Industrial & Engineering Chemistry Research, 2021. [[paper]](https://pubs.acs.org/doi/10.1021/acs.iecr.1c00552)
