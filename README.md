# NODE 2-Stage Approach
 Examples to illustrate NODE 2-Stage Approach to parameter estimation of Ordinary Differential Equations (ODEs) (see reference [1]). Examples also illustrate tips for successful estimation of system dynamics from training data.  

## Requirements

Code was tested using the following Python modules:

* Python 3.10.11
* Pytorch 1.12.1

# The Advantage of Hybridizing First-Principles and Machine Learning
A simple example of how a Neural ODE can fit time-series data can be found in the [`KineticRxn/AG_Rxn+Deg.py`](./KineticRxn/AG_Rxn+Deg.py.py) file.
In this example, the Neural ODE readily fits the measured state data, but as shown in the below graphic, when simulation conditions greatly differ from training data, the Neural ODE prediction diverge from the true dyanmics.  In contrast, if the Neural ODE derivative estimates are used to train the parameters of a correctly formulated (i.e., first-principles or mechanistically-inspired) ODE model, this model can accurately predict the dynamics of the underlying system for conditions far beyond the range of the original training data.


# References
Further reading on the advantages of Neural ODEs when applying the 2-stage approach to ODE parameter estimation can be found in the paper:

Bradley, W. and F. Boukouvala, *Two-Stage Approach to Parameter Estimation of Differential Equations Using Neural ODEs.* Industrial & Engineering Chemistry Research, 2021. [[paper]](https://pubs.acs.org/doi/10.1021/acs.iecr.1c00552)
