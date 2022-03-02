# Thermal Neural Networks
This repository demonstrates the usage of thermal neural networks (TNNs) on an electric motor data set.

The TNN is introduced in:
```
@misc{kirchgässner2021thermal,
  title={Thermal Neural Networks: Lumped-Parameter Thermal Modeling With State-Space Machine Learning}, 
  author={Wilhelm Kirchgässner and Oliver Wallscheid and Joachim Böcker},
  year={2021},
  eprint={2103.16323},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

The data set is freely available at [Kaggle](https://www.kaggle.com/wkirgsn/electric-motor-temperature).

## Topology

![](img/topology.png)

Three function approximators (e.g., multi-layer perceptrons (MLPs)) model the thermal characteristics of an arbitrarily complex component arrangement forming a system of interest.
One outputs thermal conductances, another the inverse thermal capacitances, and the last one the power losses generated within the components.

In contrast to other neural network architectures, a TNN needs at least to know which features are temperatures and which are not.
The TNN's inner cell working is that of [lumped-parameter thermal networks](https://en.wikipedia.org/wiki/Lumped-element_model#Thermal_systems) (LPTNs).
A TNN can be interpreted as a hyper network that is parameterizing an LPTN, which in turn is iteratively solved for the current temperature prediction.

In a nutshell, a TNN solves the difficult-to-grasp nonlinearity and scheduling-vector-dependency in [quasi-LPV](https://en.wikipedia.org/wiki/Linear_parameter-varying_control) systems, which an LPTN represents.

## Code Structure

The streamlined usage can be seen in the jupyter notebooks, one with tensorflow2 and one with pytorch.
The tf2-version makes heavy use of auxiliary functions and classes declared in `aux`, whereas the pytorch notebook is self-contained and does not import from 'aux'. 
The TNN keras model can be found in [aux/lptn_model](aux/lptn_model.py) as `TNNCell`.

In both frameworks, the TNN is defined as a cell class that is plugged into an outer RNN layer.
