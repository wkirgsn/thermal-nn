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

## Topology

![](img/topology.png)

Three function approximators (e.g., multi-layer perceptrons (MLPs)) model the thermal characteristics of an arbitraryly complex component arrangement forming a system of interest.
One outputs thermal conductances, another the inverse thermal capacitances, and the last one the power losses generated within the components.

In contrast to other neural network architectures, a TNN needs at least to know which features are temperatures and which are not.
The TNN's inner cell working is that of [lumped-parameter thermal models](https://en.wikipedia.org/wiki/Lumped-element_model#Thermal_systems) (LPTNs).

In a nutshell, a TNN solves the difficult-to-grasp nonlinearity and scheduling-vector-dependency in [quasi-LPV](https://en.wikipedia.org/wiki/Linear_parameter-varying_control) systems, which an LPTN represents.

## Code Structure

The streamlined usage can be seen in the [jupyter notebook](ThermalNeuralNetworks.ipynb).
It makes heavy use of auxiliary functions and classes declared in `aux`. 
The TNN keras model can be found in [aux/lptn_model](aux/lptn_model.py) as `TNNCell`.

