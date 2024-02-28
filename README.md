<div align="center">
<h1>Thermal Neural Networks </h1>
<h3>Lumped-Parameter Thermal Modeling With State-Space Machine Learning </h3>

[Wilhelm Kirchgässner](https://github.com/wkirgsn), [Oliver Wallscheid](https://github.com/wallscheid), [Joachim Böcker](https://scholar.google.de/citations?user=vmyBqw0AAAAJ&hl=de&oi=ao)

[Paderborn University](https://www.uni-paderborn.de/en/), [Dept. of Power Electronics and Electrical Drives](https://ei.uni-paderborn.de/en/lea)

Paper: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0952197622005279)
Preprint: [arXiv 2103.16323](https://arxiv.org/abs/2103.16323)

</div>


# Thermal Neural Networks
This repository demonstrates the usage of thermal neural networks (TNNs) on an electric motor data set.



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

# Citation

The TNN is introduced in:
```
@misc{kirchgässner2023_tnn,
  title={Thermal Neural Networks: Lumped-Parameter Thermal Modeling With State-Space Machine Learning}, 
  author={Wilhelm Kirchgässner and Oliver Wallscheid and Joachim Böcker},
  year={2021},
  eprint={2103.16323},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

Further, this repository supports and demonstrates the findings around a TNN's generalization to [Neural Ordinary DIfferential Equations](https://arxiv.org/abs/1806.07366) as presented on [IPEC2022](https://www.ipec2022.org/index.html). (citation t.b.a.)