<div align="center">
<h1>Thermal Neural Networks </h1>
<h3>Lumped-Parameter Thermal Modeling With State-Space Machine Learning </h3>

[Wilhelm Kirchgässner](https://github.com/wkirgsn), [Oliver Wallscheid](https://github.com/wallscheid), [Joachim Böcker](https://scholar.google.de/citations?user=vmyBqw0AAAAJ&hl=de&oi=ao)

[Paderborn University](https://www.uni-paderborn.de/en/), [Dept. of Power Electronics and Electrical Drives](https://ei.uni-paderborn.de/en/lea)

Paper: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0952197622005279),
Preprint: [arXiv 2103.16323](https://arxiv.org/abs/2103.16323)

</div>

## Abstract
With electric power systems becoming more compact with higher power density, the relevance of thermal stress and precise real-time-capable model-based thermal monitoring increases. Previous work on thermal modeling by lumped-parameter thermal networks (LPTNs) suffers from mandatory expert knowledge for their design and from uncertainty regarding the required power loss model. In contrast, deep learning-based temperature models cannot be designed with the low amount of model parameters as in a LPTN at equal estimation accuracy. In this work, the thermal neural network (TNN) is introduced, which unifies both, consolidated knowledge in the form of heat-transfer-based LPTNs, and data-driven nonlinear function approximation with supervised machine learning. The TNN approach overcomes the drawbacks of previous paradigms by having physically interpretable states through its state-space representation, is end-to-end differentiable through an automatic differentiation framework, and requires no material, geometry, nor expert knowledge for its design. Experiments on an electric motor data set show that a TNN achieves higher temperature estimation accuracies than previous white-/gray- or black-box models with a mean squared error of 3.18 K² and a worst-case error of 5.84 K at 64 model parameters.

## Purpose
This repository demonstrates the application of thermal neural networks (TNNs) on an electric motor data set.

The data set is freely available at [Kaggle](https://www.kaggle.com/wkirgsn/electric-motor-temperature).

The TNN declaration and its usage are demonstrated in the jupyter notebooks, with tensorflow in `TNN_tensorflow.ipynb` and with PyTorch in `TNN_pytorch.ipynb`.


## Topology

![](img/topology.png)

Three function approximators (e.g., multi-layer perceptrons (MLPs)) model the thermal parameters (i.e., thermal conductances, thermal capacitances, and power losses) of an arbitrarily complex component arrangement in a system.
Such a system is assumed to be sufficiently representable by a system of ordinary differential equations (not partial differential equations!).

One function approximator outputs thermal conductances, another the inverse thermal capacitances, and the last one the power losses generated within the components.
Although thermal parameters are to be estimated, their ground truth is not required.
Instead, measured component temperatures can be plugged into a cost function, where they are compared with the estimated temperatures that result from the thermal parameters that are estimated from the current system excitation.
[Error backprop through time](https://en.wikipedia.org/wiki/Backpropagation_through_time) will take over from here. 

The TNN's inner cell working is that of [lumped-parameter thermal networks](https://en.wikipedia.org/wiki/Lumped-element_model#Thermal_systems) (LPTNs).
A LPTN is an electrically equivalent circuit whose parameters can be interpreted to be thermal parameters of a system.
A TNN can be interpreted as a hyper network that is parameterizing a LPTN, which in turn is iteratively solved for the current temperature prediction.

In contrast to other neural network architectures, a TNN needs at least to know which input features are temperatures and which are not.
Target features are always temperatures.

In a nutshell, a TNN solves the difficult-to-grasp nonlinearity and scheduling-vector-dependency in [quasi-LPV](https://en.wikipedia.org/wiki/Linear_parameter-varying_control) systems, which a LPTN represents.


## Citing

The TNN is introduced in:
```
@article{kirchgaessner_tnn_2023,
title = {Thermal neural networks: Lumped-parameter thermal modeling with state-space machine learning},
journal = {Engineering Applications of Artificial Intelligence},
volume = {117},
pages = {105537},
year = {2023},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2022.105537},
url = {https://www.sciencedirect.com/science/article/pii/S0952197622005279},
author = {Wilhelm Kirchgässner and Oliver Wallscheid and Joachim Böcker}
}
```

Further, this repository supports and demonstrates the findings around a TNN's generalization to [Neural Ordinary DIfferential Equations](https://arxiv.org/abs/1806.07366) as presented on [IPEC2022](https://www.ipec2022.org/index.html). 
If you want to cite that work, please use
```
@INPROCEEDINGS{kirchgässner_node_ipec2022,
  author={Kirchgässner, Wilhelm and Wallscheid, Oliver and Böcker, Joachim},
  booktitle={2022 International Power Electronics Conference (IPEC-Himeji 2022- ECCE Asia)}, 
  title={Learning Thermal Properties and Temperature Models of Electric Motors with Neural Ordinary Differential Equations}, 
  year={2022},
  volume={},
  number={},
  pages={2746-2753},
  doi={10.23919/IPEC-Himeji2022-ECCE53331.2022.9807209}}
```

The data set is freely available at [Kaggle](https://www.kaggle.com/wkirgsn/electric-motor-temperature) and can be cited as
```
@misc{electric_motor_temp_kaggle,
  title={Electric Motor Temperature},
  url={https://www.kaggle.com/dsv/2161054},
  DOI={10.34740/KAGGLE/DSV/2161054},
  publisher={Kaggle}, 
  author={Wilhelm Kirchgässner and Oliver Wallscheid and Joachim Böcker}, 
  year={2021}}
```
