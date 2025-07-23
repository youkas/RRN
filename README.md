# 📦 Reparametrization and Regression Network (RRN)

**RRN** is an unsupervised learning framework for **simultaneous dimensionality reduction and surrogate modeling**, specifically designed for high-dimensional engineering problems with weak or no input correlations.

🔗 **Read the original paper**:  
[Unsupervised re-parametrization for simultaneous dimension reduction and surrogate modeling: Application to aerodynamic shape optimization](https://www.sciencedirect.com/science/article/pii/S0957417425022432)  

Published in *Expert Systems with Applications* (2025).

---

## 🔍 Overview

RRN integrates two jointly trained components: a Reparametrization model, which extracts a low-dimensional latent representation of the input space, and a Regression model, which maps this representation to the target output. Unlike traditional approaches that rely on correlations among input variables, RRN effectively handles weakly correlated or uncorrelated inputs, making it particularly well-suited for high-dimensional engineering problems. When benchmarked against a baseline combining an AutoEncoder and an ANN-based surrogate model, RRN demonstrates superior efficiency and robustness. By leveraging input–output relationships rather than depending on internal input correlations, RRN offers a new paradigm in combined dimension reduction and surrogate modeling—especially valuable in design scenarios where data is generated using space-filling techniques such as Latin Hypercube Sampling (LHS).

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 📁 Repository Structure

data/                 # Datasets or dataset generators  
model/                # Resulting Regression model   
rrn/  
└── rrn.py            # Core RRN implementation and loss function  
example.py            # A usage test case  
README.md             # Project documentation (this file)  

---

## 📚 Reference

To cite this repository:  

[![DOI](https://zenodo.org/badge/681712595.svg)](https://doi.org/10.5281/zenodo.16362756)

  
If you use RRN in your research or development, please cite the following paper:

> **Youness Karafi**, **Zakaria Moussaoui**, and **Badr Abou El Majd**  
> *Unsupervised re-parametrization for simultaneous dimension reduction and surrogate modeling: Application to aerodynamic shape optimization*  
> *Expert Systems with Applications*, Volume 292, 2025, Article 128624  
> [https://doi.org/10.1016/j.eswa.2025.128624](https://doi.org/10.1016/j.eswa.2025.128624)

```bibtex
@article{KARAFI2025128624,
  title = {Unsupervised re-parametrization for simultaneous dimension reduction and surrogate modeling: Application to aerodynamic shape optimization},
  journal = {Expert Systems with Applications},
  volume = {292},
  pages = {128624},
  year = {2025},
  issn = {0957-4174},
  doi = {https://doi.org/10.1016/j.eswa.2025.128624},
  url = {https://www.sciencedirect.com/science/article/pii/S0957417425022432},
  author = {Youness Karafi and Zakaria Moussaoui and Badr {Abou El Majd}},
  keywords = {Surrogate models, Dimension reduction, Shape optimisation, Aerodynamic, Uncertainty modeling, Artificial neural networks},
  abstract = {In aerodynamic shape optimization, dimension reduction and surrogate modeling are widely recognized for their potential to reduce the computational cost and time associated with computational fluid dynamics simulations and, subsequently, the numerical optimization process. This article introduces a novel unsupervised learning framework called the Reparameterization and Regression Network (RRN). The RRN is designed to simultaneously perform variable transformation for selective dimension reduction and surrogate modeling while constructing a regression model. The model comprises two interdependent sub-models: the reparameterization model, which transforms selected features, and the regression model, which predicts outputs based on the transformed features. Both components are trained jointly using a dataset of input variables and their corresponding outputs. The proposed approach excels in reducing the dimensionality of weakly correlated data while improving the predictive accuracy of surrogate models. Furthermore, it has proven effective and reliable for applications in deterministic and robust optimization. This capability is demonstrated through the aerodynamic optimization of a transonic wing parameterized using the free-form deformation technique.}
}


