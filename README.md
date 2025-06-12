# kepler_globals
## Hierarchical Bayesian modeling of exoplanet demographics using the Kepler data!

---
Welcome to **Kepler Globals**! This project is an attempt to find the true underlying distribution of exoplanet radius, mass, and period using photodynamical TTV/lightcurve modeling and Hierarchical Bayesian statistics. 

### Data
This code takes photodynamically modeled posterior outputs from [PhoDyMM](https://github.com/dragozzine/PhoDyMM) ([Ragozzine et al. 2020](https://www.overleaf.com/project/5cd3a16b2b033e4cb4459a1b)) and uses a clustering algorithm (modified from [Hou et al. 2012](https://iopscience.iop.org/article/10.1088/0004-637X/745/2/198)) to reject walker chains trapped in local minima. The script then randomly subsamples the output chains and adds dozens of calculated parameters for convenient analysis. Finally, the script merges the table with the system parameters in [Table 1](https://iopscience.iop.org/article/10.3847/PSJ/ad0e6e#psjad0e6et1) of [Lissauer et al. 2024](https://iopscience.iop.org/article/10.3847/PSJ/ad0e6e) and the vetting efficiency results of [Hsu et al. 2019](https://iopscience.iop.org/article/10.3847/1538-3881/ab31ab). The resulting dataset is known as the **Kepler Multis Dynamical Catalog**, or KMDC. 

The KMDC contains $\gtrsim90$% of all Kepler multiplanet systems, each photodynamicallly modeled with complete planetary radius, mass, orbital element, and stellar posteriors. It supports analyses in exoplanetary architectures, 
interiors, demographics, and dynamics.

### Models
This repository also contains code which calculates a non-parameterized 3D occurence rate grid using [emcee](https://emcee.readthedocs.io/en/stable) ([Foreman-Mackey et al. 2013](https://arxiv.org/abs/1202.3665)) to model the true underlying exoplanet distribution. This model is akin to [Foreman-Mackey et al. 2014](https://iopscience.iop.org/article/10.1088/0004-637X/795/1/64), but extended into the mass dimension.

It also creates parametric models to model this data, using the methods found in [Neil & Rogers 2020](https://iopscience.iop.org/article/10.3847/1538-4357/ab6a92/meta). 


---
### Authors

The vast majority of the code was developed by Steven Blodgett, with direction and support throughout from Darin Ragozzine. Other advice and coding help was provided by Dallin Spencer and Daniel Jones. 

For more information, contact blodgett.steven.m@gmail.com or darin_ragozzine@byu.edu.
