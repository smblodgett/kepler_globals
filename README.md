# kepler_globals
## Bayesian modeling of exoplanet demographics and population global parameters using the Kepler data!

---
 
This code takes posterior outputs from PhoDyMM ([Ragozzine et al. 2020](https://github.com/dragozzine/PhoDyMM)) and uses a clustering algorithm (citation) to choose converged walker chains. The script then thins the output chains, adding dozens of calculated parameters.
Then, it calculates non-parameterized 3D histogram weights using emcee (Foreman-Mackey et al. 2013) to model the true underlying exoplanet distribution. It also creates parametric models to model this data, using the methods found in Neil & Rogers 2020. 

---


---

Developed by Steven Blodgett and Darin Ragozzine, Brigham Young University. For more information, contact blodgett.steven.m@gmail.com or darin_ragozzine@byu.edu.
