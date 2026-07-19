---
title: 'ForMoSA: Forward Modeling tool for Spectral Analysis'
tags:
  - python
  - astronomy
  - exoplanets
  - forward modeling
  - atmosphere
authors:
  - name: ForMoSA Collaboration
    equal-contrib: true
    affiliation: "1,2,3,4,5,6,7,8,9,10"
  - name: Simon Petrus
    orcid: 0000-0003-0331-3654
    equal-contrib: true
    affiliation: "1, 2, 3, 4" # (Multiple affiliations must be quoted)
  - name: Paulina Palma-Bifani
    orcid: 0000-0002-6217-6867
    equal-contrib: true
    affiliation: "5, 6"
  - name: Matthieu Ravet
    orcid: 0009-0000-4898-4713
    equal-contrib: true
    affiliation: "5, 4, 7"
  - name: Allan Denis
    equal-contrib: true
    affiliation: "8"
  - name: Bhavesh Rajpoot
    orcid: 0009-0004-9729-6377
    equal-contrib: true
    affiliation: "7, 9" 
  - name: Mickaël Bonnefoy
    orcid: 0000-0001-5579-5339
    equal-contrib: false
    affiliation: "4" 
  - name: Gaël Chauvin
    orcid: 0000-0003-4022-8598
    equal-contrib: false
    affiliation: "7" 
  - name: Arthur Vigan
    orcid: 0000-0002-5902-7828
    equal-contrib: false
    affiliation: "8" 
  - name: Alice Radcliffe
    orcid: 0009-0003-9345-019X
    equal-contrib: false
    affiliation: "6"  
  - name: Pablo Requeijo
    orcid: 0009-0007-9285-5952
    equal-contrib: false
    affiliation: "6"
  - name: Kevin Hoy
    orcid: 0009-0004-5870-9562
    equal-contrib: false
    affiliation: "2, 3, 10" 
affiliations:
 - name: NASA-Goddard Space Flight Center, Greenbelt, MD 20771, USA
   index: 1
 - name: Instituto de Estudios Astrofísicos, Facultad de Ingeniería y Ciencias, Uni. Diego Portales, Av. Ejército 441, Santiago, Chile
   index: 2
 - name: Millennium Nucleus on Young Exoplanets and their Moons
   index: 3
 - name: Univ. Grenoble Alpes, CNRS, IPAG, F-38000 Grenoble, France
   index: 4
 - name: Laboratoire J. L. Lagrange, Université Côte d’Azur, Observatoire de la Côte d’Azur, CNRS, 06304 Nice, France
   index: 5
 - name: LIRA, Observatoire de Paris, Université PSL, Sorbonne Université, Université de Paris, 5 place Jules Janssen, 92195 Meudon, France
   index: 6
 - name: Max-Planck-Institut für Astronomie, Königstuhl 17, 69117 Heidelberg, Germany
   index: 7
 - name: Aix Marseille Univ, CNRS, CNES, LAM, Marseille, France
   index: 8
 - name: Department of Physics and Astronomy, Heidelberg University, Im Neuenheimer Feld 226, D-69120 Heidelberg, Germany
   index: 9
 - name: European Southern Observatory, Alonso de Cordova 3107, Vitacura, Santiago, Chile 
   index: 10
date: 19 August 2025
bibliography: paper_clean.bib



# Optional fields if submitting to a AAS journal too, see this blog post: 
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

**`ForMoSA` (FORward MOdeling tool for Spectral Analysis)** is an open-source `Python` package to fit spectroscopic and photometric observations using a Bayesian framework. It can utilize different self-consistent atmospheric models to perform robust parameter space exploration. It has been mainly designed for fitting directly imaged young planetary-mass brown dwarfs and exoplanets.
The developments within **`ForMoSA`** are supported by an international collaboration of several laboratories in France (IPAG, LIRA, LAM, and Lagrange), Germany (MPIA), USA (NASA Goddard), and Chile (FCLA, Universidad Diego Portales, and Universidad de Chile). The evolution of the code and the growing interest from the scientific community has led to the need for this dedicated publicatio alongside the release of **`ForMoSA` v2.0**, which has been refactored into a class architecture with user-friendly features and new functionalities.


# Statement of need

Recent advances in ground- and space-based observatories now enable routine, high-quality observations of exoplanet atmospheres across a wide range of wavelengths and resolutions. 
In practice, **`ForMoSA`** was designed to bridge the gap between these observations and atmospheric models by providing a Bayesian framework to robustly compare the two. **`ForMoSA`** adopts a forward-modeling approach based on exploring through a nested sampling [@skilling_nested_2004] the parameter space of pre-computed, self-consistent atmospheric model grids.  

The code **`ForMoSA`** addresses the following needs:

- **Data modularity:** It allows the simultaneous analysis of diverse datasets, even when they overlap in wavelength coverage. This includes multiple datasets taken at different times and with different setups.

- **Computational efficiency:** It leverages `xarray`[^1], enabling fast interpolation and manipulation of large model grids.

- **Flexible likelihood mappings:** Users can choose from various likelihood metrics to account for specific observational uncertainties [@ruffio2019].

- **Extensive parameterization:** Beyond the atmospheric model parameters, a comprehensive suite of extra-grid parameters, such as radial velocity, rotational broadening, and scaling factors, can be incorporated into the fit.

- **High-Contrast module:** It incorporates the method of [@landman2024], allowing users to include reference stellar, atmospheric transmission, and systematics spectra to accurately fit contrast-limited data.


[^1]: `xarray` documentation: <https://docs.xarray.dev/en/stable>


# State of the field

The landscape of exoplanetary atmospheric modeling is broadly divided into two categories: "free retrieval" frameworks, which compute radiative transfer on the fly to fit parameterized atmospheric structures, and "forward modeling" codes, which interpolate pre-computed grids of self-consistent physical models.


The codes in the first category are **`CHIMERA`** [@Line2013], **`petitRADTRANS`** [@Molliere19], **`NEMESIS`** [@Irwin2008], **`CROCODILE`** [@Hayoz23], and **`Pyrat Bay`** [@Cubillos21]. These tools are very flexible but also computationaly expensive. 
Other specialized tools can handle specific niche problems, such as opacity management and radiative transfer calculations [**`Exo_k`**, @Leconte2021], high-resolution autodifferentiation [**`ExoJAX`**, @Kawahara22], 1D disequilibrium chemistry [**`HOMER`**, @Himes22], or models for rocky planet emission spectra [**`rfast`**, @Robinson23]. 


**`ForMoSA`** complement these on-the-fly radiative transfer or free-retrieval codes by offering a faster, grid-interpolated alternative for parameter estimation.
Our code’s closest existing counterparts are **`species`** [@Stolker20] and **`SEDA`** [@Suarez2025], which also facilitate Bayesian inference between spectro-photometric observations and pre-computed atmospheric grids. **`species`** is a generalized, all-in-one toolkit for direct imaging that encompasses everything from flux calibration to color-magnitude diagrams. **`SEDA`** is more similar to **`ForMoSA`**, but it was only introduced early 2025 to the community, and has not yet been widely adopted. 
We chose to build **`ForMoSA`** as a standalone package to provide a highly specialized, modular inversion framework expressly optimized for data heterogeneity and statistical flexibility. Integrating these new capabilities into existing tools would have required fundamental rewriting, leading to a new code. 



# Software design

![**`ForMoSA`** workflow diagram. The shaded gray area on the right represents the core functionalities required to run the Nested Sampling. The left dark-gray area represents utility and support functions. The small boxes contain the modules, and the dashed boxes represent the contents of each folder.](schema_ForMoSA.png)


The Python-based code of **`ForMoSA`** is organized into several modules, as shown in Figure 1. In practice, **`ForMoSA`** v2.0 has a class-based architecture, with some main modules (config, observation, filter, grid, parameter, nested_sampling), and support modules (core, utils, and transform).
Because **`ForMoSA`** is an end-to-end Bayesian tool that handles all stages of the analysis, its installation requires several `Python` packages, as well as downloading the model grids needed for the fit. All information required for installation, access to available grids, and the tutorials material is available at the **`ForMoSA`** documentation[^2].

In summary, to start an analysis, the user must first provide an atmospheric model grid, the observations, and a `.ini` configuration file. This `.ini` file contains all of **`ForMoSA`**'s input parameters, including the paths, grid adaptation keys, nested sampling keys, and the free parameters priors configuration. 
**`ForMoSA`**'s main modules are centralized through a main Analysis class, which creates and handles sub-grids of atmospheric models tailored to each observation, before launching the nested sampling algorithm.
At each iteration of the nested sampling, random parameter values are drawn from the prior distributions. The sampled parameters are used to interpolate model spectra from the sub-grids, before comparing them to each observation by computing the log-likelihood function. For a given run, the data, sub-grids, parameters, and results are automatically saved to paths specified by the user, and can be easily recovered for subsequent analysis and visualized through the Plotting class.


[^2]: **`ForMoSA`** documentation: <https://ForMoSA.readthedocs.io/en/latest/index.html>



# Research impact statement 


Initiated in 2020, **`ForMoSA`** has become a highly adaptable tool for the atmospheric characterization of substellar objects. Its computational efficiency and compatibility with diverse physical models have enabled the standard analysis of targets across a wide range of spectral resolutions, from broadband photometry to high-resolution spectroscopy. 

A major strength of **`ForMoSA`** lies in its capacity to jointly fit multi-instrument, heterogeneous datasets to maximize wavelength coverage. Because combining data from diverse instruments often introduces biases, **`ForMoSA`** features the dedicated `MOSAIC` module. This framework utilizes customized likelihood scaling and parametric covariance models to mitigate inter-calibration offsets and complex noise structures, which can be tuned for each specific target and dataset. 
Altogether, these features allow for the accurate characterization and publication of atmospheric properties of various benchmark companions such as HIP 65426 [@Petrus21; @Carter23], AB Pic b [@PalmaBifani2023], VHS 1256 b [@Petrus23; @Petrus24, @Radcliffe26], AF Lep b [@Palma24], $\beta$ Pic b [@Houlle2025; @Ravet25], YSES 1 b and c [@Hoch2025], and COCONUTS-2 b [@Ravet26]. 
Furthermore, **`ForMoSA`**'s speed allows for the homogeneous analysis of large spectral libraries, such as those of brown dwarfs and planetary-mass companions [@Petrus2025; @PalmaBifani2025], resulting in the identification of parameter trends and structural systematics in atmospheric models.
Finally, **`ForMoSA`** also includes a high-contrast, high-resolution (HCHR) module applied to contrast-limited, stellar-contaminated data (e.g. VLT/HiRISE) to extract precise radial velocities and detailed orbital constraints [@Denis25; @Denis26].



**`ForMoSA`** is an active project, and its scientific production is growing rapidly, therefore a list of peer-reviewed publications is maintained on our NASA ADS public library[^3].
Our code will continue to evolve over time, with new functionalities being added to meet the requirements of data from future observatories (ELTs, HWO, etc.), updates to the state-of-the-art atmospheric models, and the parametrization of additional physics to better fit the data (e.g., disks, extinction laws, multiple columns, time variability).
For this reason, the official documentation is regularly updated at `ReadTheDocs`, and should be considered the reference description of the code.



[^3]: NASA ADS Library: <https://ui.adsabs.harvard.edu/user/libraries/PekELjOGR4yl3XOnGwOAng>







# AI usage disclosure

Our team fully designed and authored the core architecture, logic, and implementation of the code without AI generation. We used AI strictly as an auxiliary tool for debugging and managing documentation. Additionally, we utilized Gemini to refine the English and Gemini coupled with Antigravity to format and compile the main text and the bibliography.




# Acknowledgements

The authors express their sincere thanks to the Code/Astro Workshop[^4], which provided the foundational training necessary to transform **`ForMoSA`** into a professional, open-source `Python` package.
We gratefully acknowledge the funding and support for the ForM-X workshops held in Nice (2023), Heidelberg (2024/2025), and Grenoble (2025). These collaborative sessions were instrumental in the development and refinement of the code. We also thank the various laboratories and institutions, especially IPAG, Lagrange, and MPIA, for their continued support.
Furthermore, this work has been supported by the French National Research Agency (ANR) through the MIRAGES project (PI: A. Vigan, ANR-20-CE31-0017).


[^4]: Code/Astro Workshop: <https://semaphorep.github.io/codeastro/>


# Appendix A: Performance

The computational cost of **`ForMoSA`** is primarily driven by the number of forward model evaluations required for the nested sampling algorithm to converge. While absolute execution time (inversion time) depends on the specific hardware, Figure 2 provides a comparative analysis to guide users in defining the configuration file. The total inversion time is a multi-dimensional function depending on several factors, including the spectral resolution ($R_{\lambda}$), the signal-to-noise ratio (S/N), the number of live points, the dimensionality of the parameter space, and the machine used.


![Performance comparison between the nested sampling algorithms `PyMultiNest` (squares) and `Nestle` (crosses). Different colors show the number of free parameters used (from 1 to 5). From left to right: Inversion time as a function of spectral resolution ($R_{\lambda}$), signal-to-noise (S/N), and number of live points. The default setup is $R_{\lambda}$ = 368, S/N = 22, and 215 live points.](inversion_time_formosa_comp.pdf)



In the left panel of Figure 2, we observe that the main driver of computational cost is indeed the spectral resolution. As the spectral resolution increases, the number of data points also increases, and the number of points to be compared in the likelihood function grows, directly increasing the computational cost of each forward model call.
In the middle panel, we observe the dependence on the S/N. Here, noise is applied using the following formula:

$$
\vec{n} \sim \mathcal{N}\left(0, \sigma^2\right), \quad \text{with} \quad \sigma = \frac{\operatorname{mean}(d)}{\mathrm{S/N}} ,
$$

where $d$ is the synthetic flux. As the S/N increases, the error bars become smaller, which leads to a more sharply peaked likelihood function. This tighter posterior distribution requires the sampler to explore the parameter space with finer precision and typically increases the number of likelihood evaluations, thereby lengthening the total inversion time.
In the right panel, we observe that the inversion time scales approximately linearly with the number of live points.
The different colors in Figure 2 indicate how the dimensionality of the parameter space (number of free parameters) strongly drives the inversion time. As the complexity of the model increases, the "volume" of the prior space grows, requiring more iterations to isolate the high-likelihood regions.
In addition, we also provide a comparison between `PyMultiNest` (squares) and `nestle` (crosses). While `nestle` is a convenient pure-`Python` implementation, `PyMultiNest` (leveraging the Fortran `MultiNest` library) generally demonstrates superior scaling and efficiency.

To optimize the integration time of a **`ForMoSA`** inversion, we advise users to:

- Start low and run initial tests with a reduced number of live points (<100) to verify the model setup.  
- Use a spectral resolution that matches the physical information content of the data; over-sampling the spectrum increases the inversion time without improving accuracy.  
- Select the right algorithm for the inversion. For high-dimensional fits (>5 parameters), `PyMultiNest` is the recommended default.



# References
