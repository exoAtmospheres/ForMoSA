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
    orcid: 
    equal-contrib: true
    affiliation: 
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
  - name: Arthur Vigan
    orcid: 0000-0002-5902-7828
    equal-contrib: false
    affiliation: "8" 
  - name: Mickaël Bonnefoy
    orcid: 0000-0001-5579-5339
    equal-contrib: false
    affiliation: "4" 
  - name: Gaël Chauvin
    orcid: 0000-0003-4022-8598
    equal-contrib: false
    affiliation: "7" 
  - name: Alice Radcliffe
    orcid: 0009-0003-9345-019X
    equal-contrib: false
    affiliation: "6"  
  - name: Kevin Hoy
    orcid: 0009-0004-5870-9562
    equal-contrib: false
    affiliation: "2, 3, 10" 
  - name: Pablo Requeijo
    orcid: 0009-0007-9285-5952
    equal-contrib: false
    affiliation: "6"
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
date: 04 June 2026
bibliography: paper.bib



# Optional fields if submitting to a AAS journal too, see this blog post: 
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

**`ForMoSA` (FORward MOdeling tool for Spectral Analysis)** is an open-source `Python` package designed to extract physical parameters from spectroscopic and photometric observations based on a Bayesian framework. It has been mainly designed for young planetary-mass objects and directly imaged exoplanets, and it can utilize different self-consistent atmospheric models to perform robust parameter space exploration.

Over the past years, **`ForMoSA`** has evolved from a specialized research script into a mature, community-driven tool. The developments within **`ForMoSA`** are supported by an international collaboration among several laboratories in France (IPAG, LIRA, LAM, and Lagrange), Germany (MPIA), USA (NASA Goddard), and Chile (FCLA, Universidad Diego Portales, and Universidad de Chile). This evolution has led to the need for this dedicated publication where we formally present the tool to the community.

The paper is organized as follows: the `Statement of need` section provides a statement of need; the `Software design` section describes the specific functionalities of the code; the `Installation and example` section points towards the online documentation; and the `Performance` section evaluates the performance of the code. Finally, the Conclusions section concludes with prospects on future directions for **`ForMoSA`**.

To complement the information in the main body, several appendices are provided: one introduces other community standard tools; another discusses the expected accuracy of the framework; and a third summarizes the peer-reviewed work utilizing **`ForMoSA`**.


# Statement of need

Recent advances in ground- and space-based observatories now enable routine, high-quality observations of exoplanet atmospheres across a wide range of wavelengths and resolutions. The code **`ForMoSA`** was developed to address the growing need for an efficient, standardized framework to analyze these high-contrast, high-resolution observations coming from various instruments. This includes the need to analyze multiple datasets taken at different times and with different setups within a single generalized framework.

In practice, **`ForMoSA`** was designed to bridge the gap between these observations and atmospheric models by providing a Bayesian framework to robustly compare the two. While several atmospheric inference codes exist, most rely on a data-driven retrieval approach where spectra are generated on the fly using highly parameterized models [@madhusudhan2018]. In contrast, **`ForMoSA`** adopts a forward-modeling approach based on pre-computed, self-consistent atmospheric model grids.  
This approach enables a direct model-driven comparison between observations and physically motivated theoretical predictions. Examples of other similar codes are given in Appendix A. In addition, by providing both a terminal-based interface and interactive Jupyter Notebook demos, **`ForMoSA`** stands out by the following key advantages.


- **Computational efficiency:** It leverages `xarray`[^1] for optimized multi-dimensional array management, enabling fast interpolation and manipulation of large model grids.

- **Data modularity:** The framework allows for the simultaneous import and analysis of diverse datasets, including spectroscopic and photometric observations, even when they overlap in wavelength coverage.

- **Flexible likelihood mappings:** Users can choose from various likelihood metrics, including different noise and scaling prescriptions to account for specific observational uncertainties [@ruffio2019].

- **Extensive parameterization:** Beyond the standard model grid, a comprehensive suite of extra-grid parameters, such as radial velocity, rotational broadening, and scaling factors, can be incorporated into the fit.

- **High-Contrast module:** It incorporates the method of [@landman2024], allowing users to include reference stellar, atmospheric transmission, and systematics spectra to accurately fit contrast-limited data.

- **Active development:** It is regularly updated to incorporate the latest algorithmic advancements and new model versions.

[^1]: `xarray` documentation: https://docs.xarray.dev/en/stable

# Software design

The Python-based code of **`ForMoSA`** is organized into several modules, as depicted in Figure 1 and Tables 1 and 2. In practice, **`ForMoSA`** has a class-based architecture, with some main modules (config, observation, filter, grid, parameter, nested_sampling; see the right side of the workflow figure), and support modules (core, utils, and transform; see the left side of the workflow figure).

At the beginning, the user must input a grid of self-consistent, precomputed atmospheric models, along with a set of observations and a configuration file containing all of **`ForMoSA`**'s input parameters, including those for the nested sampling algorithms, the log-likelihood function, and the nested sampling parameters associated with their priors. Then, **`ForMoSA`**'s main modules are centralized through the analysis class, which creates and handles sub-grids of atmospheric models tailored to each observation, before launching the nested sampling algorithm using the support modules.

Later, at each iteration of the nested sampling, random parameter values are drawn from the prior distribution specified by the user for each parameter. The transformations related to these parameters are applied to the sub-grids, before comparing them to each observation by computing the log-likelihood function. For a given run, the data, sub-grids, parameters, and results are automatically saved to paths specified by the user, and can be easily recovered by the modules of **`ForMoSA`** for subsequent analysis.   The results can also be visualized through the Plotting class.


![**`ForMoSA`** workflow diagram. The shaded gray area on the right represents the core functionalities required to run the Nested Sampling. The left dark-gray area represents utility and support functions. The boxes contain the classes of modules of **`ForMoSA`**. The main methods of each class are depicted as subtext below each box. The larger dashed boxes represent the contents of each folder, while the smaller dashed boxes represent sub-modules. The larger boxes represent the main modules.](schema_ForMoSA.png)


**Table 1:** Main modules of **`ForMoSA`**

| Module | Description |
|-----------|---------|
| **config** | |
| ConfigGenerator (`global_config.py`) | Class handling the generation and saving of a default configuration file. |
| ConfigLoader (`global_config.py`) | Class handling the loading of parameters from a configuration file. |
| Paths (`paths.py`) | Class handling the paths used in the configuration file. |
|---------------------------------------------------------------------------------------------------------|
| **observation** | |
| Observation (`observation_base.py`) | Base class representing an observation (spectroscopic or photometric). |
| SpectralObservation (`observation_spectroscopy.py`) | Class handling a spectroscopic observation. |
| PhotometryObservation (`observation_photometry.py`) | Class handling a photometric observation. |
| ObservationSet (`observation_set.py`) | Container for a set of observations. |
| ObservationLoader (`observation_set.py`) | Generate observation objects from various input types. |
|---------------------------------------------------------------------------------------------------------|
| **filter** | |
| PhotometryFilter (`filter.py`) | Class defining a photometric filter. |
|---------------------------------------------------------------------------------------------------------|
| **grid** | |
| ModelGrid (`model_grid.py`) | Class representing a native (non-adapted) grid. |
| SubGrid (`subgrid_base.py`) | Base class representing a subgrid (spectroscopic or photometric). |
| SubGridSpectroscopy (`subgrid_spectroscopy.py`) | Class handling a spectroscopic subgrid. |
| SubGridPhotometry (`subgrid_photometry.py`) | Class handling a photometric subgrid. |
| SubGridSet (`subgrid_set.py`) | Container for a set of subgrids. |
| GridLoader (`grid_loader.py`) | Generate grid from various input types. |
|---------------------------------------------------------------------------------------------------------|
| **parameter** | |
| Parameter (`parameter.py`) | Class handling a single parameter. |
| ParameterSet (`parameter_set.py`) | Container for a set of parameters. |
| Prior (`prior.py`) | Base class representing a prior associated to a parameter. |
|---------------------------------------------------------------------------------------------------------|
| **nested_sampling** | |
| NestedSampling (`nested_sampling.py`) | Class handling the Nested Sampling. |
| NSResults (`results.py`) | Class handling the results of the Nested Sampling. |
| NSAnalysis (`ns_analysis.py`) | Class for post-analysis of the Nested Sampling results, including reconstruction of the model from the best fit and the confidence intervals. |
| Plottings (`plotting.py`) | Class handling the visualization of the results of the Nested Sampling. |



**Table 2:** Support modules of **`ForMoSA`**

| Module | Description |
|--------|-------------|
| **transform** | |
| ApplyPhysicsEffects (`apply_effects.py`) | Apply physics-based transformations to a model. This includes, for example, Doppler shift, rotational broadening, radius scaling. |
| ApplyObservationEffects (`apply_effect.py`) | Apply observation-based transformations to a model. This includes resolution decreasing, analytic scaling, or high-contrast modeling. |
| ObservedParameters (`observed.py`) | Class defining a set of parameters and their associated values drawn from the Nested Sampling at a given iteration. |
| ObservedModel (`observed.py`) | Class defining the model drawn from the Nested Sampling at a given iteration. |
| PhotometricEffects (`photometric_effects.py`) | Class defining the physics-based and observation-based transformations associated to photometry. |
| SpectroscopicEffects (`spectroscopic_effects.py`) | Class defining the physics-based and observation-based transformations associated to spectroscopy. |
|---------------------------------------------------------------------------------------------------------|
| **core** | |
| `config.py` | General configuration setup and examples for plotting and saving. |
| `enums.py` | Enumerations of various keys used in **`ForMoSA`**. |
| `errors.py` | Exception class. |
| `loggings.py` | Logging module. |
|---------------------------------------------------------------------------------------------------------|
| **utils** | |
| `logL_functions.py` | Definition of loglikelihood functions used in **`ForMoSA`**. |
| `misc.py` | Miscellaneous utilities. |
| `prior_functions.py` | Definition of priors used in **`ForMoSA`**. |
| `spec.py` | Support functions for the transformation of spectrophotometric data. |




# Installation and example

Because **`ForMoSA`** is an end-to-end Bayesian tool that handles all stages of the analysis, such as adapting model grids to the data format, performing the Bayesian inversion, and generating the results, its installation requires several `Python` packages, as well as downloading the model grids needed for the fit. All information required for installation, access to available grids, and the demonstration material is available at the **`ForMoSA`** documentation[^2].

Some of the most commonly used model grids are available in a **`ForMoSA`**-compatible format. If a specific model required for an analysis is not yet available, the corresponding grid can be generated and reformatted by the developer upon request. Finally, to facilitate onboarding and training of new users, demonstration notebooks and tutorials are provided.

[^2]: **`ForMoSA`** documentation: https://ForMoSA.readthedocs.io/en/latest/index.html



# Performance

The computational cost of **`ForMoSA`** is primarily driven by the number of forward model evaluations required for the nested sampling algorithm to converge. While absolute execution time (inversion time) depends on the specific hardware, this section provides a comparative analysis to guide users in defining the configuration file.  

As astrophysicists, even though computing-time optimization is useful and important, we are in practice primarily driven by the retrieval accuracy. We complement this analysis by evaluating the accuracy for a specific test case in Appendix B.


![Performance comparison between the nested sampling algorithms `PyMultiNest` (squares) and `Nestle` (crosses). Different colors show the number of free parameters used (from 1 to 5). From left to right: Inversion time as a function of spectral resolution (R$_{\lambda}$), signal-to-noise (S/N), and number of live points. The default setup is R$_{\lambda}$ = 368, S/N = 22, and 215 live points. This is intended to inform the user of the order of magnitude in time they should expect for their fit to converge.](inversion_time_formosa_comp.pdf)


The total inversion time is a multi-dimensional function depending on several factors, including the spectral resolution (R$_{\lambda}$), the signal-to-noise ratio (S/N), the number of live points, the dimensionality of the parameter space, and the machine used. Figure 2 illustrates how these various parameters scale the inversion time using synthetic observations.

In the left panel, we observe that the main driver of computational cost is indeed the spectral resolution. As the spectral resolution increases, the number of data points also increases, and the number of points to be compared in the likelihood function grows, directly increasing the computational cost of each forward model call.

In the middle panel, we observe the dependence on the S/N. Here, noise is applied using the following formula:

$$
\vec{n} \sim \mathcal{N}\left(0, \sigma^2\right), \quad \text{with} \quad \sigma = \frac{\operatorname{mean}(d)}{\mathrm{S/N}} ,
$$

where $d$ is the synthetic flux. As the S/N increases, the error bars become smaller, which leads to a more sharply peaked likelihood function. This tighter posterior distribution requires the sampler to explore the parameter space with finer precision and typically increases the number of likelihood evaluations needed for convergence, thereby lengthening the total inversion time.

In the right panel, we observe that the inversion time scales approximately linearly with the number of live points. It is therefore important to consider that while more points improve the sampling of the posterior, they significantly increase the time required to reach the termination criterion.

The different colors in the performance figure indicate how the dimensionality of the parameter space (number of free parameters) strongly drives the inversion time. As the complexity of the model increases, the "volume" of the prior space grows, requiring more iterations to isolate the high-likelihood regions.

In addition, we also provide a comparison between `PyMultiNest` and `nestle`. While `nestle` is a convenient pure-`Python` implementation, `PyMultiNest` (leveraging the Fortran `MultiNest` library) generally demonstrates superior scaling and efficiency, especially as the number of free parameters increases beyond three.

To optimize the integration time of a **`ForMoSA`** inversion, we advise users to:


- Start low and run initial tests with a reduced number of live points (<100) to verify the model setup.  
- Use a spectral resolution that matches the physical information content of the data; over-sampling the spectrum increases the inversion time without improving accuracy.  
- Select the right algorithm for the inversion. For high-dimensional fits (>5 parameters), `PyMultiNest` is the recommended default.



# Conclusions

**`ForMoSA`** is an active project that has been used in multiple published projects of research (see Appendix C). It will continue to evolve over time, with new functionalities being added to meet the requirements of future instruments (ELTs, HWO, etc.), updates to the included model grids, and the coupling of atmospheric models with additional physics to better fit the data (e.g., disks, extinction laws, multiple dimensions, time variability). Therefore, the information contained in this paper may change in the future. For this reason, the official documentation at ReadTheDocs, which will be regularly updated, should be considered the reference description of the code.
As **`ForMoSA`** aims to provide the community with a tool that meets both current and future needs, the development team strongly encourages all users to provide feedback and suggestions via the `GitHub` issues tool in order to improve the code.


# Authors contributions

Simon Petrus, Paulina Palma-Bifani, Matthieu Ravet, Allan Denis, and Bhavesh Rajpoot contributed equally to this work. The author order reflects the chronological order in which they joined the project. These authors developed the core functionalities, wrote the documentation and tutorials, and utilized **`ForMoSA`** to address diverse scientific questions regarding a variety of exoplanets and brown dwarfs.

Mickaël Bonnefoy and Gaël Chauvin have served as the main coordinators and supervisors of the team since the project's start, guiding the first four authors who were doctoral students during the development of **`ForMoSA`**.

Arthur Vigan joined the project alongside Allan Denis in the context of HiRISE observations and the development of the High-Contrast High-Resolution (HCHR) module. Kevin Hoy joined the project in the context of his HiRISE observing program and has served as a primary tester and debugger for the HCHR module.

Alice Radcliffe contributed by computing a new version of the `Exo-REM` model grid and testing its performance using **`ForMoSA`**, which also required the implementation of new features within the code.

# AI usage disclosure

We did not rely on AI for the development of the code itself. All core architecture, logic, and implementation were designed and written by our team. AI was used solely as a supporting tool for documentation management and as an auxiliary resource during debugging.

# Acknowledgements

The authors express their sincere thanks to the Code/Astro Workshop[^3], which provided the foundational training necessary to transform **`ForMoSA`** into a professional, open-source `Python` package.

[^3]: Code/Astro Workshop: https://semaphorep.github.io/codeastro/

We gratefully acknowledge the funding and support for the ForM-X workshops held in Nice (2023), Heidelberg (2024/2025), and Grenoble (2025). These collaborative sessions were instrumental in the development and refinement of the code. We also thank the various laboratories and institutions, especially IPAG, Lagrange, and MPIA, for their continued support.

Furthermore, this work has been supported by the French National Research Agency (ANR) through the MIRAGES project (PI: A. Vigan, ANR-20-CE31-0017).

We acknowledge support in France from the French National Research Agency (ANR) through project grant ANR-20-CE31-0012.

S. Petrus was supported by an appointment to the NASA Postdoctoral Program at the NASA-Goddard Space Flight Center, administered by Oak Ridge Associated Universities under contract with NASA.

# Appendix A: State of the field on developments of atmospheric modeling tools

The field of exoplanetary atmospheric modeling has seen a significant increase in the availability of tools, largely divided between retrieval frameworks, which parameterize the atmospheric structures to fit the data, and forward modeling codes, which utilize self-consistent parametrized grids. While not intended as an exhaustive review of every available resource, Table 3 summarizes several of the most classical tools currently employed by the community. By situating **`ForMoSA`** within this broader ecosystem, we highlight its specific role as a modular, grid-based framework optimized for the next generation of substellar atmospheric analysis.


**Table 3:** Comparison of publicly available atmospheric modeling codes

| Name & Reference | Description |
|--------|-----------------|
| **`CHIMERA`** 
@Line2013 | `CHIMERA` (CaltecH Inverse ModEling and Retrieval Algorithms) is an open-source `Python` 3 package designed for the atmospheric retrieval and forward modeling of exoplanets. It interprets transmission and emission spectra from various instruments to constrain atmospheric properties. The code employs a chemically consistent framework or a free retrieval approach to estimate temperature profiles, gas abundances, and cloud properties. It utilizes high-performance acceleration via numba, supports multiple Bayesian samplers (e.g., `PyMultiNest`, `Dynesty`), and handles radiative transfer using correlated-K opacity treatments and two-stream approximations for scattering. |
|---------------------------------------------------------------------------------------------------------|
| **`CROCODILE`** 
@Hayoz23 | `CROCODILE` (Cross-correlation Retrievals of Directly Imaged self-Luminous Exoplanets) is a `Python`-based statistical framework designed for the atmospheric characterization of close-in directly imaged exoplanets. It integrates different observing techniques (photometry, low-resolution spectroscopy, and medium-resolution cross-correlation spectroscopy) into a single robust Bayesian retrieval process. By handling the complexities of stellar speckle contamination, the code enables more accurate constraints on atmospheric thermal and chemical properties. It relies on the `petitRADTRANS` radiative transfer package and `PyMultiNest` for sampling. |
|---------------------------------------------------------------------------------------------------------|
| **`Exo_k`** 
@Leconte2021 | `Exo_k` is a `Python` 3 library designed to handle radiative opacities for atmospheric applications. It provides tools to efficiently interpolate, convert, and adapt correlated-k and cross-section tables from a wide variety of sources and formats (e.g., Exomol, Nemesis, TauREx). Beyond opacity management, the library features an integrated radiative transfer framework and a full-fledged 1D atmospheric evolution model capable of computing the physical state of planetary atmospheres in radiative-convective equilibrium. | 
|---------------------------------------------------------------------------------------------------------|
| **`NEMESIS`** 
@Irwin2008 | `NEMESIS` (Non-linear optimal Estimator for MultivariatE spectral analySIS) is a general-purpose radiative transfer and retrieval package designed for the characterization of planetary atmospheres, applicable to both Solar System bodies and exoplanets. It consists of two primary components: the retrieval tool itself and the underlying `RADTRAN` radiative transfer code. The framework is highly versatile, supporting a wide range of observing geometries and capable of operating in both correlated-k and line-by-line modes to analyze diverse spectral data. | 
|---------------------------------------------------------------------------------------------------------|
| **`petitRADTRANS`** 
@Molliere19 | `petitRADTRANS` is a versatile `Python` package that solves the radiative transfer equation for both transmission and emission geometries, featuring an updated opacity database that includes various isotopologues. Incorporating both low-resolution correlated-k and high-resolution Line-by-Line treatments, it enables the modeling of exoplanet atmospheres across a wide range of temperatures and pressures. Furthermore, it includes a robust integrated retrieval module that utilizes nested sampling to infer atmospheric parameters, such as chemical abundances and temperature profiles, directly from observed spectra. | 
|---------------------------------------------------------------------------------------------------------|
| **`SEDA`** 
@Suarez2025 | `SEDA` (Spectral Energy Distribution Analyzer) is an open-source `Python` package designed for the forward modeling and empirical analysis of ultracool objects, including brown dwarfs, giant exoplanets, and low-mass stars. It facilitates comparisons between observed spectro-photometric data and grids of atmospheric models. The toolkit operates within a Bayesian framework to sample posteriors or utilizes chi-square minimization to determine best-fit models, while also providing data visualization functions. | 
|---------------------------------------------------------------------------------------------------------|
| **`species`** 
@Stolker20 | `species` is a toolkit designed for the spectral and photometric analysis of directly imaged exoplanets and brown dwarfs. It creates a unified framework that integrates observational data with various models and analysis tools, including flux calibration, spectral classification, analysis of evolutionary models, interpolation of model grids, parameter inference, and the creation of color-magnitude diagrams. | 


The codes mentioned in Table 3 are just a few examples; many others exist. Notably, `APOLLO` [@Howe2017] is designed to model spectra and phase curves for transit spectroscopy. For high-resolution spectroscopy (R$_{\lambda} > 20,000$), `ExoJAX` [@Kawahara22] offers an auto-differentiable framework built on `JAX`. `HOMER` [@Himes22] focuses on 1D radiative-convective equilibrium, incorporating disequilibrium chemistry and photochemistry. For Bayesian inference, `Pyrat Bay` [@Cubillos21] provides a versatile retrieval framework using MCMC and Nested Sampling. Finally, `rfast` [@Robinson23] is a high-speed, semi-analytical code optimized for the thermal and reflected light of rocky exoplanets.


# Appendix B:  Accuracy performances

Figure 3 illustrates the retrieval accuracy for the five parameters of the `Exo-REM k26` grid (T$_{eff}$, log(g), [M/H], C/O, and $f_{sed}$) as a function of spectral resolution, signal-to-noise ratio (S/N), and number of live points, following the same parameters explored in Figure 3. 

The synthetic spectra simulate a K-band observation (1.9–2.4 $\mu$m), which significantly limits the coverage of the spectral energy distribution (SED). At low resolution and low S/N, the impact is most pronounced for parameters sensitive to the SED shape ($T_{eff}$, log(g), and $f_{sed}$). Overall, above ~30, the number of live points has minimal effect on the retrieval accuracy.


![Accuracy comparison using `PyMultiNest` with varying spectral resolution (R$_{\lambda}$), signal-to-noise (S/N), and number of live points. Each dotted red line represents the expected value and black points the retrieved posteriors for each parameter explored during the nested sampling. The default setup is R$_{\lambda}$ = 368, S/N = 22, and 215 live points. Each plot illustrates the effect of varying one parameter while keeping the others fixed.](accuracy_formosa.png)



# Appendix C: Research impact statement 

Initiated in 2020, **`ForMoSA`** has since been employed for a variety of atmospheric analyses of substellar objects, owing to its flexibility regarding model selection and data compatibility. These works are based on different observations and also on different physical models.

To track the scientific output associated with this tool, we maintain a specific public library on the NASA ADS platform[^4].  


Here, we first showcase the standard usage of **`ForMoSA`** across different spectral resolutions, ranging from medium (R$_{\lambda}$ < 10,000) to low resolution and broadband photometry. We then demonstrate the application of this method to large homogeneous target libraries observed and extracted under similar conditions. Subsequently, we discuss results obtained by combining heterogeneous datasets, and finally, we present the latest developments extending **`ForMoSA`** to analyze high-contrast, high-resolution (R$_{\lambda}$ > 10,000) observations.

[^4]: **`ForMoSA`** public library: [NASA ADS](https://ui.adsabs.harvard.edu/public-libraries/PekELjOGR4yl3XOnGwOAng)


Hereafter, we provide a brief overview of the 13 works published in peer-reviewed journals, as well as several recent submissions.

### Standard usage

**Medium resolution spectroscopy**  

@Petrus21 re-detected the planetary-mass companion HIP 65426 b in VLT/SINFONI data using the molecular mapping approach, which has recently emerged as a promising technique to filter stellar residuals in datacubes of planetary systems obtained with Integral-Field Units such as VLT/ERIS or Keck/Oriris. They used **`ForMoSA`** to demonstrate that molecular mapping, coupled with atmospheric models, was able to characterize the atmosphere of HIP 65426 b.

@Petrus23 characterized the companion VHS 1256 b, observed with VLT/X-Shooter (R$_{\lambda}$ ~ 8000, $\Delta\lambda$ = 0.5–2.5 $\mu$m). They fitted the data using the `ATMO` model and identified the limitations of the approach, especially the non-reproducibility of the data at wavelengths below 1.0 $\mu$m and the non-physical radius retrieved. They also showed that the well-known variability of this object could be traced via the Teff estimated with **`ForMoSA`**.

**Low resolution spectroscopy**  

@Palma24 analyzed the low-resolution spectrum of the companion AF Lep b obtained with VLT/SPHERE (R$_{\lambda}$ ~ 30, $\Delta\lambda$ = 0.95–1.65 $\mu$m). They used the `Exo-REM` model for the fits and found atmospheric properties consistent with a young, cold, early-T super-Jovian planet.

**Photometry only**  

@Carter23 successfully extracted seven photometric points ($\Delta\lambda$ = 2.5–15.5 $\mu$m) of HIP 65426 b, the first planetary-mass companion imaged with JWST/NIRCam + MIRI. They fitted these points using the `BT-Settl` models and found atmospheric properties consistent with previous studies.


### Large population studies

The relatively short computational time required to perform a fit with **`ForMoSA`** enables the homogeneous analysis of large spectral libraries. This approach allows for comparative studies aimed at identifying and mitigating potential systematic errors introduced by the adopted models. Such a strategy improves the robustness of the results and reduces the risk of misinterpretation.

@Petrus20 compared VLT/X-Shooter spectra of nine warm brown dwarfs with the `BT-Settl` atmospheric models. They identified systematic discrepancies between the observations and the models that could be partially alleviated by introducing interstellar extinction (A$_{V}$) as a free parameter. This result points to an incomplete treatment of clouds, dust, and haze in the models, leading to biases in the retrieved colors, and consequently, in the inferred Teff of the targets.

@PalmaBifani2025 characterized the atmospheres of 21 M5–L5 companions and isolated brown dwarfs observed with VLT/SINFONI (R$_{\lambda}$ ~ 4000; $\Delta\lambda$ = 1.95–2.45 $\mu$m). They measured a decrease in Teff of more than 500 K across the M/L transition. They also confirmed that the retrieved atmospheric properties are strongly model-dependent, making their physical interpretation challenging. Finally, they found no significant differences between the atmospheric properties of companions and those of isolated objects.

@Petrus2025 analyzed the X-SHYNE library, which comprises 43 young (<500 Myr), low-mass (<20 MJup), and cold (Teff ~ 600–2000 K) isolated brown dwarfs and wide-separation companions observed with VLT/X-Shooter. They compared the physical properties predicted by evolutionary models with those retrieved using **`ForMoSA`** in combination with atmospheric models. One of the main results of the study is the identification of a discrepancy in the inferred surface gravity (log g). This inconsistency is likely linked to the diversity of cloud coverage in the atmospheres of X-SHYNE objects, potentially driven by differences in viewing geometry (e.g., equator-on versus pole-on orientations). Overall, the objects in the X-SHYNE sample exhibit near-solar metallicities and C/O ratios, as expected for objects formed in isolation as failed stars.


### A framework for combination of datasets

The spectra provided by a given instrument are inherently limited by its spectral resolution and wavelength coverage. To maximize the diversity of spectral information used to constrain an object’s atmosphere during the fitting process, it can be advantageous to combine data acquired with different instruments. However, the resulting inhomogeneity of the combined dataset must be carefully accounted for in the interpretation of the results to avoid introducing biases.

@PalmaBifani2023 combined multiple spectroscopic datasets and photometric measurements to cover a broad fraction of the spectral energy distribution of the planetary-mass companion AB Pic b. They included J, H, and K band spectra ($\Delta\lambda$ = 1.1–1.4, 1.45–1.85, and 1.95–2.45 $\mu$m, respectively) obtained with VLT/SINFONI (R$_{\lambda}$ ~ 1500–4000), L'-band spectroscopy from Magellan-AO/CLIO2 (R$_{\lambda}$ ~ 300, $\Delta\lambda$ = 3.4–4.1 $\mu$m), as well as photometric measurements from HST/WFC3 ($\Delta\lambda$ = 0.53–0.92 $\mu$m) and Spitzer/IRAC ($\Delta\lambda$ = 3.6–8 $\mu$m). They confirmed that the derived atmospheric properties are both model- and wavelength-dependent, and that the associated uncertainties are likely underestimated.

@Petrus24 extended the characterization of VHS 1256 b previously performed by @Petrus23 using **`ForMoSA`** by analyzing its spectrum obtained with JWST/NIRSpec+MIRI (R$_{\lambda}$ ~ 1500–3500, $\Delta\lambda$ = 1.0–18.0 $\mu$m). They exploited the wavelength dependence of the derived parameters to quantify biases induced by the models and propagated this systematic contribution directly into the uncertainties of the derived atmospheric properties. This approach enabled a more robust characterization of VHS 1256 b. In addition, they used **`ForMoSA`** to estimate the pseudo-continuum at the location of the absorption feature at 10 $\mu$m, which is not reproduced by current atmospheric models. This allowed them to measure its equivalent width and quantify the cloud content in the atmosphere of the object.

@Houlle2025 presented an analysis of the first VLTI/MATISSE observations of $\beta$ Pic b. In this study, they combined VLTI/MATISSE data obtained in the L and M bands (R$_{\lambda}$ ~ 500, $\Delta\lambda$ = 2.75–5.0 $\mu$m) with previous VLTI/GRAVITY observations in the K band (R$_{\lambda}$ ~ 500, $\Delta\lambda$ = 2.0–2.45 $\mu$m). Using **`ForMoSA`**, they fitted the two datasets both independently and jointly, and found that the derived atmospheric parameters vary depending on the fitting configuration adopted.

@Ravet25 developed the `MOSAIC` module within **`ForMoSA`** to mitigate biases arising from the combination of independent datasets obtained with different instruments. The method consists of computing an individual likelihood for each dataset and then combining them into a meta-likelihood, after introducing a scaling parameter, $\alpha$, retrieved during the fit to account for potential inter-calibration offsets. They applied this strategy to an inhomogeneous dataset composed of spectra and photometric measurements of $\beta$ Pic b obtained with ten different instruments. Their results demonstrate the importance of analyzing independent datasets with an adapted statistical framework in order to avoid misinterpretation of the derived atmospheric properties.

@Ravet26 characterized the cold companion COCONUTS-2 b (T9) by combining photometric measurements from Spitzer (3.6 and 4.5 $\mu$m) and WISE W1 (3.32 $\mu$m), W2 (4.56 $\mu$m) with spectra obtained from FLAMINGOS-2 (R$_{\lambda}$ ~ 900, $\Delta\lambda$ = 1.01–2.50 $\mu$m), JWST/NIRSpec (R$_{\lambda}$ ~ 3000, $\Delta\lambda$ = 2.88–5.14 $\mu$m), and JWST/MIRI-LRS (R$_{\lambda}$ ~ 100, $\Delta\lambda$ = 5.45–11 $\mu$m). Using the `MOSAIC` module of **`ForMoSA`**, they were able to robustly characterize the atmosphere of this object. They also demonstrated that classical noise-scaling schemes are poorly suited for heterogeneous, multi-modal datasets with varying spectral resolution and signal-to-noise ratio. In contrast, parametric covariance models provide a more reliable alternative for low- to moderate-sized observations.


### The high-contrast high-resolution (HCHR) module

@Denis25 compared VLT/HiRISE data (R$_{\lambda}$ ~ 140,000, $\Delta\lambda$ = 1.43–1.77 $\mu$m; @vigan_first_2024) of AF Lep b with the `Exo-REM` atmospheric model. They were able to precisely measure the radial velocity of the companion to robustly constrain its orbit. They also confirmed the chemical abundances estimated from previous studies.

@Denis26 extended the methodology previously developed for AF Lep b to the colder planet 51 Eri b (~600 K), enabling the measurement of its radial velocity and revealing the complexity of its orbit.

### Extended use cases of `ForMoSA`

@Hoch2025 used **`ForMoSA`** to model the circumplanetary disk (CPD) detected around the exoplanet YSES 1 b. They combined the `Exo-REM` atmospheric model with a simple CPD model consisting of a single blackbody component. From this analysis, they derived a disk effective temperature Teff = 371 ± 50 K and a radius of 7.35 ± 2.25 RJup, consistent with results obtained using the parametric model `petitRADTRANS`.

Radcliffe et al., (accepted) implemented a two-column approach within **`ForMoSA`** to improve the fitting performance of heterogeneous atmospheres using 1D atmospheric models. This strategy consists of calling the same grid of precomputed synthetic spectra twice at each iteration and combining them into a single synthetic spectrum using a weighting coefficient treated as a free parameter. This approach effectively simulates a heterogeneous atmosphere composed of two types of cloud patches distributed across the surface. They tested this method using a new version of the `Exo-REM` grid, which now includes the sedimentation factor fsed as a free parameter. The dataset analyzed was the JWST spectrum of VHS 1256 b. Their results show a significant improvement in fit quality compared to previous analyses.






# References
