---
title: 'ForMoSA: Forward Modeling tool for Spectral Analysis'
tags:
  - python
  - astronomy
  - exoplanets
  - forward modeling
  - atmosphere
authors:
  - name: Simon Petrus
    orcid: 0000-0003-0331-3654
    equal-contrib: true
    affiliation: "1, 2, 3, 4" # (Multiple affiliations must be quoted)
  - name: Paulina Palma-Bifani
    orcid: 0000-0002-6217-6867
    equal-contrib: true
    affiliation: "5, 6" # (Multiple affiliations must be quoted)
  - name: Matthieu Ravet
    orcid: 0009-0000-4898-4713
    equal-contrib: true
    affiliation: "5, 4, 7" # (Multiple affiliations must be quoted)
  - name: Allan Denis
    equal-contrib: true
    affiliation: "8" # (Multiple affiliations must be quoted)
  - name: Arthur Vigan
    orcid: 0000-0002-5902-7828
    equal-contrib: true
    affiliation: "8" # (Multiple affiliations must be quoted)
  - name: Mickaël Bonnefoy
    orcid: 0000-0001-5579-5339
    equal-contrib: true
    affiliation: "4" # (Multiple affiliations must be quoted)
  - name: Gaël Chauvin
    orcid: 0000-0003-4022-8598
    equal-contrib: true
    affiliation: "7" # (Multiple affiliations must be quoted)
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
 - name: LESIA, Observatoire de Paris, Université PSL, Sorbonne Univer-sité, Université de Paris, 5 place Jules Janssen, 92195 Meudon, France
   index: 6
 - name: Max-Planck-Institut für Astronomie, Königstuhl 17, 69117 Heidelberg, Germany
   index: 7
 - name: Aix Marseille Univ, CNRS, CNES, LAM, Marseille, France
   index: 8
date: 19 August 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post: 
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

`ForMoSA` is an open-source  `python` package designed for fitting spectra and/or photometry of directly imaged exoplanets using forward modeling. It employs the Nested Sampling [@skilling_nested_2004] algorithm and extensively utilizes the `xarray` package for simple and efficient manipulation of multi-dimensional arrays. `ForMoSA` was designed as a comprehensive tool; allowing users to adapt, invert, and plot data in a modular fashion.

# Statement of need

# Implementation and usage

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References