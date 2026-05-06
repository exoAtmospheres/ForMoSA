.. _home:

ForMoSA 2.0.0 Documentation
===========================

ForMoSA is an open-source Python package for modeling exoplanetary atmospheres
using a forward modeling approach. It compares observed spectra and photometry
against grids of atmospheric models via nested sampling to derive posterior
distributions on physical parameters.

The public API in v2.0.0 is built around the :class:`ForMoSA.Analysis` class.
The surrounding documentation focuses on the workflows that can be verified from
the repository: loading observations from FITS files, loading atmospheric grids
from NetCDF files, adapting grids to the observations, running nested sampling,
and analysing or plotting the resulting posterior samples.

Highlights
----------

* Class-based workflow centred on :class:`ForMoSA.Analysis`
* Support for photometry, spectroscopy, MOSAIC-style multi-observation fitting,
  and high-contrast spectroscopy
* Three nested-sampling backends: ``nestle``, ``pymultinest``, and ``ultranest``
* Automatic retrieval of photometric filter curves through the SVO Filter
  Profile Service
* Structured configuration through Python dataclasses, with ``config.ini``
  generation and loading still available
* Built-in plotting for posterior distributions, chains, radar plots, best-fit
  spectra, CCFs, and RV-v sin i maps

Statement Of Need
-----------------

For the formal statement of need, scientific motivation, and citation details,
see `Petrus et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023A%26A...670L...9P/abstract>`_.

How ForMoSA Works
-----------------

.. graphviz::

   digraph formosa_workflow {
       rankdir=LR;
       node [shape=box, style="rounded,filled", fillcolor="#f7f7f7", color="#36536b"];

       paths [label="1. Define paths\nConfigPath"];
       analysis [label="2. Create Analysis\nload model grid + observations"];
       adapt [label="3. Adapt data and subgrids\nanalysis.adapt(...)"];
       sample [label="4. Run nested sampling\nanalysis.nested_sampling(...)"];
       inspect [label="5. Analyse and visualise\nNSResults, NSAnalysis,\nanalysis.plot(...)"];

       paths -> analysis -> adapt -> sample -> inspect;
   }

This sequence is the backbone of the class-based API. The same scientific
workflow can also be driven from a ``config.ini`` file through
:class:`ForMoSA.config.global_config.ConfigGenerator` and
:class:`ForMoSA.config.global_config.ConfigLoader`.

Example Outputs
---------------

The maintained source tree currently includes two end-to-end example notebooks:

* photometry-only: VHS 1256 b
* spectroscopy-only: AB Pic b (SINFONI K-band)

Additional tutorial material requested by the core developers is tracked in the
documentation structure below. Where a notebook or figure is not yet present in
the repository, the corresponding page uses a square-bracket placeholder rather
than inventing content.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   tutorials/index
   guidelines/index
   scaling
   api/index
   whats_new


Attribution
-----------

Please cite `Petrus et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023A%26A...670L...9P/abstract>`_
when using ForMoSA in research.


Issues
------

If you run into a bug, a documentation gap, or a workflow that no longer
matches the code, please open an issue on
`GitHub <https://github.com/exoAtmospheres/ForMoSA/issues>`_.


Acknowledgments
---------------

Our sincere thanks to `Code/Astro <https://semaphorep.github.io/codeastro/>`_.
