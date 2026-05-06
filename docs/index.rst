.. ForMoSA documentation master file, created by
   sphinx-quickstart on Sat Jul 27 09:35:15 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

   <br />

Forward Modeling Tool for Spectral Analysis (ForMoSA)
=====================================================

Welcome to the documentation of ForMoSA, an open-source Python package.
Using a forward modeling approach, we designed this tool to model exoplanetary atmospheres.
We encourage the community to exploit its capabilities!


.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   guidelines/index
   demos/index
   api/index
   whats_new



Attribution
+++++++++++

Please cite `Petrus et al (2023) <https://ui.adsabs.harvard.edu/abs/2023A%26A...670L...9P/abstract>`_.



Issues (?)
++++++++++

If you run into any other problem, please create an issue on `GitHub <https://github.com/exoAtmospheres/ForMoSA/issues>`_.



Version Track
+++++++++++++

- ``2.0.0`` Complete rewrite with a class-based API (``Analysis``), Python dataclass configuration, restructured package layout, automatic photometry filter retrieval, CCF / RV–v sin i analysis, structured logging, and typed error handling. **Not backwards-compatible with v1.x.** See :ref:`whats_new` for the full details.

- ``1.1.6`` Addition of high-contrast models, ultranest and automatically generated config files.

- ``1.1.2`` Version adapted for including multiple instruments and high spectral resolution observations.

- ``1.0.13`` First version distributed, presented at `Cloud Academy 3. <https://alienearths.space/cloud-academy-3/>`_.

- ``1.0.5`` First operational release.


Acknowledgments
+++++++++++++++
Our sincere thanks to `Code/Astro <https://semaphorep.github.io/codeastro/>`_. 
