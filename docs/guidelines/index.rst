.. _guidelines:

Guidelines
==========

This section collects practical rules of thumb for preparing inputs and running
analyses without fighting the file system.

Working Layout
--------------

A clean project layout pays for itself quickly, especially when you want to
reuse adapted grids or compare multiple targets:

.. code-block:: text

   /your/project/root/
   ├── atm_grids/
   ├── targets/
   │   └── target_name/
   │       ├── data/
   │       ├── adapted_grid/
   │       └── results/
   └── notebooks/  [optional]

Suggested practice:

* keep native model grids separate from per-target data;
* keep observation FITS files under ``data/``;
* keep adapted grids in a dedicated directory so they can be reused;
* keep sampler outputs and figures inside ``results/``.

Input Format References
-----------------------

The existing notebook references remain the most detailed file-format guides in
the repository:

.. toctree::
   :maxdepth: 1

   input_format/obs_format.ipynb
   input_format/model_format.ipynb

* [The legacy ``config_format.ipynb`` notebook is being refreshed for the
  class-based v2.0.0 workflow.]

What To Check Before A Run
--------------------------

* Make sure every observation FITS file matches the intended observation type.
* Use a NetCDF grid file that passes the ``GridLoader`` validation rules.
* Decide whether you are using the dataclass route, the INI route, or both.
* If you are fitting several observations together, check which configuration
  fields should be global and which should be per-observation.
* If you increase ``n_jobs``, remember that it affects grid adaptation rather
  than the nested-sampling backend itself.

Notes On Console Output
-----------------------

The v2.0.0 codebase has already moved a large part of the user feedback from
plain ``print`` statements to the Python ``logging`` stack.

* [Further cleanup of user-facing checkup messages is still desirable.]
