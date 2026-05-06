.. _tutorials-mosaic:

MOSAIC
======

In ForMoSA 2.0.0, MOSAIC-style fitting means analysing multiple observations in
one run by providing multiple FITS paths in ``ConfigPath.observation_path`` and
by using list-valued configuration fields where needed.

What The Current Code Verifies
------------------------------

* ``ConfigPath.observation_path`` accepts a list of FITS files.
* ``Analysis`` loads those files into one :class:`ForMoSA.observation.observation_set.ObservationSet`.
* ``ConfigAdapt`` and ``ConfigInversion`` broadcast single values to all
  observations or validate one-entry-per-observation lists.
* The plotting configuration assigns colours per observation based on the
  central wavelength of each dataset.

Practical Advice
----------------

* Keep one FITS file per observation.
* Use informative facility and instrument labels.
* Decide early whether you want shared settings across all observations or one
  value per observation for fields such as ``target_res_obs``, ``target_res_mod``,
  ``wav_fit``, or ``logL_type``.

Status Of Worked Examples
-------------------------

* [Beta Pic b spectro+photo notebook pending.]
* [File-label conventions for MOSAIC examples to be added here.]
