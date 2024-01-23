********************
Macromolecule Census
********************

.. image:: https://img.shields.io/pypi/v/macromol_census.svg
   :alt: Last release
   :target: https://pypi.python.org/pypi/macromol_census

.. image:: https://img.shields.io/pypi/pyversions/macromol_census.svg
   :alt: Python version
   :target: https://pypi.python.org/pypi/macromol_census

.. image:: https://img.shields.io/readthedocs/macromol_census.svg
   :alt: Documentation
   :target: https://macromol-census.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/github/actions/workflow/status/kalekundert/macromol_census/test.yml?branch=master
   :alt: Test status
   :target: https://github.com/kalekundert/macromol_census/actions

.. image:: https://img.shields.io/coveralls/kalekundert/macromol_census.svg
   :alt: Test coverage
   :target: https://coveralls.io/github/kalekundert/macromol_census?branch=master

.. image:: https://img.shields.io/github/last-commit/kalekundert/macromol_census?logo=github
   :alt: Last commit
   :target: https://github.com/kalekundert/macromol_census

*Macromolecule Census* is a set of tools for creating machine-learning datasets 
from macromolecular structure data, especially those made available by the 
protein data bank (PDB).  The purpose of these tools is to account for the 
following:

- Filter for high-quality (e.g. high resolution, low R-factor), low-redundancy 
  (i.e. sequence identity cutoffs) structures.

- Make robust training/validation/test splits by accounting for domain-level 
  structural similarities.

- Store atomic coordinates in a compact, portable, standard format (SQLite).
