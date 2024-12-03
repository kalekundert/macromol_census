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

*Macromolecule Census* is a tool for identifying high-quality, non-redundant 
subsets of the biological assemblies in the protein data bank (PDB).  A 
particular emphasis is to accommodate all kinds of macromolecules; not just 
proteins.  Briefly, this process involves the following steps:

- Rank each structure by metrics including clash score, resolution, $R_{free}$, 
  Q-score, NMR restraints.

- Rank each assembly by biological relevance, subchain cover, and size.

- Cluster protein, DNA, and RNA molecules by sequence similarity.

- Cluster small molecules and branched polysaccharides by identity.

The primary use-case for this software is the creation of datasets for machine 
learning.  This typically entails iterating through each assembly in ranked 
order, adding unique training examples as they appear.
