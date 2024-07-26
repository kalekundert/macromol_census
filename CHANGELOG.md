# CHANGELOG



## v0.2.1 (2024-07-26)

### Fix

* fix: upgrade to polars 1.0 ([`e9b43e7`](https://github.com/kalekundert/macromol_census/commit/e9b43e7e2ce517ff25f42f39851c3afc089cdb68))


## v0.2.0 (2024-05-13)

### Feature

* feat: restore support for python~=3.10 ([`0bdbb3e`](https://github.com/kalekundert/macromol_census/commit/0bdbb3eef9c30be1c10257f6e6cab1770486cb45))


## v0.1.0 (2024-05-01)

### Chore

* chore: debug automated releases ([`a39786c`](https://github.com/kalekundert/macromol_census/commit/a39786c8543d984d633d3bc5ac9e228fb93527ae))

* chore: don&#39;t make assumptions about CWD when testing ([`367446d`](https://github.com/kalekundert/macromol_census/commit/367446d6f6cc30a5c0c084a810e0c3470ec04e6c))

* chore: add missing test dependency ([`7aee64d`](https://github.com/kalekundert/macromol_census/commit/7aee64dd8aa726711e701e7deb0853c6799806b8))

* chore: uncomment polars dependency ([`a8cba2f`](https://github.com/kalekundert/macromol_census/commit/a8cba2f67f2b1eae54da8aafef9e81a8e3e6905a))

* chore: update github actions ([`658b62a`](https://github.com/kalekundert/macromol_census/commit/658b62ac7c262dc0caad41d8a35230edebe79c49))

* chore: remove PISCES code ([`8ded7de`](https://github.com/kalekundert/macromol_census/commit/8ded7de17657f2fe7375d77b3d490abe9684a6ff))

* chore: apply cookiecutter ([`d3d7c83`](https://github.com/kalekundert/macromol_census/commit/d3d7c83d80d01987e0ab3b96f47496e7b76ca0b4))

### Feature

* feat: allow opening the database in read-only mode ([`d36156c`](https://github.com/kalekundert/macromol_census/commit/d36156c950078326115123b6968e9d7bd01d870e))

* feat: include pdb id in progress bar ([`246e03e`](https://github.com/kalekundert/macromol_census/commit/246e03ef3047ac65aeb0369cd504189fb21efaef))

* feat: distinguish between subchain symmetry mates

Specifically, this commit modifies the visitor API to expect (subchain
PDB id, symmetry mate) tuples rather than just subchain PDB ids.  This
causes different copies of the same subchain to be recognized as
different molecules, and the rest of the algorithm can proceed as
before. ([`c278f76`](https://github.com/kalekundert/macromol_census/commit/c278f76522a5c5cb4ad24d8121a4d2d5e2112988))

* feat: rank the assemblies within each structure ([`1d60b17`](https://github.com/kalekundert/macromol_census/commit/1d60b17cd049220247d1c5b2ccb1a7b4bd71d521))

* feat: better integrate visit_assemblies() with macromol_training

- Create a new visitor for each structure.  This makes it easier to
  cache structure-level data.

- Skip assemblies that have no viable subchains.

- Don&#39;t require candidates to be hashable. ([`ca731e1`](https://github.com/kalekundert/macromol_census/commit/ca731e17016752b17bc6caff4624be7fcf80531c))

* feat: ingest assembly metadata ([`4c754ad`](https://github.com/kalekundert/macromol_census/commit/4c754adc2f90ed0adf0b1500ce12a63a7c0a0fc9))

* feat: allow assembly picking to be customized ([`ae201ad`](https://github.com/kalekundert/macromol_census/commit/ae201adae0132fc95e9da1a45b99b929f9a26234))

* feat: exclude structures above 10Å resolution ([`b7e6ab0`](https://github.com/kalekundert/macromol_census/commit/b7e6ab0a848eb9e7e7138a982dc6d358ed7f8642))

* feat: show PDB ids for picked assemblies ([`eece3cd`](https://github.com/kalekundert/macromol_census/commit/eece3cd8a8b957207ba04dc08501a16e3b1a7603))

* feat: prefer subchain pairs from the same chain ([`92736c6`](https://github.com/kalekundert/macromol_census/commit/92736c619d245611c1180b770c3e7c0734994f2a))

* feat: filter ligands by molecular weight ([`0d7da3a`](https://github.com/kalekundert/macromol_census/commit/0d7da3a00c5360d9b6e0f66112ce2916c22e3730))

* feat: pick nonredundant assemblies ([`94fbe0f`](https://github.com/kalekundert/macromol_census/commit/94fbe0fba0a0e7e6ebbe07ed36532089eaa41227))

* feat: include subchains in the database ([`7949362`](https://github.com/kalekundert/macromol_census/commit/79493620e5869834e563f17e9828e826ba23542b))

* feat: identify non-full-atom models ([`048a9d5`](https://github.com/kalekundert/macromol_census/commit/048a9d55b98890a772cc362e60cce0ad1adc7992))

* feat: ingest validation reports ([`008ece7`](https://github.com/kalekundert/macromol_census/commit/008ece71dfa7d78fe0e8fcc21d48c22ce3c4e458))

* feat: ingest blacklist ([`764f4e6`](https://github.com/kalekundert/macromol_census/commit/764f4e676dee919d79364eca46f6d0c2df2718d6))

* feat: ingest entity clusters ([`961491e`](https://github.com/kalekundert/macromol_census/commit/961491e462fed35c8c6352f61ec3192cb39e19ed))

* feat: ingest assembly/chain/entity relationships ([`2a68d10`](https://github.com/kalekundert/macromol_census/commit/2a68d10112c8dddd0f7bb27b3caa9c7316d2d545))

* feat: parse PISCES files ([`22571d8`](https://github.com/kalekundert/macromol_census/commit/22571d81bb908f56ff0a3d9b77bcde7b999e4000))

* feat: implement I/O for atoms database ([`c326aea`](https://github.com/kalekundert/macromol_census/commit/c326aea91bfd639cd89fe39c947b877218544bc7))

### Fix

* fix: don&#39;t double-count assemblies with multiple matrices ([`10006c8`](https://github.com/kalekundert/macromol_census/commit/10006c888927e54f44a3e5b1901419a96183e2f3))

* fix: require that PDB entry ids are unique ([`088ff13`](https://github.com/kalekundert/macromol_census/commit/088ff13951544dfba34869781167c7cccf7d6a3a))

* fix: cluster subchains of the same entity together ([`202782d`](https://github.com/kalekundert/macromol_census/commit/202782d13555f931682cfbc4a8839b9a58448c6a))

* fix: use EMDB, not FSC 0.143, resolutions ([`f6d3bbd`](https://github.com/kalekundert/macromol_census/commit/f6d3bbd833989198bdb00d725b3de89fcf0432c2))

* fix: ignore clashscores of -1 ([`e91e4ca`](https://github.com/kalekundert/macromol_census/commit/e91e4cacc80d280dd6a5cab3c822b4410c953710))

* fix: link validation reports to the correct models ([`76091cc`](https://github.com/kalekundert/macromol_census/commit/76091cc87461299dc78c9ccb313f81cc9dba13e8))

* fix: don&#39;t interpret unspecified resolutions as 0Å ([`701e3ba`](https://github.com/kalekundert/macromol_census/commit/701e3baa130a2d348cddf4346aac0dd58466019b))

### Performance

* perf: insert picked assemblies in one batch ([`e9f5ebe`](https://github.com/kalekundert/macromol_census/commit/e9f5ebefbedf81298c057376a4ce92d16227ceb2))

### Refactor

* refactor: fix a number of linting errors ([`eab1fe3`](https://github.com/kalekundert/macromol_census/commit/eab1fe3f1dbe81b1899a5a3cda6a7db6fcd50d86))

* refactor: remove code for neighbor location dataset ([`5908acb`](https://github.com/kalekundert/macromol_census/commit/5908acb27af31e87065b0fcb1a931393ab61af5e))
