# Legacy Julia experiments

This directory preserves exploratory material that is not part of the tested
simulation interface. Files here may contain stale absolute paths,
working-directory assumptions, duplicated implementations, or unguarded entry
points. They are retained for reference only.

Some files were recovered from checkpoint directories because no canonical
source file existed. Their `-checkpoint` suffix is deliberate.

Generated JLD2 and PNG artifacts were removed from this cleanup branch. They
remain available on `main` and in Git history at commit `015e907`. In
particular, the historical IFT and phi-four collision outputs predate the
two-particle-sector correction and must not be interpreted as valid
two-particle scattering results.
