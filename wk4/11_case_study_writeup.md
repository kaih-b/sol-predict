# Case Study

## Molecule Used

The case study focused on the molecule which had the greatest magnitude of error between its predicted logS and experimental logS from the Delaney ESOL dataset. The molecule, anthraquinone, has the following SMILES identifier:

**O=C1c2ccccc2C(=O)c3ccccc13**

## Model Behavior

For anthraquinone, the tuned Random Forest model predicted a logS value of -2.85, which differs from the Delaney dataset's experimental logS value of -5.19 by a magnitude of 2.34. This represents an error 3.66x the size of the RMSE (0.639) across all datapoints tested. This failure is a case of extreme insolubility being inaccurately predicted as low-moderate insolubility.

This discrepancy seems to be primarily attributable to the RF model's heavy reliance on **logP** to predict solubility. logP is only moderately high (2.46), so the molecule is treated as a “typical” moderately lipophilic compound rather than an exceptionally insoluble one. Chemically, this corresponds with the molecule's poor hydration, strong lattice structure, and lack of hydrogen bond donors, which work together to prevent aqueous dissolution. With regards to the RF model in general, extremes tend to be underpredicted because outcomes are averaged across terminal leaves.

Beyond logP effects, anthraquinone is **rigid and highly planar**, leading to strong solid-state stabilization through tight crystal packing and π–π stacking. These features reduce the entropy gain, substantially suppressing spontaneous aqueous solubility. These features are not directly indicated by the top three descriptors - logP, BertzCT, MolWt. Aromaticity-related descriptors - AroProp, for example - are weakly weighted in the RF model, limiting its ability to account for planarity-driven insolubility.