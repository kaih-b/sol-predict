# RF vs MLP: Permutation Importance

## Results
![Bar Chart for Descriptor Importance via Permutation in Expanded MLP & RF](../exports/permutation_importance_comparison.png)

Although **logP** maintains its status as the best single predictor of aqueous solubility across both models, **HeavyAtomCount**, a descriptor pruned from the tuned RF for colinearity, provides the second strongest signal for the expanded MLP. In terms of chemical rationale, the most probable explanation is that **HeavyAtomCount** is providing a similar, but more interpretable signal for the relationship between molecule size and solubility, similar to how **BertzCT** and **MolWt** were second and third most influential for the tuned RF.

Across the board, though, the MLP seems to utilize the full array of descriptors, assigning nontrivial importance to many more descriptors than than the RF model, which relies almost entirely on **logP** for solubility signaling. Practically, this implies that the MLP is integrating multiple correlated descriptors to form its prediction, whereas the RF is behaving more as a single-signal model with minor adjustments from other factors. This does not imply the MLP is more “correct" (although that is implied by its lower mean RMSE and std), but it does indicate that the MLP is extracting signal from a broader portion of the descriptor space, represented most strongly by **HeavyAtomCount** and **TPSA**.

## Next (Final) Steps

- Create final visualizations comparing findings
- Formalize writeups
- Briefly analyze performance on completely unseen data
    - Find new datasets that contain SMILES (or where SMILES can be easily extracted) and experimental logS for many organic compounds