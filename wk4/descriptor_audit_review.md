# Descriptor Audit Review

## Overview

The objective of the descriptor audit is to trim all redundant (colinear correlation of |r| > 0.9) and non-important (variance falls below 10<sup>-4</sup>) descriptors from the original dataset. The code in **descriptor_audit.py** is, for this collection of descriptors, unnecessary -- none of the descriptors had negligible variance or strong correlations. This is a positive indication for the validity of Week 3's work, at least for the Delaney ESOL dataset. 

The script is, however, very useful for repeating this experiment with larger collections of molecular descriptors, as you might do in a long-term or more intensive review of solubility properties. The script includes the analysis needed to decide which of two correlated descriptors to remove based on mean correlation with the other descriptors. Thus, it can be reused in future iterations for testing new molecular descriptors.

Removing redundant descriptors is a crucial part of analysis as it prevents the Random Forest from inflating importance estimates for colinear features and keeps claims about descriptor importance more statistically defensible.