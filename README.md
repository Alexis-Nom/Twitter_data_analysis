# Twitter_data_analysis
Analysis of data from X (ex-Twitter) on alternative medicines to spot significant trends

**The Data**

The dataset used for this project can be found in the file *Data_adapted.xlsx*. It contains the number of search on X for 39 words related to alternative medicines during the period 2015-2024 for 3 languages : French, English and Spanish.

**Preliminary analysis**

The analysis of this data started with the test of variability of the data among the years. Tests of models (including linear regressions) for the whole period (2015-2024) showed inconsistencies for some words, namely when comparing the periods pre-COVID (2015-2019) and post-COVID (2020-2024). This led us to focus on the period 2015-2019, where it was more likely to detect significant trends. Yet, a limitation following this decision is the quality of the dataset. Indeed, limiting the dataset to the period 2015-2019 provides us datasets of 5 elements only for each word.

**Main analysis**

The main analysis of the dataset is composed of :
- an approximation of trends via a linear regression
- a first verification of these models through the computation of the coefficient of determination (models with a coefficient under 0.7 were disqualified)
- a second verification of these models through a permutation test and the computation of a p-value (models with a p-value over 0.05 were disqualified)
- the computation of a 95 % confidence interval for each significant evolution (e.g for each word which passed the two previous tests) through bootstraping
- visual representations of the analysis with several attempts (3 figures were finally kept, including arrays, bar diagrams)


**The code**

The code provided in this repository only contains the functions/scripts essentials to the final analysis of data and creation of the visuals, including :
- *significance_array.py* to compute the linear regression, the p-value and display them in an array
- *arrays_summary.py* to gather the results from *significance_array.py*
- *array_with_CI* has the same data as the ones given by *significance_array.py*, and adding the computation of 95 % CIs through bootstraping
- *diagram.py* to build a bar diagram representing the evolutions and the initial number of search for each significant word for each language, finally adding the CIs
- *diagrams_combined.py* builds the figure gathering the diagrams built with diagram.py
- *evolution_comparison.py* rearranges the data to compare each language for any significant word
