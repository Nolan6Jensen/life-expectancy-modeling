# Life Expectancy Modeling

## Overview
This project analyzes global determinants of life expectancy using cross-country data from the World Health Organization (WHO). The goal is to identify which socioeconomic, behavioral, and health-related factors are most strongly associated with life expectancy after controlling for overlapping influences.

## Data
- Source: World Health Organization (WHO)
- Observations: ~150 countries
- Key variables:
  - Life expectancy (response)
  - Schooling (average years)
  - Development status
  - Alcohol consumption (including zero-consumption indicator)
  - BMI Index

## Methods
- Exploratory data analysis and visualization
- Multiple linear regression (OLS)
- Partial residual (CCPR) diagnostics
- Multicollinearity assessment (VIF)
- Variable selection via backward AIC
- Residual diagnostics (Q-Q plots)

## Key Findings
- Schooling is the strongest predictor of life expectancy after controlling for other variables.
- GDP per capita shows a strong marginal relationship with life expectancy, but provides little additional explanatory power once schooling is included, suggesting that schooling serves as a proxy for broader societal infrastructure and institutional development, absorbing much of the variation through which income affects longevity.
- Behavioral and structural indicators (alcohol consumption patterns, development status) are significant contributors.

## Files
- `analysis.ipynb`: Full exploratory analysis and modeling workflow
- `plot_tools.py`: Reusable plotting functions for regression diagnostics
- `figures/`: Generated visualizations used in analysis

## Tools
Python, pandas, numpy, matplotlib, statsmodels
