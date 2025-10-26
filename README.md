# AI & Robotics Adoption in Healthcare: County-Level Analysis

**Author:** Aaron Johnson, Drexel University  
**License:** MIT (see LICENSE file)

This repository contains standalone Python scripts for analyzing the relationship between healthcare technology adoption (AI and robotics) and county-level health outcomes. Each script is designed for independent execution to test specific hypotheses and perform targeted data analyses.

## 📚 Research Overview

This codebase supports doctoral dissertation research examining:
- AI and robotics adoption patterns in U.S. hospitals
- Causal effects of technology adoption on health outcomes (premature death, preventable hospitalizations)
- Geographic disparities in healthcare technology access
- Hospital system characteristics and technology adoption
- Spatial clustering and health equity implications

## 🗂️ Repository Structure

### Core Analysis Scripts

#### Causal Inference & Treatment Effects
- **`YPLL_AIPW_Analysis_062825_v25.py`** - Causal analysis using AIPW (Augmented Inverse Probability Weighting) to estimate technology adoption impact on premature death rates
- **`OLS_IPTW_AIPW_ADJPD_070225_v5.py`** - Three-part causal analysis (OLS, IPTW, AIPW) with state-clustered standard errors and FDR correction
- **`Replicate_Results_090625_v7.py`** - Comprehensive replication script for dissertation results including hypothesis testing and robustness checks

#### Moderation & Interaction Effects
- **`AHA_Survey_Regressions_v4.py`** - County-level regression analysis of AI & robotics adoption vs health outcomes
- **`ML_Multithreaded_Multiverse_101925_v1.py`** - State-blocked multiverse analysis with parallel processing for:
  - Direct adoption predictions (L1-Logit stability selection + Random Forest)
  - CHR × (AI/Robotics) moderation effects on health outcomes
  - AI × Robotics interaction effects
  - Pairwise AI × AI interactions
  - AI stacking/dose-response analysis

#### Robustness & Validity
- **`Negative_Controls_101825_v4.py`** - Comprehensive negative control suite including:
  - Negative control outcomes (NCO)
  - Negative control exposures (NCE)
  - Placebo outcomes
  - Temporal falsification tests
  - Permutation tests

#### Geographic & Spatial Analysis
- **`Geospatial_Analysis_061525_v4.py`** - Spatial analysis of technology adoption patterns with geospatial visualization
- **`Geospatial_Lorenz_Curve_070525_v14.py`** - Lorenz curve and Gini coefficient analysis for geographic equity assessment
- **`Driving_Distance_Blue_Grey_071925_v7.py`** - AI Hospital Equity Index (AHEI) calculation based on population-weighted drive times to AI-enabled hospitals
- **`LISA_Cluster_Regressions_072025_v3.py`** - Local Indicators of Spatial Association (LISA) cluster analysis with OLS regression on health outcomes

#### Descriptive & Comparative Analysis
- **`6_County_Categories_060925_v6.py`** - Descriptive analysis comparing six county categories based on hospital AI/robotics capabilities
- **`Hospital_System_Analysis_062125_v2.py`** - System-affiliated vs independent hospital technology adoption analysis

#### Visualization & Mapping
- **`Map_6_Hospital_Categories_v2.py`** - U.S. county-level choropleth maps showing hospital technology adoption categories (contiguous U.S., Alaska, Hawaii)
- **`6_Hospital_Category_Poster_071925_v1.py`** - Poster-ready visualization generation for conference presentations

#### Utilities
- **`Load_Parquet_File_071425_v4.py`** - Helper script for loading and exploring Parquet data files
- **`LaTex_Compiler_052525_v5.py`** - LaTeX compilation utility with smart BibTeX detection

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- PostgreSQL database with research data
- Anaconda or pip for package management

### Required Environment Variables

Set the following environment variables for database access:

```bash
export POSTGRES_HOST="your_host"
export POSTGRES_DB="your_database"
export POSTGRES_USER="your_username"
export POSTGRESQL_KEY="your_password"
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Usage

Each script is standalone and can be run independently. Most scripts follow this pattern:

```bash
python script_name.py
```

### Example: Running Multiverse Analysis

```bash
python ML_Multithreaded_Multiverse_101925_v1.py \
  --outdir ./multiverse_output \
  --n-reps 150 \
  --n-jobs 8
```

### Example: Negative Controls Suite

```bash
python Negative_Controls_101825_v4.py
```

## 📊 Data Requirements

Scripts expect access to PostgreSQL database views including:
- `vw_conceptual_model` - County-level health and technology data
- `vw_county_tech_summary` - Hospital technology adoption metrics
- County Health Rankings (CHR) data tables
- AHA (American Hospital Association) survey data

**Note:** Data access requires proper credentials and institutional data use agreements.

## 📝 Output Files

Scripts generate various outputs in designated directories:
- **CSV files**: Regression results, summary statistics, predictions
- **PNG/PDF visualizations**: Maps, plots, forest plots, heatmaps
- **Log files**: Detailed execution logs for reproducibility
- **Parquet files**: Intermediate data snapshots

Common output directories:
- `logs/` - Execution logs
- `output/`, `output_advanced/` - Analysis results
- `plots/`, `figs/`, `figs_advanced/` - Visualizations
- `multiverse_output/` - Multiverse analysis results
- `negative_controls_output/` - Robustness test results

## 🔬 Methodological Notes

### Causal Inference Methods
- **OLS with clustered SEs**: State-level clustering to account for spatial correlation
- **IPTW (Inverse Probability of Treatment Weighting)**: Propensity score weighting for causal effect estimation
- **AIPW (Augmented IPW)**: Doubly robust estimation combining propensity scores and outcome modeling
- **FDR correction**: Benjamini-Hochberg procedure for multiple testing

### Machine Learning Approaches
- **Stability Selection**: L1-regularized logistic regression with bootstrap resampling
- **Random Forest**: Ensemble method for feature importance and predictions
- **State-blocked cross-validation**: Prevents data leakage across geographic units

### Spatial Methods
- **Local Moran's I**: LISA cluster identification (HH, LL, HL, LH)
- **Lorenz curves & Gini coefficients**: Inequality measurement
- **BallTree nearest neighbor**: Efficient geospatial distance calculations

## 📖 Citation

If you use this code in your research, please cite:

```
Johnson, A. (2025). AI & Robotics Adoption in Healthcare: County-Level Analysis 
[Computer software]. Drexel University. https://github.com/undergroundbreakfast/research-scripts-data
```

## 🤝 Contributing

This is academic research code. For questions or collaboration inquiries, please contact the author.

## ⚠️ Disclaimer

This software is provided for research purposes only. Results should be interpreted within the context of the dissertation research design and limitations. The code represents association/correlation analysis and causal inference methods - interpret findings accordingly.

## 📄 License

MIT License - see LICENSE file for full text.

Copyright (c) 2025 Aaron Johnson, Drexel University
