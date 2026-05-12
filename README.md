# Property Valuation Analysis

> CSCI316 Big Data Mining · University of Wollongong in Dubai · Python · Apache Spark · Ensemble Learning · Docker

This repository contains a distributed machine learning pipeline for classifying Dubai property valuations into low, medium, and high price categories using Apache Spark and ensemble learning techniques.

The project is built around a real dataset published by the Dubai Land Department through the Dubai Pulse open data platform, covering property valuation records from 2000 to 2026. The core question driving the project is how location and property type influence valuations across different property categories, and whether ensemble methods can improve on baseline classifiers when working with large, heterogeneous real estate data.

The full pipeline covers data ingestion, type casting, cleaning, feature engineering, manual 10-fold cross-validation, baseline modelling, bagging, boosting, and Docker containerisation for reproducibility.

---

## What This Project Does

- Ingests 88,129 Dubai property valuation records into Apache Spark with fully manual type casting
- Cleans the dataset by handling missing values, removing irrelevant columns, flagging missing area data, and eliminating duplicates
- Applies feature engineering including procedure area correction, percentile-based target variable creation, one-hot encoding for 217 area names and 63 property sub-types, and a 289-dimensional feature vector assembly
- Implements manual 10-fold cross-validation from scratch using deterministic fold assignment without any automated utility
- Trains and evaluates Logistic Regression and Decision Tree classifiers as baseline models
- Implements a bagging ensemble using Spark's RandomForestClassifier configured to simulate pure bootstrap aggregation
- Implements a multi-class SAMME boosting algorithm from scratch using depth-1 decision trees as weak learners with adaptive sample weight updates
- Compares all four models across accuracy, precision, recall, and F1-score using consistent cross-validation conditions
- Containerises the full pipeline using Docker for reproducibility across any system

---

## Project Objective

The main objective of this project was to build a scalable and reproducible big data pipeline for classifying Dubai property valuations.

This involved answering questions such as:

- How do location and property type influence property valuations?
- Are the relationships between property features and valuation categories linear or non-linear?
- Do ensemble methods meaningfully outperform single classifiers on this type of data?
- How can a distributed processing framework like Spark support machine learning workflows on large, heterogeneous datasets?
- Does bagging or boosting better suit this valuation classification problem?

---

## My Contributions

This was a 6-person group project. My specific contributions:

- **Spark pipeline development** - built and ran the scalable Spark pipelines that processed the 88K+ Dubai property records, handling data ingestion, type casting, cleaning, and transformation across the full dataset
- **Ensemble modelling** - contributed to building and cross-validating the ensemble models, applying the bagging and boosting approaches and evaluating performance across all 10 folds to achieve the 80% accuracy result
- **Docker containerisation** - built the Docker image and configured the containerised environment, ensuring the full pipeline could be reproduced on any machine without manual dependency setup
- **Documentation** - co-wrote the project report, covering the methodology, results, discussion, and conclusion sections

---

## Dataset

The dataset is `dld_valuation-open`, published by the Dubai Land Department through [Dubai Pulse](https://www.dubaipulse.gov.ae/data/dld-valuations/dld_valuation-open).

| Detail | Value |
| --- | --- |
| Raw rows | 88,129 |
| Rows after cleaning | 88,092 |
| Raw columns | 20 |
| Columns after cleaning | 15 |
| Years covered | 2000 to 2026 |
| Property value range | ~159 AED to over 16 billion AED |

Key numeric features include `property_total_value`, `actual_area`, and `procedure_area`. Categorical features include property type, property sub-type, and area name in both English and Arabic. The Arabic columns, redundant identifiers, and metadata fields were removed during cleaning.

---

## Repository Structure

```
316project1/
├── data/
│   ├── raw/
│   │   └── Valuation.csv                        # Original DLD dataset
│   ├── interim/
│   │   └── valuation_cleaned_updated.csv        # Cleaned output from notebook 01
│   └── processed/
│       ├── valuation_ml_ready.parquet           # Feature-engineered dataset
│       ├── valuation_label_ref.parquet          # Label reference table
│       └── valuation_ml_cv_ready.parquet        # Final CV-ready dataset with fold_id
├── notebooks/
│   ├── 01_cleaning.ipynb                        # Spark ingestion, type casting, cleaning
│   ├── 02_feature_engineering.ipynb             # Target variable, encoding, VectorAssembler
│   ├── 03_cross_validation.ipynb                # Manual 10-fold CV fold assignment
│   ├── 04_baseline_models.ipynb                 # Logistic Regression and Decision Tree
│   ├── 05_bagging_model.ipynb                   # Random Forest bagging ensemble
│   └── 06_boosting_ensemble.ipynb               # SAMME boosting from scratch
├── outputs/
│   ├── metrics/
│   │   ├── logistic_results.csv                 # Per-fold metrics for Logistic Regression
│   │   ├── tree_results.csv                     # Per-fold metrics for Decision Tree
│   │   ├── bagging_results_FINAL.csv            # Per-fold metrics for Bagging
│   │   └── boosting_results.csv                 # Per-fold metrics for Boosting
│   └── artifacts/
│       ├── 02_feature_engineering_outputs.zip   # Archived feature engineering outputs
│       └── valuation_ml_cv_ready.zip            # Archived CV-ready dataset
├── checkpoints/                                 # Spark RDD checkpoints for boosting stability
├── Dockerfile                                   # Docker image definition
├── requirements.txt                             # Python dependencies
└── README.md
```

---

## Methodology

### Data Ingestion and Cleaning

Schema inference was deliberately disabled when loading the CSV so that all columns were first read as strings. This allowed controlled manual type casting for all numeric and identifier columns before any analysis was performed.

Missing numeric values were filled with 0 and missing categorical values were filled with "Unknown". A boolean column `actual_area_missing` was created to preserve information about originally missing area records rather than silently discarding it. Arabic columns, identifier duplicates, and metadata fields were dropped. Duplicate rows were removed using `dropDuplicates()`.

### Feature Engineering

Several steps were taken to prepare the dataset for modelling.

The `procedure_area` column had 841 rows where the value was 0, which is unrealistic for a property record. These were replaced with the corresponding `actual_area` value, as the zeros appeared to be data entry errors.

The target variable was created by binning `property_total_value` into three classes using percentile thresholds at the 33rd and 66th percentiles. Fixed cutoffs were not used because the price distribution is highly right-skewed and fixed cutoffs produced severely imbalanced classes. The percentile approach produced roughly balanced classes of approximately 28,600, 29,300, and 30,100 records. `StringIndexer` then encoded the classes as numeric labels for Spark ML.

Three categorical columns were encoded using `StringIndexer` followed by `OneHotEncoder`: `property_type_en` (3 unique values), `property_sub_type_en` (63 unique values), and `area_name_en` (217 unique values). The `handleInvalid="keep"` setting was applied so that unseen categories in cross-validation folds would not break the pipeline.

`VectorAssembler` combined all features into a 289-dimensional vector. Median imputation was applied to any remaining numeric nulls because extreme property values distort the mean. The final feature-engineered dataset was saved as Parquet since Spark ML vectors do not serialise correctly into CSV.

### Manual 10-Fold Cross-Validation

Cross-validation was implemented entirely from scratch. Each record was assigned a `fold_id` using the rule `fold_id = row_id mod 10`. This guaranteed consistent fold assignment across runs and even distribution across 10 folds. No automated cross-validation utilities were used. For each fold k, training used `fold_id != k` and evaluation used `fold_id == k`. Metrics were averaged across all 10 folds.

### Models

**Logistic Regression** was used as the primary linear baseline. Regularisation and iteration limits were applied to control overfitting and ensure convergence.

**Decision Tree** was used as a non-linear baseline with a restricted maximum depth to balance overfitting and expressiveness.

**Bagging** was implemented using `RandomForestClassifier` configured with all features available at each split, so that diversity came entirely from bootstrapped samples rather than feature subsampling. The final configuration used 50 trees, maximum depth of 10, and seed 42.

**Boosting** was implemented from scratch using the SAMME multi-class algorithm with depth-1 decision trees as weak learners. Training ran for 10 rounds. Sample weights were updated after each round to focus on misclassified observations, and learner weights were computed using the SAMME formulation. Spark RDD checkpointing was used to improve runtime stability during weight updates.

### Docker Containerisation

The full pipeline runs inside a Docker container based on `python:3.11-slim` with OpenJDK installed for PySpark. All dependencies are installed from `requirements.txt`. The project directory is mounted at `/app` and Jupyter Notebook is exposed on port 8888. This isolates the runtime from the host system and allows any evaluator to rebuild the image and reproduce all results without manual dependency setup.

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| Logistic Regression | ~69.2% | ~68.6% | ~69.2% | ~68.8% |
| Decision Tree | ~75.6% | ~76.6% | ~75.6% | ~75.8% |
| Boosting (SAMME) | ~72.3% | ~72.5% | ~72.3% | ~72.3% |
| Bagging (Random Forest) | ~80.0% | ~80.0% | ~80.0% | ~80.0% |

All scores are averages across 10 cross-validation folds. Per-fold breakdowns are saved in `outputs/metrics/`.

Logistic Regression performed the weakest, which suggests that linear decision boundaries are not sufficient to capture the relationships between property features and valuation categories. The Decision Tree improved results by capturing non-linear patterns across location, area, and property type.

Boosting improved over the linear baseline but did not surpass the single Decision Tree. Bagging delivered the strongest and most stable results across all metrics, with consistent performance across all 10 folds. This suggests that variance reduction through bootstrap aggregation is particularly well suited to this dataset.

---

## Getting Started

### Prerequisites

- Docker installed on your machine

### Running with Docker

```bash
# Clone the repository
git clone https://github.com/ibrawr/property-valuation-analysis.git
cd property-valuation-analysis

# Build the Docker image
docker build -t property-valuation .

# Run the container
docker run -p 8888:8888 -v $(pwd):/app property-valuation
```

Then open `http://localhost:8888` in your browser and run the notebooks in order, from `01_cleaning.ipynb` through to `06_boosting_ensemble.ipynb`.

### Running Locally

```bash
pip install -r requirements.txt
jupyter notebook
```

### Dependencies

```
pyspark
pandas
pyarrow
numpy
jupyter
findspark
```

---

## Tech Stack

**Language:** Python 3.11  
**Distributed Processing:** Apache Spark (PySpark), SparkML  
**Feature Engineering:** StringIndexer, OneHotEncoder, VectorAssembler  
**Machine Learning:** Logistic Regression, Decision Tree, RandomForestClassifier, custom SAMME Boosting  
**Data Format:** Parquet (Snappy compressed), CSV  
**Containerisation:** Docker (python:3.11-slim, OpenJDK)  
**Data Analysis:** Pandas, NumPy, PyArrow  
**Dataset Source:** Dubai Pulse Open Data Platform

---

## Key Skills Demonstrated

- Distributed data processing with Apache Spark
- Manual type casting and schema control at ingestion
- Missing value handling and feature flagging
- Percentile-based target variable creation for imbalanced distributions
- High-cardinality categorical encoding at scale
- Manual cross-validation implementation without automated utilities
- Baseline modelling with Logistic Regression and Decision Tree
- Bagging via bootstrapped random forest configuration
- Boosting from scratch using the SAMME multi-class algorithm
- RDD checkpointing for long-running ensemble training
- Model comparison using averaged cross-validation metrics
- Docker containerisation for reproducible ML environments
- Working with real open government datasets

---

## Key Takeaways

This project shows that for a property valuation classification problem with heterogeneous features, extreme value dispersion, and high-cardinality categorical variables, non-linear and ensemble approaches clearly outperform linear baselines.

The most important finding is that bagging, through variance reduction via bootstrap aggregation, consistently outperformed both boosting and the single decision tree. This suggests the dataset has high variance across its 88,000 records that individual trees struggle to generalise from, but aggregated trees handle much more reliably.

Building both cross-validation and the SAMME boosting algorithm from scratch also reinforced a deeper understanding of how these methods actually work under the surface, rather than relying on black-box utilities.

---

## Project Status

This project is complete as part of the CSCI316 Big Data Mining module at the University of Wollongong in Dubai.

---

## Author

**Ibrar Bhatti**  
GitHub: [ibrawr](https://github.com/ibrawr)
