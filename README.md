# Cognitive Inequality Research System

Investigates causal pathways from socioeconomic inequality to cognitive decline
in young adults (25-45) using NHANES, BRFSS, and GSS public health datasets.

## Quick Start

```bash
cd cognitive-inequality-research

# Install dependencies
pip install -r requirements.txt

# Download and preprocess NHANES (auto-downloads from CDC)
python -m src.data.download_all_datasets

# Run descriptive analysis
python -m src.main analyze --dataset nhanes

# Run full pipeline
python -m src.main pipeline
```

## Datasets

| Dataset | N (ages 25-45) | Cognitive Measure | Primary Use |
|---------|---------------|-------------------|-------------|
| NHANES 2013-2014 | ~2,500 | DSST + CERAD (validated) | Mediation analysis |
| BRFSS 2022 | ~100,000 | Self-reported difficulty | Validation |
| GSS 2010-2022 | ~4,000 | Vocabulary test | Longitudinal trends |

## Architecture

```
src/
├── data/        # Data loaders (NHANES, BRFSS, GSS) + harmonization
├── analysis/    # Mediation, causal discovery, prediction model
├── llm/         # 4-tier LLM validation (DeepSeek)
├── simulation/  # Counterfactual intervention simulator
├── visualization/  # Publication-ready figures
└── utils/       # Bootstrap, survey weights, E-values
```

## License

MIT
