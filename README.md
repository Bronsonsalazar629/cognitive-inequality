# Cognitive Inequality Research System

Investigates causal pathways from socioeconomic inequality to cognitive decline
in young adults (25-45) using NHANES, BRFSS, and GSS public health datasets.

## Start

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

- NHANES 2013-2014 (Mediation analysis)

- BRFSS 2022 (Validation)

- GSS 2010-2022 (Longitudinal Trends)

## License

MIT
