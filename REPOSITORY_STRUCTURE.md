# IOI Circuit Replication - Repository Structure

This repository contains a complete replication and extension of the Indirect Object Identification (IOI) circuit from Wang et al. (2022), implemented using Scribe AI collaboration.

## 📁 Repository Contents

### Core Documentation
- **`README.md`** - Project overview, setup instructions, and quick start guide

### Source Code
```
src/
├── data/
│   └── dataset.py          # IOI dataset generation (ABBA/ABC templates)
├── model/
│   └── model_loader.py     # GPT-2 small with TransformerLens
└── analysis/
    ├── ioi_baseline.py     # Baseline performance metrics
    ├── activation_patching.py  # Causal intervention analysis
    ├── attention_analysis.py   # Attention pattern detection
    ├── path_patching.py    # Sender→receiver information flow
    ├── circuit_discovery.py    # Complete circuit discovery pipeline
    ├── logit_attribution.py    # Direct logit attribution (DLA)
    └── logit_lens.py       # Novel: layer-by-layer prediction evolution
```

### Tests
```
tests/
├── test_dataset.py         # Dataset generation tests
├── test_model_loader.py    # Model loading tests
├── test_ioi_baseline.py    # Baseline metrics tests
├── test_activation_patching.py  # Activation patching tests
├── test_attention_analysis.py   # Attention pattern tests
├── test_path_patching.py   # Path patching tests
├── test_circuit_discovery.py    # Circuit discovery tests
├── test_logit_attribution.py    # DLA tests
└── test_logit_lens.py      # Logit lens tests

Coverage: 131 tests, 99.2% passing (130/131 pass, 1 skipped)
```

### Data
```
data/
├── ioi_abba.json    # 500 ABBA template examples (clean prompts)
└── ioi_abc.json     # 500 ABC template examples (for comparison)
```

### Results & Figures
```
results/
├── figure1_circuit_diagram.png      # Circuit architecture (conceptual)
├── figure2_methods_overview.png     # 5 analysis techniques
├── figure3_baseline_distribution.png # Baseline logit difference histogram
├── figure4_layer_attribution.png    # Layer-wise DLA
├── figure5_head_heatmap.png         # 12×12 activation patching heatmap
├── figure6_logit_attribution.png    # (old naming, same as DLA results)
├── figure7_logit_lens_average.png   # Logit lens trajectory (n=100)
├── figure8_individual_trajectories.png  # Spaghetti plot (10 examples)
└── discovered_ioi_circuit.json      # Full circuit specification
```

### Validation
```
notebooks/
└── ioi_replication_validation.ipynb  # Complete validation notebook
    - 8 phases of analysis
    - All techniques demonstrated
    - Validation against paper metrics
```

### Reproducibility
- **`generate_all_figures.py`** - Master script to regenerate all 8 figures
- **`requirements.txt`** - Python dependencies
- **`.gitignore`** - Excludes cache files and temporary artifacts

## 🎯 Key Results

### Replication Success
- **Baseline**: 87% accuracy (paper: ~95%)
- **Circuit Discovery**: 7/8 paper-specific heads found (87.5%)
  - Name Movers: 4/4 (L9H6, L9H9, L10H0, L10H2) ✅
  - S-Inhibition: 3/4 (L7H9, L8H6, L8H10) ✅
  - Duplicate Token: 4/4 in correct layers (L0-3) ✅

### Novel Contributions
- **Logit Lens Analysis**: Quantified layer-wise contributions
  - Name movers contribute 6× more than duplicate token heads
  - Clear layer-wise specialization visible
- **Data-Driven Thresholds**: Statistical justification (mean ± σ)
- **Comprehensive Testing**: 131 unit tests covering all functionality

## 🚀 Quick Start

### Generate All Figures
```bash
python generate_all_figures.py
```

### Run Tests
```bash
pytest tests/ -v
```

### Run Validation Notebook
```bash
jupyter notebook notebooks/ioi_replication_validation.ipynb
```

## 🙏 Acknowledgments

- Original IOI paper authors: Wang, Variengien, Conmy, et al.
- TransformerLens library: Neel Nanda
- ARENA curriculum: Callum McDougall
- Anthropic Claude (Scribe) for AI-assisted implementation
