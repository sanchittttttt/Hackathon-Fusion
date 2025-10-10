# 🧬 Synapse.AI - Drug Binding Prediction Dashboard

Live Demo : https://synapse-ai-functional-bits.streamlit.app/

A production-quality Streamlit dashboard for AI-powered drug discovery and molecule-protein binding prediction.

## 🚀 Features

### Core Functionality
- **Fast Prediction**: Get binding affinity predictions in seconds using deep learning models
- **Industry Metrics**: Comprehensive calculations including Kd, Ki, IC50, EC50, and ΔG
- **Uncertainty Quantification**: Monte Carlo dropout for confidence estimation
- **Interactive Ranking**: Sortable, filterable results with confidence-based prioritization
- **Phase 1 Readiness**: AI-powered drug development scoring using ADMET heuristics

### Advanced Features
- **Substructure Highlighting**: Visual molecular analysis with important fingerprint bit highlighting
- **Fingerprint Analysis**: Interactive charts showing top important molecular features
- **Multiple Input Methods**: Text input (FASTA/SMILES) or CSV file upload
- **Export Capabilities**: Download results in CSV or JSON formats
- **Responsive Design**: Modern UI with professional styling and responsive layout

## 🏗️ Architecture

### Backend Components
- **Protein Encoder**: ESM-2 transformer for protein sequence encoding
- **Molecule Encoder**: Morgan fingerprints with RDKit for molecular representation
- **Binding Predictor**: Multi-layer perceptron for binding affinity prediction
- **Interpretability**: Fingerprint weight analysis and substructure highlighting
- **Metrics**: Comprehensive drug discovery metrics (pKd, Kd, Ki, IC50, EC50, ΔG)

### Frontend Components
- **Streamlit Dashboard**: Three-tab interface (Testing, Detailed Overview, About)
- **Input Forms**: FASTA/SMILES text areas and CSV upload functionality
- **Results Table**: Sortable, filterable dataframe with ranking
- **Visualizations**: Plotly charts for fingerprint analysis and binding comparisons
- **Highlights**: Substructure image generation and display

## 📋 Installation

### Prerequisites
- Python 3.8+
- PyTorch
- RDKit
- Streamlit
- Plotly

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd Hackathon-Fusion

# Install dependencies
pip install torch rdkit streamlit plotly transformers matplotlib scikit-learn pandas numpy

# Run the application
streamlit run app.py
```

## 🎯 Usage

### Input Data
1. **Text Input**: Paste protein FASTA sequences and SMILES strings directly
2. **CSV Upload**: Upload a CSV file with 'protein' and 'smiles' columns
3. **Broadcasting**: Single protein sequences are automatically broadcast to multiple molecules

### Parameters
- **MC Dropout Samples**: Number of samples for uncertainty estimation (10-100)
- **Confidence Threshold**: Minimum confidence score for filtering results
- **Minimum pKd**: Minimum binding affinity threshold
- **Top K Fingerprint Bits**: Number of important fingerprint bits to analyze
- **Substructure Highlights**: Toggle for generating molecular highlight images

### Results
- **Ranked Table**: Results sorted by pKd (descending) then confidence (descending)
- **Summary Metrics**: Best pKd, median confidence, average Phase 1 readiness
- **Fingerprint Charts**: Bar charts showing top important molecular features
- **Binding Analysis**: Scatter plots comparing binding probability vs Phase 1 readiness
- **Substructure Highlights**: Visual molecular analysis images
- **Export Options**: Download filtered results in CSV or JSON format

## 🔧 Technical Details

### Model Architecture
- **Protein Encoder**: ESM-2 (facebook/esm2_t12_35M_UR50D) with 480-dimensional output
- **Molecule Encoder**: Morgan fingerprints (radius=2, 2048 bits) using RDKit
- **Binding Predictor**: 4-layer MLP (2528 → 1024 → 512 → 256 → 1) with dropout

### Performance
- **Model Accuracy**: AUROC 0.87
- **Prediction Speed**: ~0.31s per molecule
- **Throughput**: Tested on 15,230+ molecules

### Caching
- Results are cached using `st.cache_data` to avoid recomputation for identical inputs
- Model objects are not cached to ensure fresh predictions

## 📊 Metrics

### Binding Affinity Metrics
- **pKd**: Negative log of dissociation constant
- **Kd**: Dissociation constant (nM)
- **Ki**: Inhibition constant (nM)
- **IC50**: Half-maximal inhibitory concentration (nM)
- **EC50**: Half-maximal effective concentration (nM)
- **ΔG**: Binding free energy (kcal/mol)

### Drug Development Metrics
- **Phase 1 Readiness**: Composite score based on Lipinski's Rule of 5 and toxicity alerts
- **Binding Probability**: Sigmoid-transformed pKd values
- **Confidence Score**: Uncertainty-based confidence estimation

## 🎨 UI Design

### Visual Theme
- **Primary Colors**: Purple (#6366F1) and Teal (#06B6D4)
- **Background**: Clean white theme with subtle shadows
- **Typography**: Modern sans-serif fonts with clear hierarchy
- **Layout**: Responsive three-column grid system

### Component Styling
- **Header Banner**: Gradient background with centered branding
- **Feature Chips**: Colored pills highlighting key capabilities
- **Metric Cards**: Clean cards with colored borders and large numbers
- **Buttons**: Gradient styling with hover effects
- **Tables**: Sortable with custom column formatting

## 🔮 Future Enhancements

### Planned Features
- **Gemini Integration**: LLM-powered insights and analysis (Round 2)
- **3D Visualization**: Interactive molecular structure viewing
- **Batch Processing**: Enhanced support for large-scale analysis
- **API Integration**: RESTful API for programmatic access

### Technical Improvements
- **Model Updates**: Continuous model retraining and improvement
- **Performance Optimization**: GPU acceleration and batch processing
- **Extended Metrics**: Additional ADMET properties and drug-likeness scores

## 📁 Project Structure

```
Hackathon-Fusion/
├── app.py                          # Main Streamlit application
├── predict.py                      # Prediction pipeline with run_pipeline function
├── train.py                        # Model training script
├── metrics.py                      # Drug discovery metrics calculations
├── .streamlit/
│   └── config.toml                 # Streamlit configuration and theming
├── models/
│   ├── protein_encoder.py          # ESM-2 protein encoding
│   ├── molecule_encoder.py         # Morgan fingerprint encoding
│   ├── binding_predictor.py        # MLP binding affinity predictor
│   └── saved_models/
│       └── binding_model.pt        # Trained model weights
├── utils/
│   ├── interpretability.py         # Fingerprint weight analysis
│   ├── molecular_highlight.py      # Substructure highlighting
│   └── phase1_readiness.py         # Drug development scoring
├── data/
│   ├── raw/
│   │   └── davis.csv              # Training dataset
│   └── processed/                 # Processed data files
├── outputs/
│   └── highlights/                # Generated substructure images
└── assets/
    └── ui/                        # UI reference images
```

## 🤝 Contributing

This project was developed for a hackathon focused on AI-powered drug discovery. The codebase is designed to be modular and extensible for future enhancements.

### Key Design Principles
- **No Hardcoded Data**: All inputs come from user interaction
- **Production Ready**: Error handling, caching, and responsive design
- **Extensible**: Clean separation of concerns for easy feature additions
- **User-Friendly**: Intuitive interface with comprehensive help and guidance

## 📄 License

This project is developed for educational and research purposes in drug discovery applications.

---

**Synapse.AI** - Accelerating drug discovery through AI-powered binding prediction and molecular analysis.
