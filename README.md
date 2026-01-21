# 💧 Morpheus

**A bioisostere and R-group replacement tool for hit to lead and lead optimisation**

Morpheus is an interactive web application for molecular decomposition, fragment replacement, and retrosynthetic analysis. 

It helps medicinal & computational chemists explore and create focused libraries by replacing scaffold or R-group fragments with bioisosteres and evaluating the generated compounds.

## Features

### 🧩 Molecule Decomposition
- Input molecules via SMILES string or draw using the built-in Ketcher editor
- Automatically decomposes molecules into ring and non-ring fragments
- Visualize all fragments and select one for replacement

### 🔄 Bioisostere Replacement
- Once selected, search for similar fragments from a curated database of ChEMBL-derived fragments
- Generate new molecules by substituting the selected fragment

### 📊 Property Filtering
- Filter generated molecules by:
  - Molecular Weight (MW)
  - H-Bond Donors/Acceptors (HBD/HBA)
  - Topological Polar Surface Area (TPSA)
  - cLogP
  - Synthetic Accessibility (SA) Score
  - QED Drug-likeness Score
  - Tanimoto Similarity to parent molecule

### 🔬 3D Structure Visualization
- View molecules in 3D with py3Dmol
- Align generated molecules to input (reference) ligands

### 🧪 Retrosynthetic Planning
- Analyze synthetic routes for selected molecules using SynPlanner
- Visualizes retrosynthetic trees with building block identification
- Labels commercial building blocks with their IDs

### ⚠️ Structural Alerts
- The tool will discard generated molecules that have any of the pre-defined structural alerts. These are listed at the end (bottom) of the app

## Installation

### Prerequisites
- Python 3.9+ (Preferably 3.12.11)
- Conda (recommended) or pip

### Quick Setup with environment.yaml (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/morpheus.git
   cd morpheus
   ```

2. **Create the conda environment:**
   ```bash
   conda env create -f environment.yaml
   conda activate morpheus
   ```

3. **Run the app:**
   ```bash
   streamlit run morpheus.py
   ```

### Manual Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/morpheus.git
   cd morpheus
   ```

2. **Create a conda environment:**
   ```bash
   conda create -n morpheus python=3.12
   conda activate morpheus
   ```

3. **Install RDKit:**
   ```bash
   conda install -c conda-forge rdkit
   ```

4. **Install other dependencies:**
   ```bash
   pip install streamlit streamlit-ketcher mols2grid py3Dmol pandas
   ```

5. **Install SynPlanner (optional, for retrosynthetic planning):**
   ```bash
   pip install synplanner CGRtools
   ```

6. **Decompress data files:**
   The compressed data files (`.gz`) will be automatically decompressed on first run.
   Alternatively, manually decompress them:
   ```bash
   gunzip data/*.gz
   ```

## Running the App

1. **Activate your environment:**
   ```bash
   conda activate morpheus
   ```

2. **Run Streamlit:**
   ```bash
   streamlit run morpheus.py
   ```

3. **Open in browser:**
   The app will automatically open at `http://localhost:8501`

## Data Files

The `data/` folder contains:
- **Fragment databases** derived from ChEMBL for bioisostere replacement
- **Building blocks** with IDs for retrosynthetic planning

Large files are stored compressed (`.gz`) and decompressed on first run.

The `synplan_data/` folder (auto-downloaded on first use of retrosynthesis):
- Reaction rules and policy networks for SynPlanner
- Building block databases (~1GB total, downloaded automatically)

## Usage

1. **Enter a molecule**: Type a SMILES string or draw a molecule using the Ketcher editor
2. **View fragments**: The molecule is automatically decomposed into fragments
3. **Select a fragment**: Click on any fragment to select it for replacement
4. **Find replacements**: Adjust similarity threshold and view potential bioisosteres
5. **Generate molecules**: New molecules are created by replacing the selected fragment
6. **Filter & analyze**: Use property sliders to filter results, check for structural alerts
7. **Plan synthesis**: Select promising molecules for retrosynthetic analysis

## License

MIT License

## Acknowledgments

- [RDKit](https://www.rdkit.org/) - Cheminformatics toolkit
- [Streamlit](https://streamlit.io/) - Web app framework
- [SynPlanner](https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner) - Retrosynthetic planning
- [ChEMBL](https://www.ebi.ac.uk/chembl/) - Source of fragment data
