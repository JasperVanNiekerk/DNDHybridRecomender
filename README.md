# D&D Hybrid Recommender System

A sophisticated hybrid recommendation system for Dungeons & Dragons character builds that combines collaborative filtering, content-based filtering, and narrative analysis to provide personalized recommendations for feats, weapons, armor, and next class choices.

## 🎯 Overview

This project implements a multi-layered recommendation system that analyzes D&D character data to suggest:
- **Feats** that complement a character's build
- **Weapons** suitable for the character's class and playstyle
- **Armor** appropriate for the character's capabilities
- **Next class** recommendations for multiclassing

The system uses a hybrid approach that blends three recommendation strategies:
1. **Item-based Collaborative Filtering** - "Characters like yours also used..."
2. **Narrative-based Content Filtering** - "Characters with similar backstories prefer..."
3. **Popularity-based Recommendations** - "Most commonly chosen options..."

## 🏗️ Architecture

### Core Components

- **`recs/`** - Main recommendation engine modules
  - `hybrid.py` - Hybrid recommendation blending logic
  - `baselines.py` - Collaborative filtering algorithms (ItemKNN, popularity)
  - `text.py` - TF-IDF-based narrative similarity analysis
  - `legal.py` - D&D rules compliance and eligibility checking
  - `evaluate.py` - Leave-one-out evaluation framework
  - `class_eligibility.py` - Multiclass ability score requirements
  - `features.py` - Data normalization and feature engineering
  - `parsing.py` - Character data parsing utilities
  - `vocab.py` - Vocabulary management and data type handling
  - `tune.py` - Hyperparameter optimization
  - `report.py` - Analysis and reporting utilities

- **`scripts/`** - Execution and evaluation scripts
  - `preprocess.py` - Data preprocessing pipeline
  - `hybrid_eval.py` - Main evaluation and recommendation generation
  - `recommend_next_class_hybrid.py` - Next class recommendation system
  - `build_and_eval.py` - Build and evaluation pipeline

- **`data/`** - Data storage
  - `raw/` - Original character data (Excel format)
  - `processed/` - Processed and normalized data (Parquet format)

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Required packages (see `requirements.txt`)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DNDHybridRecomender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Place your character data Excel file in `data/raw/characters.xlsx`
   - Ensure the Excel file contains columns for classes, feats, weapons, armor, and narrative fields

### Usage

1. **Preprocess the data:**
```bash
python scripts/preprocess.py
```

2. **Generate recommendations:**
```bash
python scripts/hybrid_eval.py
```

3. **Get next class suggestions:**
```bash
python scripts/recommend_next_class_hybrid.py
```

## 📊 Data Format

### Input Data Requirements

Your Excel file should contain the following columns:

**Required:**
- `class` or `classes` - Character class information (e.g., "Fighter (Battle Master) Level 5 | Wizard (War Magic) Level 3")

**Optional but recommended:**
- `feats` - Character feats (comma/pipe/semicolon separated)
- `weapons` - Weapon choices (comma/pipe/semicolon separated)
- `armor` - Armor choices (comma/pipe/semicolon separated)
- `appearance` - Character appearance description
- `backstory` - Character backstory
- `ideals` - Character ideals
- `bonds` - Character bonds
- `flaws` - Character flaws
- `personality` - Character personality traits

**For multiclass recommendations:**
- Ability score columns (strength, dexterity, constitution, intelligence, wisdom, charisma)

### Output Data

The system generates several output files in the `processed/` directory:

- `recommendations.csv` - Top recommendations for each character
- `recommendations_explained.csv` - Detailed explanations with attribution scores
- `next_class_hybrid.csv` - Next class recommendations
- `next_class_explained.csv` - Detailed next class explanations with eligibility info

## 🔧 How It Works

### 1. Data Preprocessing

The system normalizes character data into three main tables:
- **Mechanical**: Class levels, feats, weapons, armor (structured data)
- **Narrative**: Concatenated text from appearance, backstory, ideals, etc.
- **Classes Long**: Exploded class information for multiclass analysis

### 2. Recommendation Generation

For each recommendation type (feats, weapons, armor), the system:

1. **Builds collaborative filtering models:**
   - Item-item co-occurrence matrices
   - Popularity rankings

2. **Creates narrative similarity models:**
   - TF-IDF vectorization of character descriptions
   - Cosine similarity for finding similar characters

3. **Blends recommendations:**
   - Combines collaborative, narrative, and popularity signals
   - Applies D&D rules compliance (ability score requirements, class restrictions)
   - Optimizes weights through hyperparameter tuning

### 3. Evaluation

The system uses leave-one-out evaluation:
- Hides one item from each character's known items
- Generates recommendations based on remaining items
- Measures recall@5 and MRR@5 (Mean Reciprocal Rank)

## ⚙️ Configuration

### Weight Tuning

The system automatically tunes recommendation weights for each field:

```python
# Default weights (can be overridden)
W_ITEMKNN = 0.5  # Collaborative filtering weight
W_NEIGH   = 0.4  # Narrative similarity weight  
W_POP     = 0.1  # Popularity weight
```

### Eligibility Rules

The system enforces D&D multiclass requirements:
- Barbarian: STR 13+
- Bard: CHA 13+
- Cleric: WIS 13+
- Druid: WIS 13+
- Fighter: STR 13+ OR DEX 13+
- Monk: DEX 13+ AND WIS 13+
- Paladin: STR 13+ AND CHA 13+
- Ranger: DEX 13+ AND WIS 13+
- Rogue: DEX 13+
- Sorcerer: CHA 13+
- Warlock: CHA 13+
- Wizard: INT 13+
- Artificer: INT 13+

## 📈 Performance

The system provides detailed evaluation metrics:

- **Recall@5**: Percentage of held-out items found in top-5 recommendations
- **MRR@5**: Mean reciprocal rank of held-out items
- **Attribution**: Detailed breakdown of which signals contributed to each recommendation

Example output:
```
[feats] rows=150 nonempty=120 avg_len=3.45
feats    -> Pop     R@5:0.234 MRR@5:0.156 (n=120)
         ItemKNN R@5:0.267 MRR@5:0.189 (n=120)
         Hybrid* R@5:0.312 MRR@5:0.223 (n=120)  w=(0.35, 0.55, 0.10)
```

## 🛠️ Customization

### Adding New Recommendation Fields

1. Add the field to your input data
2. Update the field mapping in `recs/features.py`
3. Add evaluation logic in `scripts/hybrid_eval.py`

### Modifying Eligibility Rules

Update the `REQS` dictionary in `recs/class_eligibility.py` to modify multiclass requirements.

### Adjusting Recommendation Weights

Modify the weight functions in `scripts/hybrid_eval.py` or `scripts/recommend_next_class_hybrid.py`.

## 📝 Example Output

### Character Recommendations
```csv
row_id,primary_class,top_feats,top_weapons,top_armor
0,fighter,"['great_weapon_master', 'polearm_master', 'sentinel']","['glaive', 'halberd', 'pike']","['plate_armor', 'splint_armor']"
```

### Detailed Explanations
```csv
row_id,field,item,score,from_itemknn,from_narrative,from_pop,penalty,primary_class
0,feats,great_weapon_master,0.847,0.234,0.456,0.157,0.0,fighter
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built for the D&D community
- Uses scikit-learn for machine learning components
- Leverages pandas for data manipulation
- Implements standard recommendation system algorithms adapted for D&D character optimization