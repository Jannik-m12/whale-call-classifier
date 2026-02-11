# Whale Call Classifier

## Overview
The Whale Call Classifier project aims to develop a machine learning model that can classify whale calls from audio recordings. This project includes data handling, feature extraction, model training, evaluation, and visualization components.

## Project Structure
```
whale-call-classifier
├── data
│   ├── raw
│   │   └── .gitkeep
│   └── processed
│       └── .gitkeep
├── notebooks
│   └── whale_call_classifier.ipynb
├── src
│   ├── __init__.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── download.py
│   │   └── preprocess.py
│   ├── features
│   │   ├── __init__.py
│   │   └── extract_features.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   ├── visualization
│   │   ├── __init__.py
│   │   └── plot_spectrograms.py
│   └── utils
│       ├── __init__.py
│       └── audio_utils.py
├── tests
│   ├── __init__.py
│   ├── test_preprocess.py
│   └── test_model.py
├── configs
│   └── config.yaml
├── requirements.txt
├── setup.py
└── README.md
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd whale-call-classifier
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
- Use the Jupyter notebook located in the `notebooks` folder for data exploration, model training, and evaluation.
- The source code is organized into modules for data handling, feature extraction, model training, evaluation, and visualization.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.