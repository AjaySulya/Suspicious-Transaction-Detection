# PaySim Fraud Detection 🔍💳

> A sophisticated machine learning web application that detects fraudulent transactions using advanced feature engineering and graph-based analysis.

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.25+-green.svg)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

[🚀 Live Demo](#-usage) • [📖 Documentation](#-about-the-project) • [🐛 Report Bug](https://github.com/AjaySulya/Suspicious-Transaction-Detection/issues) • [✨ Request Feature](https://github.com/AjaySulya/Suspicious-Transaction-Detection/issues)

</div>

---

## 📋 Table of Contents

- [🚀 About the Project](#-about-the-project)
- [📂 Project Structure](#-project-structure)
- [⚙️ Tech Stack](#️-tech-stack)
- [✨ Features](#-features)
- [📊 Dataset & Approach](#-dataset--approach)
- [📥 Installation](#-installation)
- [▶️ Usage](#️-usage)
- [📈 Results](#-results)
- [🛠️ Future Improvements](#️-future-improvements)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [🙌 Acknowledgments](#-acknowledgments)

---

## 🚀 About the Project

**PaySim Fraud Detection** is an advanced machine learning solution designed to identify fraudulent transactions in financial datasets. Built with a focus on **modularity**, **scalability**, and **interpretability**, this project combines traditional ML techniques with cutting-edge graph-based feature engineering.

### 🎯 Key Highlights

- **High Performance**: Achieves 99% accuracy with 99% recall
- **Advanced Features**: Graph-based analysis using NetworkX and PageRank
- **User-Friendly**: Interactive Streamlit web interface
- **Production-Ready**: Dockerized deployment with modular architecture
- **Comprehensive**: End-to-end ML pipeline with feature engineering

---

## 📂 Project Structure

```
PaySim-Fraud-Detection/
├── 📁 src/
│   ├── 📁 components/          # ML pipeline components
│   │   ├── data_ingestion.py   # Data loading and validation
│   │   ├── transformation.py   # Feature engineering
│   │   └── model_training.py   # Model training logic
│   ├── 📁 pipeline/            # End-to-end workflows
│   │   ├── training.py         # Training pipeline
│   │   └── prediction.py       # Prediction pipeline
│   └── 📁 utils/               # Utility functions
│       ├── save_load.py        # Model persistence
│       └── helpers.py          # Helper functions
├── 📁 artifacts/               # Trained models & features
├── 📄 app.py                   # Streamlit web interface
├── 📄 requirements.txt         # Production dependencies
├── 📄 requirements-dev.txt     # Development dependencies
├── 🐳 Dockerfile              # Production container
├── 🐳 Dockerfile.dev           # Development container
├── 📄 setup.py                 # Package configuration
└── 📄 README.md                # This file
```

---

## ⚙️ Tech Stack

<table>
<tr>
<td><strong>🧠 Machine Learning</strong></td>
<td>

![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)

</td>
</tr>
<tr>
<td><strong>🕸️ Graph Analysis</strong></td>
<td>

![NetworkX](https://img.shields.io/badge/NetworkX-PageRank-blue)

</td>
</tr>
<tr>
<td><strong>🌐 Web Interface</strong></td>
<td>

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)

</td>
</tr>
<tr>
<td><strong>🐳 Deployment</strong></td>
<td>

![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)

</td>
</tr>
</table>

---

## ✨ Features

### 🔥 Core Capabilities
- ✅ **Real-time Fraud Detection** - Instant predictions through web interface
- ✅ **Advanced Feature Engineering** - Cyclical encoding, delta balance analysis
- ✅ **Graph-based Analysis** - Transaction network analysis with PageRank
- ✅ **High Accuracy** - 99% accuracy with 98% precision
- ✅ **Modular Design** - Easily extensible codebase
- ✅ **Interactive UI** - User-friendly Streamlit interface
- ✅ **Containerized** - Docker support for easy deployment

### 🧮 Advanced Features
- **Cyclical Encoding**: Sine/cosine transformations for temporal and categorical features
- **Delta Balance Analysis**: Detection of unusual account balance changes
- **Network Analysis**: PageRank scoring to identify influential accounts
- **Transaction Graph**: NetworkX-based relationship mapping

---

## 📊 Dataset & Approach

### 📈 Dataset Overview
- **Source**: PaySim Financial Dataset
- **Size**: ~600,000+ transactions
- **Features**: Transaction type, amount, account balances, timestamps
- **Target**: Binary classification (fraud/legitimate)

### 🔬 Methodology

| **Phase** | **Approach** | **Techniques** |
|-----------|-------------|----------------|
| **Data Preprocessing** | Feature engineering & cleaning | Cyclical encoding, balance deltas |
| **Graph Analysis** | Network construction | NetworkX, PageRank scoring |
| **Model Training** | Binary classification | Scikit-learn algorithms |
| **Evaluation** | Performance metrics | Accuracy, Precision, Recall, F1 |

### 🎯 Feature Engineering Strategy

> **Cyclical Encoding**: Transform time and transaction type into sine/cosine pairs to capture cyclical patterns
> 
> **Delta Balance**: Calculate `newbalance - oldbalance` to detect suspicious balance changes
> 
> **Graph Features**: Use PageRank to identify accounts with high network influence

---

## 📥 Installation

### 📋 Prerequisites

- **Python** >= 3.8
- **pip** package manager
- **Virtual environment** (recommended)

### 🛠️ Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AjaySulya/Suspicious-Transaction-Detection.git
   cd Suspicious-Transaction-Detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   
   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Launch the application**
   ```bash
   streamlit run app.py
   ```

### 🐳 Docker Installation

**Production Environment:**
```bash
docker build -t Suspicious-Transaction-Detection .
docker run -p 8501:8501 Suspicious-Transaction-Detection
```

**Development Environment:**
```bash
docker build -f Dockerfile.dev -t Suspicious-Transaction-Detection-dev .
docker run -p 8501:8501 Suspicious-Transaction-Detection-dev
```

---

## ▶️ Usage

### 🌐 Web Interface

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the interface**
   
   Open your browser and navigate to: `http://localhost:8501`

3. **Make predictions**
   
   Fill in the transaction details and click **"Predict Fraud"**

### 📱 Interface Preview

The Streamlit interface provides:
- **Input Form**: Transaction details (amount, type, balances)
- **Real-time Prediction**: Instant fraud probability
- **Visual Feedback**: Clear fraud/legitimate classification
- **Interactive Elements**: User-friendly form controls

### 🔌 API Usage

```python
# Example prediction request
transaction_data = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 1000,
    "nameOrig": "C12345",
    "oldbalanceOrg": 5000,
    "newbalanceOrig": 4000,
    "nameDest": "C67890",
    "oldbalanceDest": 2000,
    "newbalanceDest": 3000
}

# API Response
{
    "fraud_prediction": 1,
    "status": "success"
}
```
## 🔑 Model & Artifacts
The trained model and feature files (`.pkl`) are not included in this repository due to GitHub's 100 MB file size limit.

You can download them here: [Google Drive / HuggingFace / S3 Link]

Place the files inside the `artifacts/` folder before running the project.

---

## 📈 Results

### 🏆 Model Performance

| **Metric** | **Score** | **Interpretation** |
|------------|-----------|-------------------|
| **Accuracy** | **99%** | Overall correctness |
| **Recall** | **99%** | Fraud detection rate |
| **Precision** | **98%** | False positive minimization |
| **F1-Score** | **98.5%** | Balanced performance |

### 📊 Key Insights

> **🎯 High Recall**: The model successfully identifies 99% of fraudulent transactions, crucial for financial security
> 
> **⚡ Low False Positives**: 98% precision ensures legitimate transactions aren't flagged unnecessarily
> 
> **🕸️ Graph Features**: Network analysis significantly improves detection of complex fraud patterns

### 🔍 Feature Importance

1. **Graph-based Features** (PageRank, Network Influence)
2. **Delta Balance Changes** (Unusual account movements)
3. **Transaction Amount** (Statistical outliers)
4. **Cyclical Time Features** (Temporal patterns)

---

## 🛠️ Future Improvements

### 🚀 Planned Enhancements

- [ ] **🤖 Ensemble Models** - Combine multiple algorithms for better performance
- [ ] **🧠 Deep Learning** - Implement neural networks for complex pattern detection
- [ ] **☁️ Cloud Deployment** - Deploy on AWS, GCP, or Azure
- [ ] **📊 Batch Processing** - API endpoints for bulk transaction analysis
- [ ] **🎨 Advanced Visualizations** - Interactive network graphs and fraud analytics
- [ ] **⚡ Real-time Streaming** - Process live transaction feeds
- [ ] **📱 Mobile App** - Native mobile interface

### 🔧 Technical Roadmap

| **Priority** | **Feature** | **Timeline** |
|--------------|-------------|--------------|
| **High** | Ensemble methods | Q1 2026 |
| **High** | Cloud deployment | Q1 2026 |
| **Medium** | Deep learning models | Q2 2026 |
| **Medium** | Batch API | Q2 2026 |
| **Low** | Mobile app | Q3 2026 |

---

## 🤝 Contributing

We welcome contributions! Here's how you can help improve this project:

### 🛤️ Contribution Workflow

1. **🍴 Fork** the repository
2. **🌿 Create** a feature branch
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **💾 Commit** your changes
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **🚀 Push** to the branch
   ```bash
   git push origin feature/amazing-feature
   ```
5. **📝 Open** a Pull Request

### 🎯 Contribution Areas

- 🐛 **Bug fixes** and performance improvements
- ✨ **New features** and enhancements
- 📚 **Documentation** improvements
- 🧪 **Test coverage** expansion
- 🎨 **UI/UX** enhancements

---

## 📜 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

```
Copyright 2025 Suspicious-Transaction-Detection Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
```

---

## 🙌 Acknowledgments

### 💡 Inspiration & Resources

- **PaySim Dataset** - Synthetic financial transaction data
- **NetworkX Community** - Graph analysis algorithms
- **Scikit-learn Team** - Machine learning framework
- **Streamlit** - Interactive web application framework

### 🌟 Special Thanks

- Contributors who helped improve the codebase
- Open source community for amazing tools and libraries
- Financial technology researchers for fraud detection insights

---

<div align="center">

### 🌟 Star this repository if you found it helpful!

**Built with ❤️ by [AjaySulya](https://github.com/AjaySulya)**

[⬆️ Back to Top](#paysim-fraud-detection-)

</div>

---

*Last updated: August 30, 2025*
