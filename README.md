# Clean Room Data Processor

A proof-of-concept data cleaning and standardization tool that uses Microsoft Phi-3.5 Mini to intelligently identify similar columns across multiple datasets and suggest standardization rules.

## 🎯 Overview

This tool demonstrates how to build a data integration application that:
- **Uploads raw files** from multiple sources
- **Identifies similar columns** across datasets using embeddings and LLM analysis
- **Suggests standardization rules** for data mapping and transformation
- **Validates data quality** and detects anomalies
- **Generates comprehensive reports** for data cleaning workflows

## 🚀 Quick Start

### Prerequisites
- macOS (tested on M1/M2/M3/M4, works on Intel)
- Python 3.8+
- Homebrew (for Ollama installation)

### Installation

1. **Install Ollama and Phi-3.5 Mini:**
```bash
brew install ollama
brew services start ollama
ollama pull phi3
```

2. **Set up Python environment:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run streamlit_app.py
```

4. **Access the application:**
- Open http://localhost:8501 in your browser
- Follow the 6-step workflow in the sidebar

## 📁 Project Structure

```
DCR_PoC_V0/
├── streamlit_app.py              # Main Streamlit application
├── phi_llm_functions.py          # Core Phi-3.5 Mini integration
├── requirements.txt               # Python dependencies
├── README.md                     # This file
├── QUICK_START.md                # 5-minute setup guide
├── tests/                        # Test suite
│   ├── __init__.py              # Python package init
│   ├── README.md                # Test documentation
│   ├── run_tests.py             # Test runner script
│   ├── test_phi_functions.py    # Phi LLM functions tests
│   ├── test_streamlit_app.py    # Streamlit app tests
│   └── validation_test.py       # Comprehensive validation tests
└── sample_*.csv                 # Sample test datasets
```

## 🔧 Core Features

### 1. Multi-file Upload
- Support for CSV, Excel (.xlsx, .xls), and JSON files
- Excel sheet selection for multi-sheet workbooks
- Real-time file processing with progress indicators

### 2. Intelligent Column Analysis
- Phi-3.5 Mini integration for intelligent analysis
- Embedding-based similarity detection using sentence transformers
- Cross-dataset column matching with similarity thresholds
- Clustering algorithm to group related columns

### 3. Human-in-the-Loop Workflow
- Interactive column mapping interface
- Sample data preview for context
- AI-powered transformation suggestions
- User approval workflow (Apply/Modify/Skip)

### 4. Data Quality Validation
- Comprehensive data quality assessment
- Anomaly detection using Phi-3.5 Mini
- Data completeness and consistency checks
- Quality scoring and recommendations

### 5. Multi-format Export
- Support for Excel, CSV, and JSON formats
- Split or combine datasets options
- Metadata inclusion capabilities
- Compression options for large files

## 📊 Example Workflow

### Input Data
```
customers.csv: customer_id, first_name, last_name, email
orders.csv: CustomerID, FirstName, LastName, EmailAddress
products.csv: product_id, product_name, category, price
```

### Analysis Results
```
Similarity Clusters:
├── Cluster 1: [customer_id, CustomerID]
├── Cluster 2: [first_name, FirstName]
├── Cluster 3: [last_name, LastName]
└── Cluster 4: [email, EmailAddress]
```

### User Feedback
```
Cluster 1: ✅ Related
Standard Name: customer_id
Notes: Standardize to lowercase with underscore
```

### Output
```
cleaned_customers.csv: customer_id, first_name, last_name, email
cleaned_orders.csv: customer_id, first_name, last_name, email
cleaned_products.csv: product_id, product_name, category, price
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python tests/run_tests.py

# Run individual tests
python tests/test_phi_functions.py
python tests/test_streamlit_app.py
python tests/validation_test.py
```

## 🔧 Troubleshooting

### Common Issues

1. **Ollama not running:**
```bash
brew services start ollama
```

2. **Phi-3.5 Mini not found:**
```bash
ollama pull phi3
```

3. **Port already in use:**
```bash
streamlit run streamlit_app.py --server.port 8502
```

4. **Memory issues:**
- Close other applications
- Restart Ollama: `brew services restart ollama`

## 📈 Performance

### System Requirements
- **Minimum**: 8GB RAM, 4GB free disk space
- **Recommended**: 16GB RAM for optimal performance
- **Apple Silicon**: Excellent performance with native ARM optimization

### Supported Workloads
- **File sizes**: Up to 100MB per file
- **Dataset count**: Up to 10 datasets simultaneously
- **Response time**: < 5 seconds for typical analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is for educational and PoC purposes.

## 🙏 Acknowledgments

- Microsoft for Phi-3.5 Mini
- Ollama for local LLM deployment
- Sentence Transformers for embeddings
- Streamlit for the web interface

---

**Ready to clean your data?** 🧹✨

Start with `streamlit run streamlit_app.py` and follow the workflow! 