# Data Clean Room Processor

A sophisticated data cleaning and standardization tool that uses OpenAI's GPT-4o-mini to intelligently identify similar columns across multiple datasets and provide iterative data value standardization with human-in-the-loop refinement.

## 🎯 Overview

This tool demonstrates how to build a data integration application that:
- **Uploads multiple data files** from various sources (CSV, Excel)
- **Identifies similar column groups** using density-based clustering with semantic embeddings
- **Generates intelligent mappings** for data value standardization
- **Provides iterative refinement** with human feedback loop
- **Exports cleaned data** with standardized values in multiple formats

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. **Set up Python environment:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Configure OpenAI API key:**
```bash
# Create llm_keys.yaml file
echo "open_ai: your_openai_api_key_here" > llm_keys.yaml
```

3. **Run the application:**
```bash
streamlit run streamlit_app.py
```

4. **Access the application:**
- Open http://localhost:8501 in your browser
- Follow the 3-step workflow: Upload → Column Analysis → Data Standardization

## 📁 Project Structure

```
DCR_PoC_V0/
├── streamlit_app.py              # Main Streamlit application
├── llm_keys.yaml                 # OpenAI API key configuration
├── PepsiCo_logo.png              # Company logo
├── requirements.txt               # Python dependencies
├── README.md                     # This file
├── QUICK_START.md                # Quick setup guide
├── CHANGELOG.md                  # Version history
├── CONTRIBUTING.md               # Contribution guidelines
├── .gitignore                    # Git ignore patterns
└── sample_*.csv                  # Sample test datasets
```

## 🔧 Core Features

### 1. Intelligent File Upload
- Support for CSV and Excel files (.xlsx, .xls)
- Multi-sheet Excel support with sheet selection
- Real-time file processing with progress indicators
- File validation and error handling

### 2. Advanced Column Analysis
- **Density-based clustering** using sentence embeddings for semantic similarity
- **Cross-dataset column matching** with DBSCAN algorithm
- **Interactive column group customization** with multi-select dropdowns
- **Custom column group creation** for user-defined relationships

### 3. AI-Powered Data Standardization
- **OpenAI GPT-4o-mini integration** for intelligent data value mapping
- **Parallel processing** for multiple column groups simultaneously
- **Brand name standardization** with classification and grouping
- **Real-time progress tracking** for LLM operations

### 4. Human-in-the-Loop Refinement
- **Interactive mapping review** with editable data tables
- **Iterative feedback system** for continuous improvement
- **Inline editing** for mapping corrections
- **Multi-iteration refinement** until satisfactory results

### 5. Comprehensive Export Options
- **Final mappings Excel** (one sheet per column group)
- **Cleaned data with mappings** (original data + standardized columns)
- **Standardization impact metrics** showing reduction in unique values
- **Download buttons** with appropriate emojis for easy identification

## 📊 Workflow Example

### Step 1: Upload Data Files
```
customers.csv: customer_id, brand_preference, region
orders.csv: CustomerID, preferred_brand, location
products.csv: product_id, brand_name, category
```

### Step 2: Column Analysis Results
```
Column Groups Found:
├── Column Group 1: [customer_id, CustomerID]
├── Column Group 2: [brand_preference, preferred_brand, brand_name]
└── Column Group 3: [region, location]
```
<code_block_to_apply_changes_from>
```
Column Group 2 Mappings:
COCA-COLA → Coca Cola
coca cola → Coca Cola
Coke → Coca Cola
PEPSI → Pepsi
pepsi cola → Pepsi
```

### Final Output
```
Standardization Impact:
Column Group 2: 156 → 12 unique values (92.3% reduction)
Overall Impact: 203 → 45 unique values (77.8% reduction)
```

## 🎨 User Interface

### Design Features
- **Clean white theme** with gray/silver interactive elements
- **PepsiCo branding** with corporate logo
- **Responsive layout** with proper spacing and alignment
- **Interactive tables** for data review and editing
- **Progress indicators** for long-running operations
- **Status tracking** in sidebar for workflow progress

### Navigation
- **Upload Files**: Multi-file upload with sheet selection
- **Column Analysis**: Automated clustering with customization options
- **Data Value Standardizer**: Iterative refinement interface
- **Final Output**: Clean results page with download options

## 🔧 Configuration

### API Keys
Store your OpenAI API key in `llm_keys.yaml`:
```yaml
open_ai: your_openai_api_key_here
```

### Customization
- **Clustering parameters**: Adjust eps and min_samples in `density_based_column_clustering()`
- **LLM model**: Currently uses `gpt-4o-mini` (configurable in code)
- **Parallel processing**: Max 5 concurrent workers (adjustable)

## 📈 Performance

### System Requirements
- **Minimum**: 4GB RAM, 2GB free disk space
- **Recommended**: 8GB RAM for optimal performance
- **Network**: Stable internet connection for OpenAI API calls

### Supported Workloads
- **File sizes**: Up to 200MB per file
- **Dataset count**: Up to 10 datasets simultaneously
- **Column groups**: Efficient parallel processing for multiple groups
- **Response time**: Variable based on API latency and data size

## 🔧 Troubleshooting

### Common Issues

1. **API key not found:**
```bash
# Ensure llm_keys.yaml exists with valid OpenAI API key
echo "open_ai: your_key_here" > llm_keys.yaml
```

2. **Port already in use:**
```bash
streamlit run streamlit_app.py --server.port 8502
```

3. **Memory issues with large files:**
- Process files in smaller batches
- Restart the application between large operations

4. **LLM timeout errors:**
- Check internet connection
- Verify OpenAI API key is valid and has credits

## 🚀 Recent Updates

- **Migration to OpenAI GPT-4o-mini** for improved accuracy and reliability
- **Parallel processing implementation** for faster mapping generation
- **Enhanced UI/UX** with clean white theme and PepsiCo branding
- **Improved clustering algorithm** using sentence embeddings
- **Iterative refinement workflow** for better human-AI collaboration
- **Comprehensive export options** with impact metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

See `CONTRIBUTING.md` for detailed guidelines.

## 📄 License

This project is for educational and PoC purposes.

## 🙏 Acknowledgments

- OpenAI for GPT-4o-mini API
- Sentence Transformers for semantic embeddings
- Streamlit for the web interface
- scikit-learn for clustering algorithms
- PepsiCo for design inspiration

---

**Ready to standardize your data?** 🧹✨

Start with `streamlit run streamlit_app.py` and follow the intuitive 3-step workflow! 