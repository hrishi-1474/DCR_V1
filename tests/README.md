# 🧪 Tests

This directory contains the test suite for the Clean Room Data Processor.

## 📁 Test Files

- **`test_phi_functions.py`** - Tests for Phi LLM integration functions
- **`test_streamlit_app.py`** - Tests for Streamlit application components  
- **`validation_test.py`** - Comprehensive validation tests for data agnosticism
- **`run_tests.py`** - Main test runner that executes all tests

## 🚀 Running Tests

### **Run All Tests**
```bash
# From the project root directory
python tests/run_tests.py
```

### **Run Individual Tests**
```bash
# From the project root directory
python tests/test_phi_functions.py
python tests/test_streamlit_app.py
python tests/validation_test.py
```

## 🧪 Test Categories

### **1. Phi LLM Functions Test**
Tests the core Phi-3.5 Mini integration:
- Basic Phi-3.5 Mini connection
- Column name analysis and similarity detection
- Data quality validation and anomaly detection
- Standardization suggestions and mapping rules

### **2. Streamlit App Test**
Tests the Streamlit application components:
- Phi processor initialization
- Feedback storage and loading
- Data processing functions
- Export functionality

### **3. Validation Test**
Tests data agnosticism and robustness:
- Works with different data types and domains
- Handles empty data, null values, special characters
- Processes multiple datasets simultaneously
- Scalable for larger datasets

## 📊 Expected Results

### **All Tests Should Pass**
```
📊 Test Results: 3/3 tests passed
🎉 All tests passed! The system is ready for use.
```

## 🔧 Prerequisites

- Python 3.8+
- Virtual environment activated
- Ollama service running
- Phi-3.5 Mini model installed

### **Setup**
```bash
# Activate virtual environment
source venv/bin/activate

# Start Ollama service
brew services start ollama

# Verify Phi-3.5 Mini
ollama list | grep phi3
```

## 🐛 Troubleshooting

### **Common Issues**

1. **Import Errors**
   ```bash
   # Make sure you're running from the project root
   cd /path/to/DCR_PoC_V0
   python tests/run_tests.py
   ```

2. **Ollama Connection Issues**
   ```bash
   # Check if Ollama is running
   brew services list | grep ollama
   
   # Start Ollama if needed
   brew services start ollama
   ```

3. **Phi-3.5 Mini Not Found**
   ```bash
   # Install Phi-3.5 Mini
   ollama pull phi3
   ```

4. **Memory Issues**
   - Close other applications
   - Restart Ollama service
   - Use smaller test datasets

## 📈 Performance

- **Small datasets**: Tests complete in < 30 seconds
- **Large datasets**: Tests complete in < 2 minutes
- **Memory usage**: 2-4GB RAM during testing

---

**Ready to test?** Run `python tests/run_tests.py` from the project root! 