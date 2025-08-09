# 🚀 Quick Start Guide

## ⚡ **Get Started in 5 Minutes**

### **1. Prerequisites Check**
```bash
# Check if you have an OpenAI API key
# You'll need to add it to llm_keys.yaml
```

### **2. Setup API Key**
```bash
# Edit llm_keys.yaml and add your OpenAI API key
open_ai: "your-openai-api-key-here"
```

### **3. Setup Python Environment**
```bash
# Navigate to project directory
cd DCR_PoC_V0

# Activate virtual environment
source venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### **4. Run the Application**
```bash
# Start Streamlit app
streamlit run streamlit_app.py
```

### **5. Access the Application**
- **URL**: http://localhost:8501
- **Browser**: Open your web browser and navigate to the URL

## 📋 **Quick Workflow Guide**

### **Step 1: Upload Files** 📁
1. Click "📁 Upload Files" in the sidebar
2. Drag and drop your CSV/Excel files
3. For Excel files with multiple sheets, select which sheets to import
4. View the dataset summary

### **Step 2: Run Analysis** 🔍
1. Click "🔍 Column Analysis" in the sidebar
2. Click "🚀 Run Column Analysis"
3. Wait for the analysis to complete
4. Review the similarity clusters found

### **Step 3: Provide Feedback** 🗺️
1. Click "🗺️ Column Mapping" in the sidebar
2. For each cluster of similar columns:
   - ✅ Check if they're related
   - 📝 Suggest a standard name
   - 📋 Add notes if needed
3. Review sample data for context

### **Step 4: Select Columns** 📋
1. Click "📋 Column Selection" in the sidebar
2. Select columns you want to clean from each dataset
3. Related columns will be auto-suggested
4. View data preview for selected columns

### **Step 5: Clean Data** 🧹
1. Click "🧹 Data Cleaning" in the sidebar
2. For each selected column:
   - Review AI suggestions for cleaning rules
   - Click "✅ Apply" to accept suggestions
   - Click "✏️ Modify" to edit rules
   - Click "⏭️ Skip" to bypass

### **Step 6: Download Results** 📥
1. Click "📥 Download Data" in the sidebar
2. Choose export format (Excel, CSV, JSON)
3. Configure export options
4. Download your cleaned data

## 🎯 **Example with Sample Data**

### **Upload Sample Files**
The project includes sample data files:
- `sample_customers.csv`
- `sample_orders.csv`
- `sample_products.csv`

### **Expected Results**
```
Analysis will find:
├── Cluster 1: [customer_id, CustomerID]
├── Cluster 2: [first_name, FirstName]
├── Cluster 3: [last_name, LastName]
└── Cluster 4: [email, EmailAddress]
```

### **Sample Workflow**
1. **Upload** the three sample CSV files
2. **Run analysis** to detect similar columns
3. **Confirm relationships** between customer_id/CustomerID, etc.
4. **Select columns** for cleaning (e.g., customer_id, first_name)
5. **Apply cleaning rules** suggested by AI
6. **Download** standardized data

## 🔧 **Troubleshooting**

### **Common Issues**

**1. Ollama not running:**
```bash
brew services start ollama
```

**2. Phi-3.5 Mini not found:**
```bash
ollama pull phi3
```

**3. Port already in use:**
```bash
streamlit run streamlit_app.py --server.port 8502
```

**4. Memory issues:**
- Close other applications
- Restart Ollama: `brew services restart ollama`

### **Performance Tips**
- 🚀 **Start small**: Test with < 1000 rows first
- 💾 **Close other apps**: Free up memory
- 🔄 **Restart if slow**: `brew services restart ollama`
- 📊 **Process one at a time**: For large datasets

## 📊 **Status Indicators**

The sidebar shows real-time status:
- ✅ **Datasets loaded** (count)
- ✅ **Analysis complete**
- ✅ **Columns selected** (count)
- ✅ **Transformations applied** (count)

## 🎛️ **Sidebar Actions**

- **💾 Save Feedback**: Export user preferences
- **🔄 Reset All**: Clear all data and start over

## 📈 **Next Steps**

After getting familiar with the basic workflow:

1. **Read the full documentation**: `README.md`
2. **Explore advanced features**: Multi-sheet Excel processing
3. **Test with your own data**: Upload your CSV/Excel files
4. **Customize transformations**: Modify AI suggestions as needed

---

**🎉 Ready to clean your data!**

Start with `streamlit run streamlit_app.py` and follow the 6-step workflow in the sidebar. 