import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import re
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import random
import yaml
from langchain_openai import ChatOpenAI
import concurrent.futures
import threading
from functools import partial

# --- Load API key from YAML ---
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["open_ai"]
except Exception as e:
    st.error(f"Error loading API key: {e}")
    st.info("Please create a 'llm_keys.yaml' file with your OpenAI API key")

# --- Initialize LangChain LLM ---
try:
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, top_p=1)
except Exception as e:
    st.error(f"Error initializing LLM: {e}")
    llm = None

# Page configuration
st.set_page_config(
    page_title="Clean Room Data Processor",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean white theme with gray/silver buttons
st.markdown("""
<style>
/* Remove all custom backgrounds - use default white */
.stApp {
    background-color: white;
}

/* Fix white strip at top */
.stApp > header {
    background-color: white !important;
}

/* Main container background */
.main .block-container {
    background-color: white;
}

/* Sidebar background - keep default light gray */
.sidebar .sidebar-content {
    background-color: #f0f2f6;
}

/* Remove header background styling */
.main .block-container > div:first-child {
    background-color: transparent;
    padding: 0;
    border-radius: 0;
    margin-bottom: 0;
}

/* Change buttons to gray/silver */
.stButton > button {
    background-color: #d3d3d3 !important;
    color: #333 !important;
    border-radius: 5px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: 500;
}

.stButton > button:hover {
    background-color: #c0c0c0 !important;
    color: #333 !important;
}

/* Primary buttons - darker gray */
.stButton > button[data-baseweb="button"] {
    background-color: #a9a9a9 !important;
    color: white !important;
}

.stButton > button[data-baseweb="button"]:hover {
    background-color: #808080 !important;
    color: white !important;
}

/* Change dropdowns to gray/silver */
.stSelectbox > div > div {
    background-color: #d3d3d3 !important;
    border-color: #c0c0c0 !important;
    color: #333 !important;
}

.stSelectbox > div > div:hover {
    background-color: #c0c0c0 !important;
}

/* Multiselect dropdowns - no borders */
.stMultiSelect > div > div {
    background-color: #d3d3d3 !important;
    border: none !important;
    color: #333 !important;
    box-shadow: none !important;
}

.stMultiSelect > div > div:hover {
    background-color: #c0c0c0 !important;
    border: none !important;
    box-shadow: none !important;
}

/* Remove all multiselect container borders */
.stMultiSelect > div {
    border: none !important;
}

.stMultiSelect div[data-baseweb="select"] {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

.stMultiSelect div[data-baseweb="select"]:focus {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Remove borders from multiselect selected items */
.stMultiSelect div[data-baseweb="select"] > div {
    border: none !important;
}

/* Remove borders from dropdown arrow/control area */
.stMultiSelect [data-baseweb="select"] [role="button"] {
    border: none !important;
}

/* Radio buttons */
.stRadio > div > div {
    background-color: #d3d3d3 !important;
}

/* File uploader - gray/silver to match buttons */
.stFileUploader > div {
    background-color: #d3d3d3 !important;
    border-color: #c0c0c0 !important;
    color: #333 !important;
}

/* File uploader selected file display - match gray theme */
.stFileUploader > div > div > div > div {
    background-color: #d3d3d3 !important;
    border-color: #c0c0c0 !important;
    color: #333 !important;
}

/* File uploader file name display */
.stFileUploader [data-testid="stFileUploaderFileName"] {
    background-color: #d3d3d3 !important;
    color: #333 !important;
    border-radius: 5px !important;
    padding: 0.5rem !important;
}

/* File uploader file size display */
.stFileUploader [data-testid="stFileUploaderFileSize"] {
    background-color: #d3d3d3 !important;
    color: #333 !important;
}

/* File uploader selected files section */
.stFileUploader section {
    background-color: #d3d3d3 !important;
    border-color: #c0c0c0 !important;
    border-radius: 5px !important;
}

/* Target the specific uploaded file display container */
.stFileUploader [data-testid="stFileUploaderDropzone"] {
    background-color: #d3d3d3 !important;
    border-color: #c0c0c0 !important;
}

/* Target the file list container */
.stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] + div {
    background-color: #d3d3d3 !important;
}

/* Target individual file items in the upload area */
.stFileUploader div[data-testid] div {
    background-color: #d3d3d3 !important;
    color: #333 !important;
}

/* More specific targeting for uploaded file display */
.stFileUploader > div > div:last-child {
    background-color: #d3d3d3 !important;
}

/* Target the uploaded file container specifically */
.stFileUploader > div > div > div:has([data-testid="stFileUploaderFileName"]) {
    background-color: #d3d3d3 !important;
    border: 1px solid #c0c0c0 !important;
    border-radius: 5px !important;
}

/* Ensure file content area matches */
.stFileUploader [role="button"] {
    background-color: #d3d3d3 !important;
    border-color: #c0c0c0 !important;
}

/* Progress bars */
.stProgress > div > div > div {
    background-color: #a9a9a9 !important;
}

/* Spinner */
.stSpinner > div > div {
    border-color: #a9a9a9 !important;
    border-top-color: transparent !important;
}

/* Change success messages to light gray/silver */
.stSuccess {
    background-color: #f5f5f5 !important;
    border-color: #d3d3d3 !important;
    color: #333 !important;
}

.stSuccess > div {
    background-color: #f5f5f5 !important;
    color: #333 !important;
}

/* Ensure all containers have white background */
.stApp > div > div > div > div {
    background-color: white !important;
}

.stApp > div {
    background-color: white !important;
}

/* Default text color */
.main .block-container {
    color: inherit;
}

.sidebar .sidebar-content {
    color: inherit;
}
</style>
""", unsafe_allow_html=True)

# Add PepsiCo logo to top right using sidebar
with st.sidebar:
    st.image("PepsiCo_logo.png", width=180)
    st.markdown("---")

# --- Simple LLM Call Function ---

# --- LLM Call Function ---
def call_llm(prompt):
    if llm is None:
        return "Error: LLM not initialized"
    try:
        return llm.predict(prompt).strip()
    except Exception as e:
        return f"Error calling LLM: {e}"

def call_llm_parallel(prompt, group_name):
    """Call LLM with error handling and group name tracking for parallel processing."""
    try:
        result = call_llm(prompt)
        return group_name, result, None
    except Exception as e:
        return group_name, None, str(e)

def process_llm_response(output, group_name):
    """Process LLM response and extract JSON mapping."""
    try:
        import re
        cleaned_output = output.strip()
        json_match = re.search(r'\[.*\]', cleaned_output, re.DOTALL | re.MULTILINE)
        if json_match:
            parsed_output = json_match.group()
            json.loads(parsed_output)  # Validate
            return parsed_output
        else:
            return output
    except Exception as e:
        st.warning(f"Failed to parse response for {group_name}: {e}")
        return output



# --- Prompt Templates ---
def initial_prompt_template(data_values):
    return f"""
You are an expert in cleaning and deduplicating product brand names in retail data.

Below is a list of brand names extracted from different data sources. These names may vary due to typos, prefixes/suffixes (e.g., "U-", "C-", version numbers), formatting inconsistencies, or minor descriptive additions.

Your task is to:
1. Identify brand names that likely refer to the same brand.
2. Group such similar names together under a shared 'canonical' brand name.
3. The canonical name must be selected from the provided variants — ideally the most commonly used or recognizable form.

CRITICAL REQUIREMENTS:
- You MUST return exactly {len(data_values)} mappings (one for each input value).
- Every input brand name must appear exactly once in the output.
- Do not add extra mappings or skip any input values.
- Do not invent new brand names not in the original brand name list.
- Every provided brand name must be included under some group (even if standalone).


Return the output strictly in the following JSON format:

**output format:**
[
  {{
    "Brand name": "GATORADE 5V5",
    "classified_as": "GATORADE"
  }},
  ...
]

Brand names ({len(data_values)} total):
{data_values}
"""

def refinement_prompt_template(prev_classification, feedback):
    return f"""
You previously classified brand names as follows:
{prev_classification}

A human reviewer has suggested the following refinements:
{feedback}

Your task is to:
- Apply the human feedback carefully to improve classification.
- Update the brand name mappings based on the human feedback provided.
- you may also apply the same change (or a consistent pattern) to other brand names that are: Lexically similar or 
  Follow a similar naming pattern
- Ensure all brand names are still classified under a canonical name from the list.

CRITICAL REQUIREMENTS:
- You MUST return exactly the same number of mappings as the previous classification.
- Every input brand name must appear exactly once in the output.
- Do not add extra mappings or skip any input values.
- Do not invent new brand names not in the original list.

Return the output strictly in the following JSON format:

**output format:**
[
  {{
    "Brand name": "GATORADE 5V5",
    "classified_as": "GATORADE"
  }},
  ...
]
"""

# Initialize session state
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
# New session state variables to preserve outputs
if 'upload_output' not in st.session_state:
    st.session_state.upload_output = None
if 'analysis_output' not in st.session_state:
    st.session_state.analysis_output = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "📁 Upload Files"

# Add cleaning interface session state variables here
if 'cleaning_iteration' not in st.session_state:
    st.session_state.cleaning_iteration = 0
if 'data_values' not in st.session_state:
    st.session_state.data_values = []
if 'output_history' not in st.session_state:
    st.session_state.output_history = []
if 'latest_output' not in st.session_state:
    st.session_state.latest_output = ""
if 'cleaning_finished' not in st.session_state:
    st.session_state.cleaning_finished = False

if 'user_cluster_selections' not in st.session_state:
    st.session_state.user_cluster_selections = {}
if 'cluster_index' not in st.session_state:
    st.session_state.cluster_index = None
if 'cluster_columns' not in st.session_state:
    st.session_state.cluster_columns = []
if 'cluster_columns_info' not in st.session_state:
    st.session_state.cluster_columns_info = []
if 'inline_feedback' not in st.session_state:
    st.session_state.inline_feedback = {}

# Add new session state variables for multi-cluster mapping
if 'all_column_groups_mappings' not in st.session_state:
    st.session_state.all_column_groups_mappings = {}
if 'all_column_groups_feedback' not in st.session_state:
    st.session_state.all_column_groups_feedback = {}
if 'mappings_generated' not in st.session_state:
    st.session_state.mappings_generated = False

def initialize_llm_processor():
    """Initialize the LLM processor."""
    if llm is None:
        st.error("LLM not initialized. Please check your API key configuration.")
        return None
    return llm

def calculate_column_group_summary(cluster_columns, dataframes):
    """Calculate summary information for a column group."""
    all_unique_values = set()
    cluster_columns_info = []
    
    for col in cluster_columns:
        # Only process string columns
        if isinstance(col, str):
            for filename, df in dataframes.items():
                if col in df.columns:
                    if df[col].dtype == 'object' or df[col].dtype == 'string':
                        # Get unique values from this column and filter for strings only
                        col_values = df[col].dropna().tolist()
                        string_values = []
                        for val in col_values:
                            if isinstance(val, str):
                                string_values.append(val)
                        
                        # Get unique string values
                        col_unique_values = list(set(string_values))
                        all_unique_values.update(col_unique_values)
                        cluster_columns_info.append({
                            'filename': filename,
                            'column': col,
                            'unique_count': len(col_unique_values),
                            'total_count': df[col].count()
                        })
    
    unique_values = list(all_unique_values)
    
    # Get sample values (first 10 or all if less than 10) for display
    sample_values = unique_values[:10] if len(unique_values) > 10 else unique_values
    
    return {
        'total_unique_values': len(unique_values),
        'sample_values': sample_values,
        'all_unique_values': unique_values,  # Add this to return all unique values
        'columns_info': cluster_columns_info
    }

def get_columns_with_filenames(selected_columns, dataframes):
    """Get columns with their corresponding file names."""
    columns_with_files = []
    for col in selected_columns:
        # Only process string columns
        if isinstance(col, str):
            for filename, df in dataframes.items():
                if col in df.columns:
                    # Extract sheet name from filename if it contains " - "
                    if " - " in filename:
                        sheet_name = filename.split(" - ")[-1]
                        display_name = sheet_name
                    else:
                        display_name = filename
                    columns_with_files.append(f"{display_name}: {col}")
                    break
    return sorted(columns_with_files)

def generate_column_groups_summary_table(user_cluster_selections, custom_clusters, dataframes):
    """Generate comprehensive summary table for all column groups."""
    summary_data = []
    
    # Process auto-generated column groups
    for cluster_key, selected_columns in user_cluster_selections.items():
        if selected_columns and cluster_key.startswith('cluster_'):
            cluster_num = cluster_key.split('_')[1]
            group_name = f"Column Group {int(cluster_num)+1}"
            
            # Calculate summary for this group
            summary = calculate_column_group_summary(selected_columns, dataframes)
            
            # Get columns with file names
            columns_with_files = get_columns_with_filenames(selected_columns, dataframes)
            
            summary_data.append({
                'Column Group Name': group_name,
                'Columns in Group': ', '.join(columns_with_files),
                'Total Unique Values': summary['total_unique_values'],
                'Sample Values': ', '.join(str(val) for val in summary['sample_values']),
                'Additional Instructions/Feedback': ''
            })
    
    # Process custom column groups
    for i, custom_cluster in enumerate(custom_clusters):
        custom_cluster_key = f"custom_cluster_{i}"
        if custom_cluster_key in user_cluster_selections:
            selected_columns = user_cluster_selections[custom_cluster_key]
            if selected_columns:
                group_name = f"Custom Column Group {i+1}"
                
                # Calculate summary for this group
                summary = calculate_column_group_summary(selected_columns, dataframes)
                
                # Get columns with file names
                columns_with_files = get_columns_with_filenames(selected_columns, dataframes)
                
                summary_data.append({
                    'Column Group Name': group_name,
                    'Columns in Group': ', '.join(columns_with_files),
                    'Total Unique Values': summary['total_unique_values'],
                    'Sample Values': ', '.join(str(val) for val in summary['sample_values']),
                    'Additional Instructions/Feedback': ''
                })
    
    return summary_data


def generate_mappings_for_all_groups(user_cluster_selections, custom_clusters, dataframes):
    """Generate initial mappings for all column groups using parallel processing."""
    all_mappings = {}
    
    # Prepare all tasks for parallel processing
    tasks = []
    
    # Process auto-generated column groups
    for cluster_key, selected_columns in user_cluster_selections.items():
        if selected_columns and cluster_key.startswith('cluster_'):
            cluster_num = cluster_key.split('_')[1]
            group_name = f"Column Group {int(cluster_num)+1}"
            
            summary = calculate_column_group_summary(selected_columns, dataframes)
            unique_values = summary['all_unique_values']
            
            if unique_values:
                initial_prompt = initial_prompt_template(unique_values)
                tasks.append((group_name, initial_prompt))
    
    # Process custom column groups
    for i, custom_cluster in enumerate(custom_clusters):
        custom_cluster_key = f"custom_cluster_{i}"
        if custom_cluster_key in user_cluster_selections:
            selected_columns = user_cluster_selections[custom_cluster_key]
            if selected_columns:
                group_name = f"Custom Column Group {i+1}"
                
                summary = calculate_column_group_summary(selected_columns, dataframes)
                unique_values = summary['all_unique_values']
                
                if unique_values:
                    initial_prompt = initial_prompt_template(unique_values)
                    tasks.append((group_name, initial_prompt))
    
    # Execute LLM calls in parallel
    if tasks:
        with st.spinner(f"Generating mappings for {len(tasks)} column groups"):
            # Create a progress container for real-time updates
            progress_container = st.empty()
            completed_count = 0
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tasks), 5)) as executor:
                # Submit all tasks
                future_to_group = {
                    executor.submit(call_llm_parallel, prompt, group_name): group_name 
                    for group_name, prompt in tasks
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_group):
                    group_name = future_to_group[future]
                    try:
                        result_group_name, output, error = future.result()
                        completed_count += 1
                        
                        # Update progress message
                        progress_container.success(f"Generating mappings for {group_name} ({completed_count}/{len(tasks)})")
                        
                        if error:
                            st.error(f"Error processing {group_name}: {error}")
                            all_mappings[group_name] = f"Error: {error}"
                        else:
                            processed_output = process_llm_response(output, group_name)
                            all_mappings[group_name] = processed_output
                            
                    except Exception as e:
                        completed_count += 1
                        progress_container.error(f"❌ Failed {group_name} ({completed_count}/{len(tasks)})")
                        st.error(f"Unexpected error processing {group_name}: {e}")
                        all_mappings[group_name] = f"Error: {e}"
    
    return all_mappings

def process_feedback_for_all_groups(all_mappings, all_feedback):
    """Process feedback for all column groups using parallel processing."""
    refined_mappings = {}
    
    # Prepare tasks for parallel processing
    tasks = []
    
    for group_name, mapping_output in all_mappings.items():
        if group_name in all_feedback and all_feedback[group_name]:
            feedback_json = []
            try:
                parsed_mapping = json.loads(mapping_output)
                for feedback_item in all_feedback[group_name]:
                    if isinstance(feedback_item, dict) and 'Brand name' in feedback_item and 'classified_as' in feedback_item:
                        feedback_json.append({
                            "Brand name": feedback_item['Brand name'],
                            "classified_as": feedback_item['classified_as']
                        })
                
                if feedback_json:
                    refinement_prompt = refinement_prompt_template(mapping_output, feedback_json)
                    tasks.append((group_name, refinement_prompt))
                else:
                    refined_mappings[group_name] = mapping_output
            except:
                refined_mappings[group_name] = mapping_output
        else:
            refined_mappings[group_name] = mapping_output
    
    # Execute LLM calls in parallel
    if tasks:
        # Create a progress container for real-time updates
        progress_container = st.empty()
        completed_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tasks), 5)) as executor:
            future_to_group = {
                executor.submit(call_llm_parallel, prompt, group_name): group_name 
                for group_name, prompt in tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_group):
                group_name = future_to_group[future]
                try:
                    result_group_name, output, error = future.result()
                    completed_count += 1
                    
                    # Update progress message - use same format as generate_mappings
                    progress_container.success(f"✅ Completed {group_name} ({completed_count}/{len(tasks)})")
                    
                    if error:
                        st.error(f"Error processing feedback for {group_name}: {error}")
                        refined_mappings[group_name] = all_mappings[group_name]
                    else:
                        processed_output = process_llm_response(output, group_name)
                        refined_mappings[group_name] = processed_output
                        
                except Exception as e:
                    completed_count += 1
                    progress_container.error(f"❌ Failed {group_name} ({completed_count}/{len(tasks)})")
                    st.error(f"Unexpected error processing feedback for {group_name}: {e}")
                    refined_mappings[group_name] = all_mappings[group_name]
    
    return refined_mappings

def create_mappings_table(all_mappings):
    """Create a comprehensive table showing all mappings."""
    table_data = []
    
    for group_name, mapping_output in all_mappings.items():
        try:
            parsed_mapping = json.loads(mapping_output)
            for item in parsed_mapping:
                if isinstance(item, dict) and 'Brand name' in item and 'classified_as' in item:
                    table_data.append({
                        'Column Group Name': group_name,
                        'Brand Name': item['Brand name'],
                        'Classified As': item['classified_as'],
                        'Feedback': ''
                    })
        except:
            # If parsing fails, add a single row with the raw output
            table_data.append({
                'Column Group Name': group_name,
                'Brand Name': 'Parsing Error',
                'Classified As': mapping_output[:100] + '...' if len(mapping_output) > 100 else mapping_output,
                'Feedback': ''
            })
    
    return table_data


def upload_files():
    """Handle file upload with Excel sheet selection."""
    st.header("Upload Data Files")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload CSV or Excel files. For Excel files, you can select specific sheets."
    )
    
    if uploaded_files:
        dataframes = {}
        excel_sheets = {}
        
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            
            try:
                if file_name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    dataframes[file_name] = df
                    st.success(f"Successfully loaded {file_name}")
                
                elif file_name.endswith(('.xlsx', '.xls')):
                    # Get available sheets
                    excel_file = pd.ExcelFile(uploaded_file)
                    available_sheets = excel_file.sheet_names
                    print(f'available_sheets: {available_sheets}')
                    if len(available_sheets) > 1:
                        st.write(f"**{file_name}** has multiple sheets:")
                        selected_sheets = st.multiselect(
                            f"Select sheets to import from {file_name}",
                            sorted(available_sheets),
                            default=sorted(available_sheets)[:1],
                            key=f"sheets_{file_name}"
                        )
                        
                        for sheet_name in selected_sheets:
                            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                            full_name = f"{file_name} - {sheet_name}"
                            dataframes[full_name] = df
                            st.success(f"Loaded {full_name}: {df.shape[0]} rows, {df.shape[1]} columns")
                    else:
                        df = pd.read_excel(uploaded_file)
                        dataframes[file_name] = df
                        st.success(f"Loaded {file_name}: {df.shape[0]} rows, {df.shape[1]} columns")
            
            except Exception as e:
                st.error(f"Error loading {file_name}: {e}")
        
        if dataframes:
            st.session_state.dataframes = dataframes
            
            # Show dataset summary
            st.subheader("Dataset Summary")
            summary_data = []
            for filename, df in dataframes.items():
                summary_data.append({
                    'Dataset': filename,
                    'Rows': df.shape[0],
                    'Columns': df.shape[1],
                    'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Store output for preservation
            st.session_state.upload_output = {
                'datasets_loaded': len(dataframes),
                'summary': summary_data,
                'filenames': list(dataframes.keys())
            }
            
            # Navigation button
            st.divider()
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Continue to Column Analysis", type="primary"):
                    analyze_columns()
                    st.session_state.current_page = "🔍 Column Analysis"
                    st.rerun()
            
            return True
    
    return False

def analyze_columns():
    """Perform automatic column analysis."""
    
    if not st.session_state.dataframes:
        st.warning("Please upload files first.")
        return False
    
    with st.spinner("Analyzing columns across all datasets..."):
        # Create a simplified report with only essential information
        report = {
            "dataset_summary": {},
            "similarity_clusters": [],
            "quality_assessment": {}
        }
        
        # Generate dataset summaries (fast, no LLM calls)
        for filename, df in st.session_state.dataframes.items():
            report["dataset_summary"][filename] = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict()
            }
            
            # Basic quality assessment (fast, no LLM calls)
            null_counts = df.isnull().sum().to_dict()
            unique_counts = df.nunique().to_dict()
            
            report["quality_assessment"][filename] = {
                "null_counts": null_counts,
                "unique_counts": unique_counts,
                "total_rows": len(df),
                "total_columns": len(df.columns)
            }
        
        # Find similar columns across all datasets using simple string similarity
        # Only consider string-type columns for similarity analysis
        all_columns = []
        string_columns = []
        
        for filename, df in st.session_state.dataframes.items():
            for col in df.columns:
                all_columns.append(col)
                # Check if column name is a string and column contains string data
                if isinstance(col, str) and (df[col].dtype == 'object' or df[col].dtype == 'string'):
                    # Additional check: sample some values to ensure they're actually strings
                    sample_values = df[col].dropna().head(10)
                    if len(sample_values) > 0 and all(isinstance(val, str) for val in sample_values):
                        string_columns.append(col)
        
        # Use density-based clustering instead of simple similarity
        clusters = density_based_column_clustering(st.session_state.dataframes)
        
        report["similarity_clusters"] = clusters
        
        # Store analysis results
        st.session_state.analysis_report = report
        st.session_state.analysis_complete = True
        
        
        # Show similarity clusters with interactive multi-select dropdowns
        if report.get('similarity_clusters'):
            st.write("**Tip:** Use the dropdowns below to customize your column groups. Select/deselect columns as needed.")
            
            for i, cluster in enumerate(report['similarity_clusters']):
                if len(cluster) > 1:  # Only show clusters with multiple columns
                    # Group columns by file
                    columns_by_file = {}
                    for col in cluster:
                        for filename, df in st.session_state.dataframes.items():
                            if col in df.columns:
                                if filename not in columns_by_file:
                                    columns_by_file[filename] = []
                                columns_by_file[filename].append(col)
                                break
                    
                    # Display cluster with interactive dropdowns
                    st.write(f"**Cluster {i+1}:**")
                    
                    # Initialize user selections for this cluster if not exists
                    cluster_key = f"cluster_{i}"
                    if cluster_key not in st.session_state.user_cluster_selections:
                        st.session_state.user_cluster_selections[cluster_key] = cluster.copy()
                    
                    # Create multi-select dropdowns for each file
                    for filename, cols in columns_by_file.items():
                        # Get all available columns from this file
                        all_file_columns = st.session_state.dataframes[filename].columns.tolist()
                        
                        # Get current selections for this file in this cluster
                        current_selections = [col for col in cols if col in st.session_state.user_cluster_selections[cluster_key]]
                        
                        # Multi-select dropdown
                        selected_columns = st.multiselect(
                            f"**{filename}** columns:",
                            options=sorted([col for col in all_file_columns if isinstance(col, str)]),
                            default=current_selections,
                            key=f"cluster_{i}_{filename}",
                            help=f"Select columns from {filename} that should be in this cluster"
                        )
                        
                        # Update user selections
                        # Remove old selections for this file
                        st.session_state.user_cluster_selections[cluster_key] = [
                            col for col in st.session_state.user_cluster_selections[cluster_key] 
                            if col not in all_file_columns
                        ]
                        # Add new selections
                        st.session_state.user_cluster_selections[cluster_key].extend(selected_columns)
                    
                    # Show current cluster summary
                    current_cluster = st.session_state.user_cluster_selections[cluster_key]
                    if current_cluster:
                        st.write(f"**Current Cluster {i+1}:** {', '.join(sorted(current_cluster))}")
                    else:
                        st.write(f"**Cluster {i+1}:** No columns selected")
                    
                    st.divider()
            
            # Custom cluster creation section

            
            # Show summary of customized clusters (only if there are any clusters)
            has_user_clusters = any(selected_columns for selected_columns in st.session_state.user_cluster_selections.values())
            
            if has_user_clusters:
                st.subheader("Customized Clusters Summary")
                st.write("**Your customized column clusters:**")
                
                total_customized_columns = 0
                
                # Show user clusters
                for cluster_key, selected_columns in st.session_state.user_cluster_selections.items():
                    if selected_columns:
                        cluster_num = cluster_key.split('_')[1]
                        st.write(f"**Cluster {int(cluster_num)+1}:** {', '.join(sorted(selected_columns))}")
                        total_customized_columns += len(selected_columns)
                
                st.write(f"**Total columns in all clusters:** {total_customized_columns}")
                
                # Add a button to reset customizations
                if st.button("Reset All Customizations"):
                    st.session_state.user_cluster_selections = {}
                    st.success("All customizations reset! Run analysis again to see original clusters.")
        

        
        # Store output for preservation
        st.session_state.analysis_output = {
            'clusters_found': len([c for c in report.get('similarity_clusters', []) if len(c) > 1]),
            'datasets_analyzed': len(report['dataset_summary']),
            'total_columns': sum(len(summary['columns']) for summary in report['dataset_summary'].values())
        }
        
        return True





def calculate_standardization_stats(mapping_output):
    """Calculate standardization statistics from mapping output."""
    try:
        parsed_mapping = json.loads(mapping_output)
        if isinstance(parsed_mapping, list):
            # Count original unique values
            original_values = set()
            # Count unique standardized values
            standardized_values = set()
            
            for item in parsed_mapping:
                if isinstance(item, dict) and 'Brand name' in item and 'classified_as' in item:
                    original_values.add(item['Brand name'])
                    standardized_values.add(item['classified_as'])
            
            return {
                'original_count': len(original_values),
                'standardized_count': len(standardized_values),
                'reduction': len(original_values) - len(standardized_values),
                'reduction_percentage': round(((len(original_values) - len(standardized_values)) / len(original_values)) * 100, 1) if len(original_values) > 0 else 0
            }
    except:
        return None
    return None

def dedicated_data_cleaning_interface():
    """Dedicated data cleaning interface with comprehensive mappings table and iterative refinement."""
    st.header("Data Value Standardizer")
    # Check if dataframes are available
    if not st.session_state.dataframes:
        st.error("No data uploaded. Please upload files first.")
        st.info("Go to 'Upload Files' to upload your datasets.")
        return
    
    # Check if mappings have been generated
    if not st.session_state.mappings_generated or not st.session_state.all_column_groups_mappings:
        st.warning("Please generate initial mappings first in the Column Analysis page.")
        st.info("Go to 'Column Analysis' to generate mappings for all column groups.")
        return
    
    # Step 1: Display all mappings and allow feedback
    if st.session_state.cleaning_iteration == 0:
        st.subheader("Iteration 0: Review and provide feedback for all column group mappings")
        
        # Display all generated mappings
        if st.session_state.all_column_groups_mappings:
            st.success(f"{len(st.session_state.all_column_groups_mappings)} column group(s) have mappings ready for review.")
            
            # Store feedback
            st.session_state.all_column_groups_feedback = {}
            
            # Display separate table for each column group
            for group_name, mapping_output in st.session_state.all_column_groups_mappings.items():
                st.markdown(f"### {group_name}")
                
                try:
                    parsed_mapping = json.loads(mapping_output)
                    
                    # Create table data for this specific group
                    group_table_data = []
                    for item in parsed_mapping:
                        if isinstance(item, dict) and 'Brand name' in item and 'classified_as' in item:
                            group_table_data.append({
                                'Brand Name': item['Brand name'],
                                'Classified As': item['classified_as'],
                                'Feedback': ''
                            })
                    
                    if group_table_data:
                        group_df = pd.DataFrame(group_table_data)
                        
                        # Display editable table for this group
                        edited_group_df = st.data_editor(
                            group_df,
                            use_container_width=True,
                            height=300,
                            key=f"group_table_{group_name}_{st.session_state.cleaning_iteration}",
                            column_config={
                                "Brand Name": st.column_config.TextColumn("Brand Name", disabled=True),
                                "Classified As": st.column_config.TextColumn("Classified As", disabled=True),
                                "Feedback": st.column_config.TextColumn("Feedback", help="Enter corrected classification here")
                            }
                        )
                        
                        # Calculate and display standardization stats
                        stats = calculate_standardization_stats(mapping_output)
                        if stats:
                            st.success(f"**Standardization Impact:** Reduced unique values from {stats['original_count']} to {stats['standardized_count']} ({stats['reduction']} fewer, {stats['reduction_percentage']}% reduction)")
                        
                        # Store feedback for this group
                        for idx, row in edited_group_df.iterrows():
                            feedback = row['Feedback']
                            if feedback.strip():
                                if group_name not in st.session_state.all_column_groups_feedback:
                                    st.session_state.all_column_groups_feedback[group_name] = []
                                st.session_state.all_column_groups_feedback[group_name].append({
                                    "Brand name": row['Brand Name'],
                                    "classified_as": feedback.strip()
                                })
                        
                        st.write(f"*{len(group_table_data)} mappings in {group_name}*")
                    else:
                        st.warning(f"No valid mappings found for {group_name}")
                        
                except Exception as e:
                    st.error(f"Error parsing mappings for {group_name}: {e}")
                    st.code(mapping_output[:200] + "..." if len(mapping_output) > 200 else mapping_output)
                
                st.divider()
            
            # Process feedback button for all groups
            st.divider()
            col1, col2 = st.columns([1, 1])

            with col1:
                if st.button("Process Feedback for All Groups", type="primary"):
                    with st.spinner("Processing feedback for all column groups..."):
                        refined_mappings = process_feedback_for_all_groups(
                            st.session_state.all_column_groups_mappings,
                            st.session_state.all_column_groups_feedback
                        )
                        st.session_state.all_column_groups_mappings = refined_mappings
                        st.session_state.cleaning_iteration = 1
                        st.success("Feedback processed for all column groups! Starting iterative refinement...")
                        st.rerun()

            with col2:
                if st.button("Apply & Finish", type="primary"):
                    st.session_state.cleaning_finished = True
                    st.success("All mappings finalized! Generating output files...")
                    st.rerun()
        else:
            st.error("No mappings available. Please generate mappings in the Column Analysis page.")
    
    # Step 2+: Iterative Refinement Loop (for all column groups)
    elif not st.session_state.cleaning_finished:
        st.subheader(f"Iteration {st.session_state.cleaning_iteration}")
        st.write("**Iterative refinement for all column groups:**")
        
        st.markdown("### Current Mappings with Inline Feedback")
        
        # Store feedback for refinement
        refinement_feedback = {}

        # Display separate table for each column group
        for group_name, mapping_output in st.session_state.all_column_groups_mappings.items():
            st.markdown(f"### {group_name}")
            
            try:
                parsed_mapping = json.loads(mapping_output)
                
                # Create table data for this specific group
                group_table_data = []
                for item in parsed_mapping:
                    if isinstance(item, dict) and 'Brand name' in item and 'classified_as' in item:
                        group_table_data.append({
                            'Brand Name': item['Brand name'],
                            'Classified As': item['classified_as'],
                            'Feedback': ''
                        })
                
                if group_table_data:
                    group_df = pd.DataFrame(group_table_data)
                    
                    # Display editable table for this group
                    edited_group_df = st.data_editor(
                        group_df,
                        use_container_width=True,
                        height=300,
                        key=f"refinement_group_table_{group_name}_{st.session_state.cleaning_iteration}",
                        column_config={
                            "Brand Name": st.column_config.TextColumn("Brand Name", disabled=True),
                            "Classified As": st.column_config.TextColumn("Classified As", disabled=True),
                            "Feedback": st.column_config.TextColumn("Feedback", help="Enter corrected classification here")
                        }
                    )
                    
                    # Calculate and display standardization stats
                    stats = calculate_standardization_stats(mapping_output)
                    if stats:
                        st.success(f"**Standardization Impact:** Reduced unique values from {stats['original_count']} to {stats['standardized_count']} ({stats['reduction']} fewer, {stats['reduction_percentage']}% reduction)")
                    
                    # Store feedback for this group
                    for idx, row in edited_group_df.iterrows():
                        feedback = row['Feedback']
                        if feedback.strip():
                            if group_name not in refinement_feedback:
                                refinement_feedback[group_name] = []
                            refinement_feedback[group_name].append({
                                "Brand name": row['Brand Name'],
                                "classified_as": feedback.strip()
                            })
                    
                    st.write(f"*{len(group_table_data)} mappings in {group_name}*")
                else:
                    st.warning(f"No valid mappings found for {group_name}")
                    
            except Exception as e:
                st.error(f"Error parsing mappings for {group_name}: {e}")
                st.code(mapping_output[:200] + "..." if len(mapping_output) > 200 else mapping_output)
                
                st.divider()

        # Refinement action buttons
        st.divider()
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Process Feedback for All Groups", key=f"process_refinement_{st.session_state.cleaning_iteration}"):
                if any(refinement_feedback.values()):
                    with st.spinner("Processing refinement feedback for all column groups..."):
                        refined_mappings = process_feedback_for_all_groups(
                            st.session_state.all_column_groups_mappings,
                            refinement_feedback
                        )
                        st.session_state.all_column_groups_mappings = refined_mappings
                        st.session_state.cleaning_iteration += 1
                        st.success("Refinement feedback processed! Starting next iteration...")
                        st.rerun()
                else:
                    st.warning("No refinement feedback provided. Please enter corrections in the Feedback column.")

        with col2:
            if st.button("Apply & Finish", key=f"apply_finish_{st.session_state.cleaning_iteration}"):
                st.session_state.cleaning_finished = True
                st.success("All mappings finalized! Generating output files...")
                st.rerun()
    
    # Final Output Display
    if st.session_state.cleaning_finished:
        st.subheader("Final Output:")
        st.write("All column groups have been processed successfully!")
        
        # Generate two specific Excel files
        try:
            # File A: Excel with only final mappings (one sheet per column group)
            with pd.ExcelWriter("final_mappings_only.xlsx", engine='openpyxl') as writer:
                for group_name, mapping_output in st.session_state.all_column_groups_mappings.items():
                    try:
                        parsed_mapping = json.loads(mapping_output)
                        
                        # Create mapping dataframe for this group
                        mapping_data = []
                        for item in parsed_mapping:
                            if isinstance(item, dict) and 'Brand name' in item and 'classified_as' in item:
                                mapping_data.append({
                                    'Original Value': item['Brand name'],
                                    'Standardized Value': item['classified_as']
                                })
                        
                        if mapping_data:
                            mapping_df = pd.DataFrame(mapping_data)
                            # Clean sheet name for Excel
                            safe_sheet_name = group_name.replace(' ', '_').replace(':', '_')[:31]
                            mapping_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
                        
                    except Exception as e:
                        st.error(f"Error processing mappings for {group_name}: {e}")
            
            # Download File A
            with open("final_mappings_only.xlsx", "rb") as file:
                st.download_button(
                    "📊 Download Final Mappings Excel",
                    data=file.read(),
                    file_name="final_mappings_only.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # File B: Excel with all original data plus new mapping columns
            with pd.ExcelWriter("cleaned_data_with_mappings.xlsx", engine='openpyxl') as writer:
                for filename, df in st.session_state.dataframes.items():
                    # Create a copy of the dataframe
                    df_with_mappings = df.copy()
                    
                    # Add mapping columns for all column groups
                    for group_name, mapping_output in st.session_state.all_column_groups_mappings.items():
                        try:
                            parsed_mapping = json.loads(mapping_output)
                            
                            # Find corresponding cluster columns for this group
                            cluster_columns = []
                            for cluster_key, selected_columns in st.session_state.user_cluster_selections.items():
                                if cluster_key.startswith('cluster_'):
                                    cluster_num = cluster_key.split('_')[1]
                                    current_group_name = f"Column Group {int(cluster_num)+1}"
                                    if current_group_name == group_name:
                                        cluster_columns = selected_columns
                                        break
                            
                            # If not found in auto-generated, check custom clusters
                            if not cluster_columns:
                                for i, custom_cluster in enumerate(st.session_state.custom_clusters):
                                    custom_cluster_key = f"custom_cluster_{i}"
                                    if custom_cluster_key in st.session_state.user_cluster_selections:
                                        current_group_name = f"Custom Column Group {i+1}"
                                        if current_group_name == group_name:
                                            cluster_columns = st.session_state.user_cluster_selections[custom_cluster_key]
                                            break
                            
                            # Create value mapping dictionary
                            value_mapping = {}
                            for item in parsed_mapping:
                                if isinstance(item, dict) and 'Brand name' in item and 'classified_as' in item:
                                    value_mapping[item['Brand name']] = item['classified_as']
                            
                            # Add mapping columns for each column in this group
                            for col in cluster_columns:
                                if col in df.columns:
                                    # Create new column name
                                    new_col_name = f"{col}_standardized"
                                    
                                    # Apply mapping to create new column
                                    df_with_mappings[new_col_name] = df[col].map(value_mapping).fillna(df[col])
                        
                        except Exception as e:
                            st.error(f"Error processing mappings for {group_name}: {e}")
                    
                    # Write to Excel sheet
                    sheet_name = filename.replace('.csv', '').replace('.xlsx', '').replace('.xls', '')[:31]
                    df_with_mappings.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Download File B
            with open("cleaned_data_with_mappings.xlsx", "rb") as file:
                st.download_button(
                    "📋 Download Cleaned Data with Mappings Excel",
                    data=file.read(),
                    file_name="cleaned_data_with_mappings.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            st.success("Two Excel files generated successfully!")
            
        except Exception as e:
            st.error(f"Error creating Excel files: {e}")
        
        # Navigation back to main app
        st.divider()
        if st.button("Start New Cleaning"):
            # Clear ALL session state variables
            st.session_state.clear()
            
            # Re-initialize essential session state variables
            st.session_state.dataframes = {}
            st.session_state.analysis_complete = False
            st.session_state.upload_output = None
            st.session_state.analysis_output = None
            st.session_state.current_page = "📁 Upload Files"
            st.session_state.cleaning_iteration = 0
            st.session_state.data_values = []
            st.session_state.output_history = []
            st.session_state.latest_output = ""
            st.session_state.cleaning_finished = False
            st.session_state.user_cluster_selections = {}
            st.session_state.cluster_index = None
            st.session_state.cluster_columns = []
            st.session_state.cluster_columns_info = []
            st.session_state.inline_feedback = {}
            st.session_state.all_column_groups_mappings = {}
            st.session_state.all_column_groups_feedback = {}
            st.session_state.mappings_generated = False
            
            # Navigate to Upload Files page
            st.session_state.current_page = "📁 Upload Files"
            st.rerun()
        
        if st.button("Back to Column Analysis"):
            st.session_state.current_page = "🔍 Column Analysis"
            st.rerun()



def apply_cleaning_mapping_to_cluster(cluster_columns, mapping_output):
    """Apply the cleaning mapping to all columns in a cluster."""
    try:
        # Parse the mapping
        if isinstance(mapping_output, str):
            mapping_data = json.loads(mapping_output)
        else:
            mapping_data = mapping_output
        
        if not isinstance(mapping_data, list):
            return
        
        # Create value mapping dictionary
        value_mapping = {}
        for item in mapping_data:
            if isinstance(item, dict) and 'Brand name' in item and 'classified_as' in item:
                value_mapping[item['Brand name']] = item['classified_as']
        
        # Apply mapping to all columns in the cluster
        columns_updated = 0
        for col in cluster_columns:
            for filename, df in st.session_state.dataframes.items():
                if col in df.columns:
                    # Apply mapping to this column
                    st.session_state.dataframes[filename][col] = st.session_state.dataframes[filename][col].map(value_mapping).fillna(st.session_state.dataframes[filename][col])
                    columns_updated += 1
                    break
        
        st.success(f"Mapping applied to {columns_updated} columns in the cluster")
        
    except Exception as e:
        st.error(f"Failed to apply mapping: {e}")

def apply_cleaning_mapping_to_specific_column(filename, column, mapping_output):
    """Apply the cleaning mapping to a specific column in a specific file."""
    try:
        # Parse the mapping
        if isinstance(mapping_output, str):
            mapping_data = json.loads(mapping_output)
        else:
            mapping_data = mapping_output
        
        if not isinstance(mapping_data, list):
            return
        
        # Create value mapping dictionary
        value_mapping = {}
        for item in mapping_data:
            if isinstance(item, dict) and 'Brand name' in item and 'classified_as' in item:
                value_mapping[item['Brand name']] = item['classified_as']
        
        # Apply mapping to specific file and column
        st.session_state.dataframes[filename][column] = st.session_state.dataframes[filename][column].map(value_mapping).fillna(st.session_state.dataframes[filename][column])
        
        st.success(f"Mapping applied to {filename} - {column}")
        
    except Exception as e:
        st.error(f"Failed to apply mapping: {e}")

def density_based_column_clustering(dataframes):
    """
    Perform density-based clustering on columns using both column names and sample values.
    Uses sentence embeddings for better semantic understanding.
    
    Args:
        dataframes: Dictionary of {filename: dataframe}
    
    Returns:
        List of clusters, where each cluster is a list of column names
    """
    # Step 1: Collect string columns and their sample values
    column_features = []
    column_names = []
    
    for filename, df in dataframes.items():
        for col in df.columns:
            # Only process string columns
            if isinstance(col, str) and (df[col].dtype == 'object' or df[col].dtype == 'string'):
                # Get sample values (10 random non-null string values)
                col_values = df[col].dropna()
                string_values = [str(val) for val in col_values if isinstance(val, str)]
                
                if len(string_values) > 0:
                    # Take 10 random samples (or all if less than 10)
                    sample_size = min(10, len(string_values))
                    sample_values = random.sample(string_values, sample_size)
                    
                    # Create feature text: column name + sample values
                    feature_text = f"Column: {col}. Sample values: {', '.join(sample_values)}"
                    
                    column_features.append(feature_text)
                    column_names.append(col)
    
    if len(column_features) < 2:
        return []
    
    # Step 2: Create sentence embeddings
    try:
        # Use a lightweight sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and effective
        
        # Create embeddings for all column features
        embeddings = model.encode(column_features, convert_to_tensor=False)
        
        # Step 3: Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Step 4: Convert similarity to distance matrix
        # Higher similarity = lower distance
        distance_matrix = 1 - similarity_matrix
        
        # Step 5: Apply DBSCAN clustering
        # eps: maximum distance between points in a cluster
        # min_samples: minimum number of points to form a cluster
        eps = 0.2  # Adjustable threshold for similarity (lower = stricter)
        min_samples = 2  # Minimum 2 columns to form a cluster
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        cluster_labels = dbscan.fit_predict(distance_matrix)
        
        # Step 6: Group columns by cluster labels
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(column_names[i])
        
        # Step 7: Return only clusters with multiple columns (exclude noise points with label -1)
        result_clusters = []
        for label, columns in clusters.items():
            if label != -1 and len(columns) > 1:  # Exclude noise and single columns
                result_clusters.append(columns)
        
        return result_clusters
        
    except Exception as e:
        st.warning(f"Embedding-based clustering failed, falling back to simple similarity: {e}")
        # Fallback to simple similarity clustering
        return simple_similarity_clustering(dataframes)

def simple_similarity_clustering(dataframes):
    """
    Fallback simple similarity clustering based on column names only.
    """
    all_columns = []
    string_columns = []
    
    for filename, df in dataframes.items():
        for col in df.columns:
            all_columns.append(col)
            if isinstance(col, str) and (df[col].dtype == 'object' or df[col].dtype == 'string'):
                sample_values = df[col].dropna().head(10)
                if len(sample_values) > 0 and all(isinstance(val, str) for val in sample_values):
                    string_columns.append(col)
    
    # Simple clustering based on string similarity
    clusters = []
    used_columns = set()
    
    for i, col1 in enumerate(string_columns):
        if col1 in used_columns:
            continue
            
        cluster = [col1]
        used_columns.add(col1)
        
        for col2 in string_columns[i+1:]:
            if col2 in used_columns:
                continue
                
            # Simple similarity check
            col1_norm = col1.lower().replace('_', '').replace(' ', '')
            col2_norm = col2.lower().replace('_', '').replace(' ', '')
            
            if col1_norm == col2_norm or col1_norm in col2_norm or col2_norm in col1_norm:
                cluster.append(col2)
                used_columns.add(col2)
        
        if len(cluster) > 1:
            clusters.append(cluster)
    
    return clusters

def main():
    """Main Streamlit application."""
    st.title("Data Clean Room Processor")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a step:",
        [
            "Upload Files",
            "Column Analysis", 
            "Data Value Standardizer"
        ],
        index=["Upload Files", "Column Analysis", "Data Value Standardizer"].index(st.session_state.current_page.replace("📁 ", "").replace("🔍 ", "").replace("🥤 ", ""))
    )
    
    # Update current page if sidebar selection changes
    if page == "Upload Files":
        current_page = "📁 Upload Files"
    elif page == "Column Analysis":
        current_page = "🔍 Column Analysis"
    elif page == "Data Value Standardizer":
        current_page = "🥤 Data Value Standardizer"
    
    if current_page != st.session_state.current_page:
        st.session_state.current_page = current_page
        st.rerun()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Status:**")
    
    if st.session_state.dataframes:
        st.sidebar.success(f"{len(st.session_state.dataframes)} dataset(s) loaded")
    else:
        st.sidebar.warning("No datasets loaded")
    
    if st.session_state.analysis_complete:
        st.sidebar.success("Analysis complete")
    else:
        st.sidebar.warning("Analysis pending")
    
    if st.session_state.cleaning_finished:
        st.sidebar.success("Data cleaned")
    else:
        st.sidebar.warning("Data cleaning pending")
    

    
    # Sidebar actions
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Actions:**")
    
    if st.sidebar.button("Reset All"):
        st.session_state.clear()
        st.sidebar.success("Reset complete")
        st.rerun()
    
    # Main content based on navigation
    if st.session_state.current_page == "📁 Upload Files":
        upload_files()
    
    elif st.session_state.current_page == "🔍 Column Analysis":
        if st.session_state.dataframes:
            # Always show analysis results (run if not already done)
            if not st.session_state.analysis_complete or not st.session_state.analysis_report:
                analyze_columns()
            # Show analysis results
            st.header("Automatic Column Analysis")
            if st.session_state.analysis_report:
                report = st.session_state.analysis_report
                
                
                # Initialize custom clusters if not exists
                if 'custom_clusters' not in st.session_state:
                    st.session_state.custom_clusters = []
                
                # Show similarity clusters in table format
                if report.get('similarity_clusters'):
                    st.write("**Tip:** Use the dropdowns below to customize your column groups. Select/deselect columns as needed.")
                    
                    # Get all file names for table headers
                    file_names = list(st.session_state.dataframes.keys())
                    
                    # Create table-like structure with borders and styling
                    st.markdown("""
                    <style>
                    .cluster-table-container {
                        border: 2px solid #ddd;
                        border-radius: 5px;
                        padding: 10px;
                        margin: 10px 0;
                        background-color: #f9f9f9;
                    }
                    .cluster-header {
                        background-color: #e9ecef;
                        border: 1px solid #ddd;
                        padding: 8px;
                        margin: 2px 0;
                        border-radius: 3px;
                        font-weight: bold;
                    }
                    .cluster-row {
                        background-color: white;
                        border: 1px solid #ddd;
                        padding: 6px;
                        margin: 1px 0;
                        border-radius: 3px;
                    }
                    .cluster-label {
                        background-color: #f0f0f0;
                        padding: 12px 8px;
                        border-radius: 3px;
                        font-weight: bold;
                        text-align: center;
                        margin: 4px 0;
                        min-height: 40px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }
                    
                    /* Ensure proper alignment of multiselect widgets */
                    .stMultiSelect {
                        margin: 0 !important;
                        padding: 0 !important;
                        display: flex !important;
                        align-items: center !important;
                    }
                    
                    /* Align columns properly */
                    .stColumns {
                        align-items: center !important;
                    }
                    
                    /* Remove any extra spacing */
                    .stMultiSelect > div {
                        margin: 0 !important;
                        padding: 0 !important;
                    }
                    
                    /* Ensure column group labels and dropdowns are at same level */
                    .cluster-label {
                        display: flex !important;
                        align-items: center !important;
                        justify-content: center !important;
                        height: 40px !important;
                        margin: 0 !important;
                    }
                    
                    /* Force vertical alignment */
                    .stColumns > div {
                        display: flex !important;
                        align-items: center !important;
                        justify-content: center !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Start table container
                    st.markdown('<div class="cluster-table-container">', unsafe_allow_html=True)
                    
                    # Create table header
                    header_cols = st.columns([1] + [1] * len(file_names))
                    with header_cols[0]:
                        st.markdown('<div class="cluster-header" style="text-align: center;">Column Group</div>', unsafe_allow_html=True)
                    for i, filename in enumerate(file_names):
                        with header_cols[i+1]:
                            # Extract sheet name from filename if it contains " - "
                            if " - " in filename:
                                sheet_name = filename.split(" - ")[-1]
                                display_name = sheet_name
                            else:
                                display_name = filename
                            st.markdown(f'<div class="cluster-header">{display_name}</div>', unsafe_allow_html=True)
                    
                    # Display auto-generated clusters as rows
                    for i, cluster in enumerate(report['similarity_clusters']):
                        if len(cluster) > 1:  # Only show clusters with multiple columns
                            cluster_key = f"cluster_{i}"
                            if cluster_key not in st.session_state.user_cluster_selections:
                                st.session_state.user_cluster_selections[cluster_key] = cluster.copy()
                            
                            # Create row for this cluster
                            if len(file_names) <= 3:
                                # For 1-3 files, use equal spacing
                                col1, *file_cols = st.columns([1] + [1] * len(file_names))
                            else:
                                # For more files, use smaller widths to fit more columns
                                col1, *file_cols = st.columns([1] + [0.8] * len(file_names))
                            
                            with col1:
                                st.markdown(f'<div class="cluster-label">Column Group {i+1}</div>', unsafe_allow_html=True)
                            
                            # Create multi-select dropdowns for each file
                            for j, filename in enumerate(file_names):
                                with file_cols[j]:
                                    # Get all available columns from this file
                                    all_file_columns = st.session_state.dataframes[filename].columns.tolist()
                                    
                                    # Get current selections for this file in this cluster
                                    current_selections = [col for col in all_file_columns if col in st.session_state.user_cluster_selections[cluster_key]]
                                    
                                    # Multi-select dropdown
                                    selected_columns = st.multiselect(
                                        "",
                                        options=sorted([col for col in all_file_columns if isinstance(col, str)]),
                                        default=current_selections,
                                        key=f"cluster_{i}_{filename}"
                                    )
                                    
                                    # Update user selections
                                    # Remove old selections for this file
                                    st.session_state.user_cluster_selections[cluster_key] = [
                                        col for col in st.session_state.user_cluster_selections[cluster_key] 
                                        if col not in all_file_columns
                                    ]
                                    # Add new selections
                                    st.session_state.user_cluster_selections[cluster_key].extend(selected_columns)
                    
                    # Display custom clusters as additional rows
                    for i, custom_cluster in enumerate(st.session_state.custom_clusters):
                        cluster_key = f"custom_cluster_{i}"
                        if cluster_key not in st.session_state.user_cluster_selections:
                            st.session_state.user_cluster_selections[cluster_key] = []
                        
                        # Create row for this custom cluster
                        if len(file_names) <= 3:
                            # For 1-3 files, use equal spacing
                            col1, *file_cols = st.columns([1] + [1] * len(file_names))
                        else:
                            # For more files, use smaller widths to fit more columns
                            col1, *file_cols = st.columns([1] + [0.8] * len(file_names))
                        
                        with col1:
                            st.markdown(f'<div class="cluster-label">Custom Column Group {i+1}</div>', unsafe_allow_html=True)
                        
                        # Create multi-select dropdowns for each file
                        for j, filename in enumerate(file_names):
                            with file_cols[j]:
                                # Get all available columns from this file
                                all_file_columns = st.session_state.dataframes[filename].columns.tolist()
                                
                                # Get current selections for this file in this custom cluster
                                current_selections = [col for col in all_file_columns if col in st.session_state.user_cluster_selections[cluster_key]]
                                
                                # Multi-select dropdown
                                selected_columns = st.multiselect(
                                    "",
                                    options=sorted([col for col in all_file_columns if isinstance(col, str)]),
                                    default=current_selections,
                                    key=f"custom_cluster_{i}_{filename}"
                                )
                                
                                # Update user selections
                                # Remove old selections for this file
                                st.session_state.user_cluster_selections[cluster_key] = [
                                    col for col in st.session_state.user_cluster_selections[cluster_key] 
                                    if col not in all_file_columns
                                ]
                                # Add new selections
                                st.session_state.user_cluster_selections[cluster_key].extend(selected_columns)
                    
                    # End table container
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Add Custom Column Group button
                    st.divider()
                    if st.button("Add Custom Column Group", type="primary"):
                        st.session_state.custom_clusters.append(f"Custom Column Group {len(st.session_state.custom_clusters) + 1}")
                        st.success("New custom column group added!")
                        st.rerun()
                    
                    # Show comprehensive summary table of all column groups
                    if st.session_state.user_cluster_selections:
                        st.subheader("Comprehensive Column Groups Summary")
                        st.write("**Detailed information for all column groups:**")
                        
                        # Generate comprehensive summary table
                        summary_data = generate_column_groups_summary_table(
                            st.session_state.user_cluster_selections, 
                            st.session_state.custom_clusters, 
                            st.session_state.dataframes
                        )
                        
                        if summary_data:
                            # Create interactive summary table
                            summary_df = pd.DataFrame(summary_data)
                            
                            # Display the table with editable feedback column
                            edited_summary_df = st.data_editor(
                                summary_df,
                                use_container_width=True,
                                height=350,
                                key="column_groups_summary_table",
                                hide_index=True,
                                column_config={
                                    "Column Group Name": st.column_config.TextColumn("Column Group Name", disabled=True),
                                    "Columns in Group": st.column_config.TextColumn("Columns in Group", disabled=True),
                                    "Total Unique Values": st.column_config.NumberColumn("Total Unique Values", disabled=True),
                                    "Sample Values": st.column_config.TextColumn("Sample Values", disabled=True),
                                    "Additional Instructions/Feedback": st.column_config.TextColumn(
                                        "Additional Instructions/Feedback", 
                                        help="Enter any notes, instructions, or feedback for this column group"
                                    )
                                }
                            )
                            
                            # Store feedback in session state
                            if 'column_groups_feedback' not in st.session_state:
                                st.session_state.column_groups_feedback = {}
                            
                            for idx, row in edited_summary_df.iterrows():
                                group_name = row['Column Group Name']
                                feedback = row['Additional Instructions/Feedback']
                                if feedback.strip():
                                    st.session_state.column_groups_feedback[group_name] = feedback
                            
                            # Show total summary
                            total_columns = sum(len(row['Columns in Group'].split(', ')) for row in summary_data)
                            total_unique_values = sum(row['Total Unique Values'] for row in summary_data)
                            
                        else:
                            st.info("No column groups have been configured yet. Use the dropdowns above to select columns for each group.")
                        
                        # Add a button to reset customizations
                        if st.button("Reset All Customizations"):
                            st.session_state.user_cluster_selections = {}
                            st.session_state.custom_clusters = []
                            st.session_state.column_groups_feedback = {}
                            st.session_state.all_column_groups_mappings = {}
                            st.session_state.all_column_groups_feedback = {}
                            st.session_state.mappings_generated = False
                            st.success("All customizations reset! Run analysis again to see original column groups.")
                            st.rerun()
                
                
                st.divider()
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if st.button("Back to Upload"):
                        st.session_state.current_page = "📁 Upload Files"
                        st.rerun()
                
                with col2:
                    if not st.session_state.mappings_generated:
                        if st.button("Generate Initial Mappings", type="primary"):
                            all_mappings = generate_mappings_for_all_groups(
                                st.session_state.user_cluster_selections,
                                st.session_state.custom_clusters,
                                st.session_state.dataframes
                            )
                            st.session_state.all_column_groups_mappings = all_mappings
                            st.session_state.mappings_generated = True
                            st.success("Mappings generated for all column groups!")
                            # Auto-navigate to Data Value Standardizer
                            st.session_state.current_page = "🥤 Data Value Standardizer"
                            st.rerun()
                    else:
                        if st.button("Regenerate Mappings", type="primary"):
                            all_mappings = generate_mappings_for_all_groups(
                                st.session_state.user_cluster_selections,
                                st.session_state.custom_clusters,
                                st.session_state.dataframes
                            )
                            st.session_state.all_column_groups_mappings = all_mappings
                            st.success("Mappings regenerated for all column groups!")
                            st.rerun()
        else:
            st.warning("Please upload files first.")
    
    elif st.session_state.current_page == "🥤 Data Value Standardizer":
        dedicated_data_cleaning_interface()
    

if __name__ == "__main__":
    main() 