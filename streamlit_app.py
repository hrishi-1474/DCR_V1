import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import tempfile
from typing import Dict, List, Tuple, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pickle

# Import our existing Phi LLM functions
from phi_llm_functions import (
    PhiLLMProcessor, 
    load_data_files, 
    create_column_mapping_matrix, 
    generate_standardization_report
)

# Page configuration
st.set_page_config(
    page_title="Clean Room Data Processor",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}
if 'column_mappings' not in st.session_state:
    st.session_state.column_mappings = {}
if 'user_feedback' not in st.session_state:
    st.session_state.user_feedback = {}
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = {}
if 'transformation_rules' not in st.session_state:
    st.session_state.transformation_rules = {}
if 'phi_processor' not in st.session_state:
    st.session_state.phi_processor = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

def initialize_phi_processor():
    """Initialize the Phi LLM processor."""
    if st.session_state.phi_processor is None:
        with st.spinner("Initializing Phi-3.5 Mini..."):
            st.session_state.phi_processor = PhiLLMProcessor()
    return st.session_state.phi_processor

def save_user_feedback():
    """Save user feedback to a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"user_feedback_{timestamp}.json"
    
    feedback_data = {
        'column_mappings': st.session_state.column_mappings,
        'user_feedback': st.session_state.user_feedback,
        'transformation_rules': st.session_state.transformation_rules,
        'timestamp': timestamp
    }
    
    with open(filename, 'w') as f:
        json.dump(feedback_data, f, indent=2)
    
    return filename

def load_user_feedback(filename: str):
    """Load user feedback from a file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        st.session_state.column_mappings = data.get('column_mappings', {})
        st.session_state.user_feedback = data.get('user_feedback', {})
        st.session_state.transformation_rules = data.get('transformation_rules', {})
        
        return True
    except Exception as e:
        st.error(f"Error loading feedback file: {e}")
        return False

def upload_files():
    """Handle file upload with Excel sheet selection."""
    st.header("📁 Upload Data Files")
    
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
                    st.success(f"✅ Loaded {file_name}: {df.shape[0]} rows, {df.shape[1]} columns")
                
                elif file_name.endswith(('.xlsx', '.xls')):
                    # Get available sheets
                    excel_file = pd.ExcelFile(uploaded_file)
                    available_sheets = excel_file.sheet_names
                    
                    if len(available_sheets) > 1:
                        st.write(f"📊 **{file_name}** has multiple sheets:")
                        selected_sheets = st.multiselect(
                            f"Select sheets to import from {file_name}",
                            available_sheets,
                            default=available_sheets[:1],
                            key=f"sheets_{file_name}"
                        )
                        
                        for sheet_name in selected_sheets:
                            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                            full_name = f"{file_name} - {sheet_name}"
                            dataframes[full_name] = df
                            st.success(f"✅ Loaded {full_name}: {df.shape[0]} rows, {df.shape[1]} columns")
                    else:
                        df = pd.read_excel(uploaded_file)
                        dataframes[file_name] = df
                        st.success(f"✅ Loaded {file_name}: {df.shape[0]} rows, {df.shape[1]} columns")
            
            except Exception as e:
                st.error(f"❌ Error loading {file_name}: {e}")
        
        if dataframes:
            st.session_state.dataframes = dataframes
            st.success(f"🎉 Successfully loaded {len(dataframes)} dataset(s)")
            
            # Show dataset summary
            st.subheader("📊 Dataset Summary")
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
            
            return True
    
    return False

def analyze_columns():
    """Perform automatic column analysis."""
    st.header("🔍 Automatic Column Analysis")
    
    if not st.session_state.dataframes:
        st.warning("Please upload files first.")
        return False
    
    phi_processor = initialize_phi_processor()
    
    with st.spinner("Analyzing columns across all datasets..."):
        # Generate comprehensive analysis
        report = generate_standardization_report(st.session_state.dataframes, phi_processor)
        
        # Store analysis results
        st.session_state.analysis_report = report
        st.session_state.analysis_complete = True
        
        # Display key findings
        st.subheader("📈 Analysis Results")
        
        # Show similarity clusters
        if report.get('similarity_clusters'):
            st.write("**🔗 Similar Column Clusters Found:**")
            for i, cluster in enumerate(report['similarity_clusters']):
                if len(cluster) > 1:  # Only show clusters with multiple columns
                    st.write(f"**Cluster {i+1}:** {', '.join(cluster)}")
        
        # Show dataset summaries
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Dataset Overview:**")
            for filename, summary in report['dataset_summary'].items():
                st.write(f"• {filename}: {summary['shape'][0]} rows, {summary['shape'][1]} columns")
        
        with col2:
            st.write("**🎯 Quality Assessment:**")
            for filename, quality in report['quality_assessment'].items():
                if isinstance(quality, dict) and 'issues' in quality:
                    issues_count = len(quality.get('issues', []))
                    st.write(f"• {filename}: {issues_count} issues found")
        
        return True

def display_column_mapping_interface():
    """Display the column mapping interface with user feedback."""
    st.header("🗺️ Column Mapping & User Feedback")
    
    if not st.session_state.analysis_complete:
        st.warning("Please run column analysis first.")
        return
    
    report = st.session_state.analysis_report
    
    # Display similarity clusters for user feedback
    if report.get('similarity_clusters'):
        st.subheader("🔗 Detected Column Similarities")
        
        for i, cluster in enumerate(report['similarity_clusters']):
            if len(cluster) > 1:
                st.write(f"**Cluster {i+1}:**")
                
                # Create a mapping interface for each cluster
                cluster_key = f"cluster_{i}"
                
                # Get user feedback for this cluster
                if cluster_key not in st.session_state.user_feedback:
                    st.session_state.user_feedback[cluster_key] = {
                        'columns': cluster,
                        'is_related': True,
                        'standard_name': '',
                        'notes': ''
                    }
                
                feedback = st.session_state.user_feedback[cluster_key]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"Columns: {', '.join(cluster)}")
                    
                    # User feedback options
                    is_related = st.checkbox(
                        "These columns are related",
                        value=feedback['is_related'],
                        key=f"related_{cluster_key}"
                    )
                    
                    if is_related:
                        standard_name = st.text_input(
                            "Suggested standard name:",
                            value=feedback['standard_name'],
                            key=f"standard_{cluster_key}"
                        )
                        
                        notes = st.text_area(
                            "Additional notes:",
                            value=feedback['notes'],
                            key=f"notes_{cluster_key}"
                        )
                        
                        # Update feedback
                        st.session_state.user_feedback[cluster_key].update({
                            'is_related': is_related,
                            'standard_name': standard_name,
                            'notes': notes
                        })
                
                with col2:
                    # Show sample data from first column in cluster
                    if cluster:
                        sample_col = cluster[0]
                        for filename, df in st.session_state.dataframes.items():
                            if sample_col in df.columns:
                                st.write(f"**Sample from {filename}:**")
                                st.write(df[sample_col].head(3).tolist())
                                break
                
                st.divider()

def column_selection_interface():
    """Interactive column selection with auto-suggestions."""
    st.header("📋 Column Selection for Cleaning")
    
    if not st.session_state.dataframes:
        st.warning("Please upload files first.")
        return
    
    # Get all columns from all datasets
    all_columns = {}
    for filename, df in st.session_state.dataframes.items():
        all_columns[filename] = df.columns.tolist()
    
    st.subheader("Select columns to clean:")
    
    # Create selection interface
    selected_columns = {}
    
    for filename, columns in all_columns.items():
        st.write(f"**📁 {filename}:**")
        
        # Get related columns from analysis
        related_columns = []
        if st.session_state.analysis_complete:
            report = st.session_state.analysis_report
            for cluster in report.get('similarity_clusters', []):
                for col in columns:
                    if col in cluster:
                        related_columns.extend([c for c in cluster if c != col])
        
        # Column selection with auto-suggestions
        selected = st.multiselect(
            f"Select columns from {filename}",
            columns,
            key=f"select_{filename}",
            help="Select columns to clean. Related columns will be auto-selected."
        )
        
        if selected:
            selected_columns[filename] = selected
            
            # Auto-select related columns
            if st.session_state.analysis_complete:
                auto_selected = []
                for col in selected:
                    for cluster in report.get('similarity_clusters', []):
                        if col in cluster:
                            for related_col in cluster:
                                if related_col != col:
                                    for other_filename, other_columns in all_columns.items():
                                        if related_col in other_columns:
                                            if other_filename not in auto_selected:
                                                auto_selected.append(other_filename)
                
                if auto_selected:
                    st.info(f"💡 Related columns found in: {', '.join(auto_selected)}")
    
    st.session_state.selected_columns = selected_columns
    
    # Show data preview for selected columns
    if selected_columns:
        st.subheader("📊 Data Preview")
        
        for filename, columns in selected_columns.items():
            if filename in st.session_state.dataframes:
                df = st.session_state.dataframes[filename]
                
                st.write(f"**{filename}:**")
                preview_df = df[columns].head(10)
                st.dataframe(preview_df, use_container_width=True)

def data_cleaning_interface():
    """Human-in-the-loop data cleaning interface."""
    st.header("🧹 Data Cleaning & Standardization")
    
    if not st.session_state.selected_columns:
        st.warning("Please select columns to clean first.")
        return
    
    phi_processor = initialize_phi_processor()
    
    # Process each selected column
    for filename, columns in st.session_state.selected_columns.items():
        if filename in st.session_state.dataframes:
            df = st.session_state.dataframes[filename]
            
            st.subheader(f"📁 Processing {filename}")
            
            for column in columns:
                st.write(f"**Column: {column}**")
                
                # Get sample data for analysis
                sample_data = df[column].dropna().head(20).tolist()
                
                # Analyze the column with Phi
                with st.spinner(f"Analyzing {column}..."):
                    analysis_prompt = f"""
                    Analyze this column data and suggest cleaning/standardization rules:
                    Column: {column}
                    Sample data: {sample_data}
                    
                    Please suggest:
                    1. Data type conversions
                    2. Format standardization
                    3. Value cleaning rules
                    4. Duplicate handling
                    
                    Format as JSON with keys: data_type, format_rules, cleaning_rules, duplicate_handling
                    """
                    
                    analysis = phi_processor.query_phi(analysis_prompt)
                    
                    # Try to parse JSON response
                    try:
                        import re
                        json_match = re.search(r'\{.*\}', analysis, re.DOTALL)
                        if json_match:
                            cleaning_rules = json.loads(json_match.group())
                        else:
                            cleaning_rules = {"raw_analysis": analysis}
                    except:
                        cleaning_rules = {"raw_analysis": analysis}
                
                # Display analysis and get user feedback
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**Current Data:**")
                    st.write(f"Sample: {sample_data[:5]}")
                    st.write(f"Data type: {df[column].dtype}")
                    st.write(f"Null values: {df[column].isnull().sum()}")
                    st.write(f"Unique values: {df[column].nunique()}")
                
                with col2:
                    st.write("**Suggested Cleaning Rules:**")
                    if isinstance(cleaning_rules, dict):
                        for key, value in cleaning_rules.items():
                            if key != "raw_analysis":
                                st.write(f"**{key}:** {value}")
                    else:
                        st.write(cleaning_rules)
                
                # User approval interface
                st.write("**Apply these cleaning rules?**")
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    apply_rules = st.button("✅ Apply", key=f"apply_{filename}_{column}")
                
                with col2:
                    modify_rules = st.button("✏️ Modify", key=f"modify_{filename}_{column}")
                
                with col3:
                    skip_column = st.button("⏭️ Skip", key=f"skip_{filename}_{column}")
                
                if apply_rules:
                    # Store transformation rules
                    rule_key = f"{filename}_{column}"
                    st.session_state.transformation_rules[rule_key] = cleaning_rules
                    
                    # Apply basic transformations
                    cleaned_df = df.copy()
                    
                    # Apply data type conversion if suggested
                    if isinstance(cleaning_rules, dict) and 'data_type' in cleaning_rules:
                        try:
                            if cleaning_rules['data_type'].lower() == 'string':
                                cleaned_df[column] = cleaned_df[column].astype(str)
                            elif cleaning_rules['data_type'].lower() == 'integer':
                                cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce')
                            elif cleaning_rules['data_type'].lower() == 'float':
                                cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce')
                        except:
                            pass
                    
                    # Update the dataframe
                    st.session_state.dataframes[filename] = cleaned_df
                    st.success(f"✅ Applied cleaning rules to {column}")
                
                st.divider()

def download_interface():
    """Interface for downloading cleaned data."""
    st.header("📥 Download Cleaned Data")
    
    if not st.session_state.dataframes:
        st.warning("No data to download.")
        return
    
    st.subheader("Export Options")
    
    # Export format selection
    export_format = st.selectbox(
        "Select export format:",
        ["Excel (.xlsx)", "CSV", "JSON"],
        help="Choose the format for your cleaned data"
    )
    
    # File naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"cleaned_data_{timestamp}"
    filename = st.text_input("Filename (without extension):", value=default_filename)
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        include_original = st.checkbox("Include original data", value=True)
        include_metadata = st.checkbox("Include metadata", value=True)
    
    with col2:
        compress_files = st.checkbox("Compress files", value=False)
        split_by_dataset = st.checkbox("Split by dataset", value=True)
    
    # Export button
    if st.button("📥 Export Cleaned Data"):
        with st.spinner("Preparing export..."):
            try:
                if split_by_dataset:
                    # Export each dataset separately
                    for filename_df, df in st.session_state.dataframes.items():
                        safe_filename = filename_df.replace('/', '_').replace(' ', '_')
                        
                        if export_format == "Excel (.xlsx)":
                            output_filename = f"{filename}_{safe_filename}.xlsx"
                            df.to_excel(output_filename, index=False)
                        elif export_format == "CSV":
                            output_filename = f"{filename}_{safe_filename}.csv"
                            df.to_csv(output_filename, index=False)
                        elif export_format == "JSON":
                            output_filename = f"{filename}_{safe_filename}.json"
                            df.to_json(output_filename, orient='records', indent=2)
                        
                        # Provide download link
                        with open(output_filename, 'rb') as f:
                            st.download_button(
                                label=f"📥 Download {safe_filename}",
                                data=f.read(),
                                file_name=output_filename,
                                mime="application/octet-stream"
                            )
                else:
                    # Export as single file
                    if export_format == "Excel (.xlsx)":
                        output_filename = f"{filename}.xlsx"
                        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
                            for filename_df, df in st.session_state.dataframes.items():
                                sheet_name = filename_df[:31]  # Excel sheet name limit
                                df.to_excel(writer, sheet_name=sheet_name, index=False)
                    elif export_format == "CSV":
                        output_filename = f"{filename}.csv"
                        # Combine all dataframes
                        combined_df = pd.concat(st.session_state.dataframes.values(), ignore_index=True)
                        combined_df.to_csv(output_filename, index=False)
                    elif export_format == "JSON":
                        output_filename = f"{filename}.json"
                        # Export as JSON with dataset names as keys
                        export_data = {}
                        for filename_df, df in st.session_state.dataframes.items():
                            export_data[filename_df] = df.to_dict('records')
                        
                        with open(output_filename, 'w') as f:
                            json.dump(export_data, f, indent=2)
                    
                    # Provide download link
                    with open(output_filename, 'rb') as f:
                        st.download_button(
                            label=f"📥 Download {output_filename}",
                            data=f.read(),
                            file_name=output_filename,
                            mime="application/octet-stream"
                        )
                
                st.success("✅ Export completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Export failed: {e}")

def main():
    """Main Streamlit application."""
    st.title("🧹 Clean Room Data Processor")
    st.markdown("**Intelligent data cleaning and standardization with Phi-3.5 Mini**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a step:",
        [
            "📁 Upload Files",
            "🔍 Column Analysis", 
            "🗺️ Column Mapping",
            "📋 Column Selection",
            "🧹 Data Cleaning",
            "📥 Download Data"
        ]
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Status:**")
    
    if st.session_state.dataframes:
        st.sidebar.success(f"✅ {len(st.session_state.dataframes)} dataset(s) loaded")
    else:
        st.sidebar.warning("⚠️ No datasets loaded")
    
    if st.session_state.analysis_complete:
        st.sidebar.success("✅ Analysis complete")
    else:
        st.sidebar.warning("⚠️ Analysis pending")
    
    if st.session_state.selected_columns:
        total_selected = sum(len(cols) for cols in st.session_state.selected_columns.values())
        st.sidebar.success(f"✅ {total_selected} column(s) selected")
    else:
        st.sidebar.warning("⚠️ No columns selected")
    
    # Sidebar actions
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Actions:**")
    
    if st.sidebar.button("💾 Save Feedback"):
        if st.session_state.user_feedback or st.session_state.transformation_rules:
            filename = save_user_feedback()
            st.sidebar.success(f"✅ Saved to {filename}")
        else:
            st.sidebar.warning("⚠️ No feedback to save")
    
    if st.sidebar.button("🔄 Reset All"):
        st.session_state.clear()
        st.sidebar.success("✅ Reset complete")
        st.rerun()
    
    # Main content based on navigation
    if page == "📁 Upload Files":
        upload_files()
    
    elif page == "🔍 Column Analysis":
        if st.session_state.dataframes:
            if st.button("🚀 Run Column Analysis"):
                analyze_columns()
        else:
            st.warning("Please upload files first.")
    
    elif page == "🗺️ Column Mapping":
        display_column_mapping_interface()
    
    elif page == "📋 Column Selection":
        column_selection_interface()
    
    elif page == "🧹 Data Cleaning":
        data_cleaning_interface()
    
    elif page == "📥 Download Data":
        download_interface()

if __name__ == "__main__":
    main() 