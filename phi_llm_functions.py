import ollama
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import json
import re
from typing import List, Dict, Tuple, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhiLLMProcessor:
    """
    A class to handle interactions with Phi-3.5 Mini for data cleaning and standardization.
    """
    
    def __init__(self, model_name: str = "phi3"):
        """
        Initialize the Phi LLM processor.
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def query_phi(self, prompt: str, system_prompt: str = None) -> str:
        """
        Send a query to Phi-3.5 Mini and get a response.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            
        Returns:
            The model's response as a string
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = ollama.chat(model=self.model_name, messages=messages)
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error querying Phi model: {e}")
            return ""
    
    def analyze_column_names(self, column_names: List[str]) -> Dict[str, Any]:
        """
        Analyze column names to identify patterns and suggest standardization.
        
        Args:
            column_names: List of column names to analyze
            
        Returns:
            Dictionary with analysis results
        """
        prompt = f"""
        Analyze these column names and provide insights:
        {column_names}
        
        Please provide:
        1. Data types you can infer from the names
        2. Potential standardization suggestions
        3. Any patterns or categories you notice
        4. Suggested standard column names
        
        Format your response as JSON with keys: data_types, standardization, patterns, suggested_names
        """
        
        response = self.query_phi(prompt)
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"raw_response": response}
        except:
            return {"raw_response": response}
    
    def find_similar_columns(self, columns1: List[str], columns2: List[str], 
                           similarity_threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Find similar columns between two datasets using embeddings.
        
        Args:
            columns1: Column names from first dataset
            columns2: Column names from second dataset
            similarity_threshold: Minimum similarity score to consider a match
            
        Returns:
            List of tuples (col1, col2, similarity_score)
        """
        # Get embeddings for all column names
        all_columns = columns1 + columns2
        embeddings = self.embedding_model.encode(all_columns)
        
        # Split embeddings back to the two datasets
        emb1 = embeddings[:len(columns1)]
        emb2 = embeddings[len(columns1):]
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(emb1, emb2)
        
        # Find matches above threshold
        matches = []
        for i, col1 in enumerate(columns1):
            for j, col2 in enumerate(columns2):
                similarity = similarity_matrix[i][j]
                if similarity >= similarity_threshold:
                    matches.append((col1, col2, similarity))
        
        # Sort by similarity score (highest first)
        matches.sort(key=lambda x: x[2], reverse=True)
        
        return matches
    
    def suggest_data_standardization(self, sample_data: Dict[str, List], 
                                   target_format: str = "standard") -> Dict[str, Any]:
        """
        Suggest data standardization rules based on sample data.
        
        Args:
            sample_data: Dictionary with column names as keys and sample values as lists
            target_format: Target format for standardization
            
        Returns:
            Dictionary with standardization suggestions
        """
        # Create a sample data preview
        sample_preview = {}
        for col, values in sample_data.items():
            sample_preview[col] = values[:5] if len(values) > 5 else values
        
        prompt = f"""
        Analyze this sample data and suggest standardization rules:
        
        Sample Data:
        {json.dumps(sample_preview, indent=2)}
        
        Target Format: {target_format}
        
        Please suggest:
        1. Data type conversions needed
        2. Format standardization (dates, numbers, text)
        3. Value mappings or transformations
        4. Data quality improvements
        
        Format as JSON with keys: conversions, formats, mappings, quality_improvements
        """
        
        response = self.query_phi(prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"raw_response": response}
        except:
            return {"raw_response": response}
    
    def detect_data_anomalies(self, data: pd.DataFrame, columns: List[str] = None) -> Dict[str, Any]:
        """
        Detect anomalies in the data using Phi-3.5 Mini.
        
        Args:
            data: DataFrame to analyze
            columns: Specific columns to analyze (if None, analyze all)
            
        Returns:
            Dictionary with anomaly detection results
        """
        if columns is None:
            columns = data.columns.tolist()
        
        # Create data summary for analysis
        data_summary = {}
        for col in columns:
            if col in data.columns:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    data_summary[col] = {
                        "dtype": str(data[col].dtype),
                        "unique_count": int(data[col].nunique()),
                        "null_count": int(data[col].isnull().sum()),
                        "sample_values": col_data.head(10).tolist()
                    }
        
        prompt = f"""
        Analyze this data summary and identify potential anomalies or data quality issues:
        
        {json.dumps(data_summary, indent=2)}
        
        Look for:
        1. Inconsistent data types
        2. Unusual value patterns
        3. Missing data patterns
        4. Potential data quality issues
        5. Suggestions for data cleaning
        
        Format as JSON with keys: anomalies, quality_issues, cleaning_suggestions
        """
        
        response = self.query_phi(prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"raw_response": response}
        except:
            return {"raw_response": response}
    
    def generate_mapping_rules(self, source_columns: List[str], target_columns: List[str],
                             sample_data: Dict[str, List] = None) -> Dict[str, Any]:
        """
        Generate mapping rules between source and target columns.
        
        Args:
            source_columns: Source column names
            target_columns: Target column names
            sample_data: Optional sample data for context
            
        Returns:
            Dictionary with mapping rules and suggestions
        """
        prompt = f"""
        Generate mapping rules between these column sets:
        
        Source Columns: {source_columns}
        Target Columns: {target_columns}
        
        {f'Sample Data: {json.dumps(sample_data, indent=2)}' if sample_data else ''}
        
        Please provide:
        1. Column-to-column mappings
        2. Transformation rules needed
        3. Data type conversions
        4. Validation rules
        
        Format as JSON with keys: mappings, transformations, conversions, validations
        """
        
        response = self.query_phi(prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"raw_response": response}
        except:
            return {"raw_response": response}
    
    def cluster_similar_columns(self, all_columns: List[str], 
                              eps: float = 0.3, min_samples: int = 2) -> List[List[str]]:
        """
        Cluster similar column names using embeddings and DBSCAN.
        
        Args:
            all_columns: List of all column names to cluster
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter
            
        Returns:
            List of clusters, where each cluster is a list of similar column names
        """
        # Get embeddings
        embeddings = self.embedding_model.encode(all_columns)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Group columns by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(all_columns[i])
        
        return list(clusters.values())
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality validation using Phi-3.5 Mini.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        # Create comprehensive data summary
        summary = {
            "shape": data.shape,
            "columns": data.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.to_dict().items()},
            "null_counts": data.isnull().sum().to_dict(),
            "unique_counts": data.nunique().to_dict(),
            "sample_data": {}
        }
        
        # Add sample data for each column
        for col in data.columns:
            sample_values = data[col].dropna().head(5).tolist()
            summary["sample_data"][col] = sample_values
        
        prompt = f"""
        Perform comprehensive data quality validation on this dataset:
        
        {json.dumps(summary, indent=2)}
        
        Please analyze:
        1. Data completeness
        2. Data consistency
        3. Data accuracy
        4. Data integrity
        5. Potential issues and recommendations
        
        Format as JSON with keys: completeness, consistency, accuracy, integrity, issues, recommendations
        """
        
        response = self.query_phi(prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"raw_response": response}
        except:
            return {"raw_response": response}

# Utility functions for the clean room PoC
def load_data_files(file_paths: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Load multiple data files and return as a dictionary.
    
    Args:
        file_paths: List of file paths to load
        
    Returns:
        Dictionary with filename as key and DataFrame as value
    """
    dataframes = {}
    
    for file_path in file_paths:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                continue
                
            filename = file_path.split('/')[-1]
            dataframes[filename] = df
            logger.info(f"Loaded {filename} with shape {df.shape}")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    return dataframes

def create_column_mapping_matrix(dataframes: Dict[str, pd.DataFrame], 
                               phi_processor: PhiLLMProcessor) -> pd.DataFrame:
    """
    Create a matrix showing column similarities across all datasets.
    
    Args:
        dataframes: Dictionary of DataFrames
        phi_processor: PhiLLMProcessor instance
        
    Returns:
        DataFrame with similarity matrix
    """
    all_columns = []
    file_names = []
    
    for filename, df in dataframes.items():
        all_columns.extend(df.columns.tolist())
        file_names.extend([filename] * len(df.columns))
    
    # Get embeddings for all columns
    embeddings = phi_processor.embedding_model.encode(all_columns)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Create DataFrame
    mapping_df = pd.DataFrame(
        similarity_matrix,
        index=[f"{file_names[i]}:{all_columns[i]}" for i in range(len(all_columns))],
        columns=[f"{file_names[i]}:{all_columns[i]}" for i in range(len(all_columns))]
    )
    
    return mapping_df

def generate_standardization_report(dataframes: Dict[str, pd.DataFrame], 
                                  phi_processor: PhiLLMProcessor) -> Dict[str, Any]:
    """
    Generate a comprehensive standardization report for all datasets.
    
    Args:
        dataframes: Dictionary of DataFrames
        phi_processor: PhiLLMProcessor instance
        
    Returns:
        Dictionary with standardization report
    """
    report = {
        "dataset_summary": {},
        "column_analysis": {},
        "similarity_clusters": {},
        "standardization_suggestions": {},
        "quality_assessment": {}
    }
    
    # Analyze each dataset
    for filename, df in dataframes.items():
        # Dataset summary
        report["dataset_summary"][filename] = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict()
        }
        
        # Column analysis
        column_analysis = phi_processor.analyze_column_names(df.columns.tolist())
        report["column_analysis"][filename] = column_analysis
        
        # Data quality assessment
        quality_assessment = phi_processor.validate_data_quality(df)
        report["quality_assessment"][filename] = quality_assessment
    
    # Find similar columns across all datasets
    all_columns = []
    for df in dataframes.values():
        all_columns.extend(df.columns.tolist())
    
    clusters = phi_processor.cluster_similar_columns(all_columns)
    report["similarity_clusters"] = clusters
    
    # Generate standardization suggestions
    for filename, df in dataframes.items():
        sample_data = {}
        for col in df.columns[:5]:  # Sample first 5 columns
            sample_data[col] = df[col].dropna().head(10).tolist()
        
        standardization = phi_processor.suggest_data_standardization(sample_data)
        report["standardization_suggestions"][filename] = standardization
    
    return report 