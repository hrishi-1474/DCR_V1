#!/usr/bin/env python3
"""
Test script for Phi-3.5 Mini functions in the clean room PoC.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from phi_llm_functions import PhiLLMProcessor, load_data_files, create_column_mapping_matrix, generate_standardization_report

def test_phi_connection():
    """Test basic connection to Phi-3.5 Mini."""
    print("🔍 Testing Phi-3.5 Mini connection...")
    
    phi_processor = PhiLLMProcessor()
    
    # Simple test query
    response = phi_processor.query_phi("Hello! Can you help me with data analysis?")
    print(f"✅ Phi-3.5 Mini Response: {response[:100]}...")
    
    return phi_processor

def test_column_analysis():
    """Test column name analysis functionality."""
    print("\n📊 Testing column name analysis...")
    
    phi_processor = PhiLLMProcessor()
    
    # Sample column names
    sample_columns = [
        "customer_id", "CustomerID", "customer_id", "cust_id",
        "first_name", "FirstName", "fname", "firstname",
        "last_name", "LastName", "lname", "lastname",
        "email_address", "Email", "email", "e_mail",
        "phone_number", "Phone", "phone", "tel"
    ]
    
    analysis = phi_processor.analyze_column_names(sample_columns)
    print(f"✅ Column Analysis Result: {analysis}")
    
    return analysis

def test_similar_column_detection():
    """Test similar column detection using embeddings."""
    print("\n🔍 Testing similar column detection...")
    
    phi_processor = PhiLLMProcessor()
    
    # Two sets of column names
    columns1 = ["customer_id", "first_name", "last_name", "email", "phone"]
    columns2 = ["CustomerID", "FirstName", "LastName", "EmailAddress", "PhoneNumber"]
    
    matches = phi_processor.find_similar_columns(columns1, columns2, similarity_threshold=0.6)
    print(f"✅ Similar Column Matches: {matches}")
    
    return matches

def test_column_clustering():
    """Test column clustering functionality."""
    print("\n📈 Testing column clustering...")
    
    phi_processor = PhiLLMProcessor()
    
    # Sample columns from different datasets
    all_columns = [
        "customer_id", "CustomerID", "cust_id", "client_id",
        "first_name", "FirstName", "fname", "firstname",
        "last_name", "LastName", "lname", "lastname",
        "email_address", "Email", "email", "e_mail",
        "phone_number", "Phone", "phone", "tel",
        "order_date", "OrderDate", "date", "purchase_date",
        "total_amount", "Amount", "price", "cost"
    ]
    
    clusters = phi_processor.cluster_similar_columns(all_columns, eps=0.3, min_samples=2)
    print(f"✅ Column Clusters: {clusters}")
    
    return clusters

def test_data_quality_validation():
    """Test data quality validation with sample data."""
    print("\n🔍 Testing data quality validation...")
    
    phi_processor = PhiLLMProcessor()
    
    # Create sample DataFrame with some quality issues
    sample_data = {
        'customer_id': [1, 2, 3, None, 5],
        'name': ['John', 'Jane', 'Bob', 'Alice', ''],
        'email': ['john@email.com', 'jane@email.com', 'invalid-email', 'alice@email.com', 'bob@email.com'],
        'age': [25, 30, 'N/A', 35, 40],
        'salary': [50000, 60000, 55000, 70000, None]
    }
    
    df = pd.DataFrame(sample_data)
    quality_report = phi_processor.validate_data_quality(df)
    print(f"✅ Data Quality Report: {quality_report}")
    
    return quality_report

def test_anomaly_detection():
    """Test anomaly detection functionality."""
    print("\n🚨 Testing anomaly detection...")
    
    phi_processor = PhiLLMProcessor()
    
    # Create sample DataFrame with anomalies
    sample_data = {
        'customer_id': [1, 2, 3, 4, 5],
        'name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'email': ['john@email.com', 'jane@email.com', 'bob@email.com', 'alice@email.com', 'invalid'],
        'age': [25, 30, 35, 40, 999],  # Anomalous age
        'salary': [50000, 60000, 55000, 70000, 1000000]  # Anomalous salary
    }
    
    df = pd.DataFrame(sample_data)
    anomalies = phi_processor.detect_data_anomalies(df)
    print(f"✅ Anomaly Detection Result: {anomalies}")
    
    return anomalies

def test_standardization_suggestions():
    """Test data standardization suggestions."""
    print("\n🔄 Testing standardization suggestions...")
    
    phi_processor = PhiLLMProcessor()
    
    # Sample data with various formats
    sample_data = {
        'customer_id': ['CUST001', 'CUST002', 'CUST003'],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'email': ['john.doe@email.com', 'jane.smith@email.com', 'bob.johnson@email.com'],
        'phone': ['(555) 123-4567', '555-234-5678', '+1-555-345-6789'],
        'date': ['2023-01-15', '01/15/2023', '15-Jan-2023']
    }
    
    standardization = phi_processor.suggest_data_standardization(sample_data)
    print(f"✅ Standardization Suggestions: {standardization}")
    
    return standardization

def test_mapping_rules():
    """Test mapping rules generation."""
    print("\n🗺️ Testing mapping rules generation...")
    
    phi_processor = PhiLLMProcessor()
    
    source_columns = ["customer_id", "first_name", "last_name", "email", "phone"]
    target_columns = ["CustomerID", "FirstName", "LastName", "EmailAddress", "PhoneNumber"]
    
    sample_data = {
        "customer_id": ["CUST001", "CUST002"],
        "first_name": ["John", "Jane"],
        "last_name": ["Doe", "Smith"],
        "email": ["john@email.com", "jane@email.com"],
        "phone": ["555-123-4567", "555-234-5678"]
    }
    
    mapping_rules = phi_processor.generate_mapping_rules(source_columns, target_columns, sample_data)
    print(f"✅ Mapping Rules: {mapping_rules}")
    
    return mapping_rules

def create_sample_datasets():
    """Create sample datasets for testing."""
    print("\n📁 Creating sample datasets...")
    
    # Dataset 1: Customer data
    customers_df = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'first_name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'last_name': ['Doe', 'Smith', 'Johnson', 'Brown', 'Wilson'],
        'email': ['john@email.com', 'jane@email.com', 'bob@email.com', 'alice@email.com', 'charlie@email.com'],
        'phone': ['555-123-4567', '555-234-5678', '555-345-6789', '555-456-7890', '555-567-8901'],
        'registration_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12']
    })
    
    # Dataset 2: Order data (different column naming)
    orders_df = pd.DataFrame({
        'CustomerID': [1, 2, 3, 4, 5],
        'FirstName': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'LastName': ['Doe', 'Smith', 'Johnson', 'Brown', 'Wilson'],
        'EmailAddress': ['john@email.com', 'jane@email.com', 'bob@email.com', 'alice@email.com', 'charlie@email.com'],
        'PhoneNumber': ['555-123-4567', '555-234-5678', '555-345-6789', '555-456-7890', '555-567-8901'],
        'OrderDate': ['2023-06-15', '2023-07-20', '2023-08-10', '2023-09-05', '2023-10-12'],
        'OrderAmount': [150.00, 200.50, 75.25, 300.00, 125.75]
    })
    
    # Dataset 3: Product data (completely different structure)
    products_df = pd.DataFrame({
        'product_id': [101, 102, 103, 104, 105],
        'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
        'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Accessories'],
        'price': [999.99, 25.50, 75.00, 299.99, 150.00],
        'supplier': ['TechCorp', 'AccessoryPro', 'AccessoryPro', 'TechCorp', 'AudioMax']
    })
    
    # Save sample datasets
    customers_df.to_csv('sample_customers.csv', index=False)
    orders_df.to_csv('sample_orders.csv', index=False)
    products_df.to_csv('sample_products.csv', index=False)
    
    print("✅ Sample datasets created: sample_customers.csv, sample_orders.csv, sample_products.csv")
    
    return {
        'customers': customers_df,
        'orders': orders_df,
        'products': products_df
    }

def test_full_workflow():
    """Test the complete workflow with sample datasets."""
    print("\n🚀 Testing complete workflow...")
    
    # Create sample datasets
    datasets = create_sample_datasets()
    
    # Initialize Phi processor
    phi_processor = PhiLLMProcessor()
    
    # Load datasets
    file_paths = ['sample_customers.csv', 'sample_orders.csv', 'sample_products.csv']
    dataframes = load_data_files(file_paths)
    
    print(f"✅ Loaded {len(dataframes)} datasets")
    
    # Generate comprehensive report
    report = generate_standardization_report(dataframes, phi_processor)
    
    print("✅ Generated standardization report")
    print(f"📊 Report contains: {list(report.keys())}")
    
    # Show some key insights
    print("\n📈 Key Insights:")
    for filename, summary in report['dataset_summary'].items():
        print(f"  - {filename}: {summary['shape'][0]} rows, {summary['shape'][1]} columns")
    
    print(f"  - Found {len(report['similarity_clusters'])} column clusters")
    
    return report

def main():
    """Run all tests."""
    print("🧪 Testing Phi-3.5 Mini Functions for Clean Room PoC")
    print("=" * 60)
    
    try:
        # Test basic connection
        phi_processor = test_phi_connection()
        
        # Test individual functions
        test_column_analysis()
        test_similar_column_detection()
        test_column_clustering()
        test_data_quality_validation()
        test_anomaly_detection()
        test_standardization_suggestions()
        test_mapping_rules()
        
        # Test complete workflow
        test_full_workflow()
        
        print("\n🎉 All tests completed successfully!")
        print("✅ Phi-3.5 Mini is ready for your clean room PoC!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 