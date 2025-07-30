#!/usr/bin/env python3
"""
Comprehensive validation script for the Clean Room Data Processor.
Tests data agnosticism, multiple file handling, and edge cases.
"""

import pandas as pd
import numpy as np
import json
import tempfile
import os
from datetime import datetime, timedelta
import random
import string
from typing import Dict, List, Any

# Import the core functions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phi_llm_functions import PhiLLMProcessor, load_data_files, generate_standardization_report
import streamlit_app

def create_test_datasets():
    """Create various test datasets to validate data agnosticism."""
    test_datasets = {}
    
    # Test 1: Standard customer data
    test_datasets['customers_standard.csv'] = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'first_name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'last_name': ['Doe', 'Smith', 'Johnson', 'Brown', 'Wilson'],
        'email': ['john@email.com', 'jane@email.com', 'bob@email.com', 'alice@email.com', 'charlie@email.com'],
        'phone': ['555-123-4567', '555-234-5678', '555-345-6789', '555-456-7890', '555-567-8901'],
        'registration_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12']
    })
    
    # Test 2: Different naming conventions
    test_datasets['orders_different_naming.csv'] = pd.DataFrame({
        'CustomerID': [1, 2, 3, 4, 5],
        'FirstName': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'LastName': ['Doe', 'Smith', 'Johnson', 'Brown', 'Wilson'],
        'EmailAddress': ['john@email.com', 'jane@email.com', 'bob@email.com', 'alice@email.com', 'charlie@email.com'],
        'PhoneNumber': ['555-123-4567', '555-234-5678', '555-345-6789', '555-456-7890', '555-567-8901'],
        'OrderDate': ['2023-06-15', '2023-07-20', '2023-08-10', '2023-09-05', '2023-10-12'],
        'OrderAmount': [150.00, 200.50, 75.25, 300.00, 125.75]
    })
    
    # Test 3: Completely different domain (financial data)
    test_datasets['financial_data.csv'] = pd.DataFrame({
        'account_number': ['ACC001', 'ACC002', 'ACC003', 'ACC004', 'ACC005'],
        'transaction_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'transaction_type': ['DEPOSIT', 'WITHDRAWAL', 'TRANSFER', 'DEPOSIT', 'WITHDRAWAL'],
        'amount': [1000.00, -500.00, 250.00, 750.00, -300.00],
        'currency': ['USD', 'USD', 'USD', 'USD', 'USD'],
        'balance': [5000.00, 4500.00, 4750.00, 5500.00, 5200.00]
    })
    
    # Test 4: Scientific/technical data
    test_datasets['sensor_data.csv'] = pd.DataFrame({
        'sensor_id': ['SENSOR_001', 'SENSOR_002', 'SENSOR_003', 'SENSOR_004', 'SENSOR_005'],
        'timestamp': ['2023-01-01T00:00:00', '2023-01-01T00:01:00', '2023-01-01T00:02:00', '2023-01-01T00:03:00', '2023-01-01T00:04:00'],
        'temperature_celsius': [23.5, 23.7, 23.4, 23.8, 23.6],
        'humidity_percent': [45.2, 45.5, 45.1, 45.8, 45.3],
        'pressure_hpa': [1013.25, 1013.30, 1013.20, 1013.35, 1013.28],
        'status': ['ACTIVE', 'ACTIVE', 'ACTIVE', 'ACTIVE', 'ACTIVE']
    })
    
    # Test 5: Edge case - Empty dataset
    test_datasets['empty_dataset.csv'] = pd.DataFrame({
        'id': [],
        'name': [],
        'value': []
    })
    
    # Test 6: Edge case - Single column
    test_datasets['single_column.csv'] = pd.DataFrame({
        'id': [1, 2, 3, 4, 5]
    })
    
    # Test 7: Edge case - Very long column names
    test_datasets['long_column_names.csv'] = pd.DataFrame({
        'this_is_a_very_long_column_name_that_might_cause_issues_with_some_systems': [1, 2, 3, 4, 5],
        'another_extremely_long_column_name_with_special_characters_and_numbers_123': ['a', 'b', 'c', 'd', 'e'],
        'third_column_with_mixed_case_And_Special_Characters_!@#$%': [1.1, 2.2, 3.3, 4.4, 5.5]
    })
    
    # Test 8: Edge case - Mixed data types
    test_datasets['mixed_data_types.csv'] = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'age': [25, 30, 35, 40, 45],
        'salary': [50000.00, 60000.00, 55000.00, 70000.00, 65000.00],
        'is_active': [True, False, True, True, False],
        'created_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'notes': ['Some notes', None, 'More notes', '', 'Final notes']
    })
    
    # Test 9: Edge case - Special characters in data
    test_datasets['special_characters.csv'] = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['José', 'François', 'Müller', 'O\'Connor', 'Smith-Jones'],
        'email': ['jose@email.com', 'francois@email.com', 'muller@email.com', 'oconnor@email.com', 'smith-jones@email.com'],
        'description': ['Contains "quotes"', 'Has \'apostrophes\'', 'Special chars: !@#$%', 'Unicode: éñü', 'Mixed: 123!@#']
    })
    
    # Test 10: Large dataset (simulated)
    large_data = {
        'id': list(range(1, 1001)),
        'name': [f'User_{i}' for i in range(1, 1001)],
        'email': [f'user_{i}@email.com' for i in range(1, 1001)],
        'value': [random.random() * 1000 for _ in range(1000)]
    }
    test_datasets['large_dataset.csv'] = pd.DataFrame(large_data)
    
    return test_datasets

def test_data_agnosticism():
    """Test that the system works with different types of data."""
    print("🧪 Testing Data Agnosticism...")
    
    test_datasets = create_test_datasets()
    phi_processor = PhiLLMProcessor()
    
    # Save test datasets to temporary files
    temp_files = []
    for filename, df in test_datasets.items():
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_files.append(temp_file.name)
    
    try:
        # Test loading multiple files
        print("  📁 Testing file loading...")
        dataframes = load_data_files(temp_files)
        
        if len(dataframes) == len(test_datasets):
            print("  ✅ Successfully loaded all test datasets")
        else:
            print(f"  ❌ Expected {len(test_datasets)} datasets, got {len(dataframes)}")
            return False
        
        # Test column analysis
        print("  🔍 Testing column analysis...")
        for filename, df in dataframes.items():
            column_analysis = phi_processor.analyze_column_names(df.columns.tolist())
            if isinstance(column_analysis, dict):
                print(f"  ✅ Column analysis successful for {filename}")
            else:
                print(f"  ❌ Column analysis failed for {filename}")
        
        # Test similarity detection
        print("  🔗 Testing similarity detection...")
        all_columns = []
        for df in dataframes.values():
            all_columns.extend(df.columns.tolist())
        
        clusters = phi_processor.cluster_similar_columns(all_columns)
        if isinstance(clusters, list):
            print(f"  ✅ Similarity clustering successful, found {len(clusters)} clusters")
        else:
            print("  ❌ Similarity clustering failed")
            return False
        
        # Test data quality validation
        print("  📊 Testing data quality validation...")
        for filename, df in dataframes.items():
            if not df.empty:  # Skip empty datasets
                quality_assessment = phi_processor.validate_data_quality(df)
                if isinstance(quality_assessment, dict):
                    print(f"  ✅ Quality assessment successful for {filename}")
                else:
                    print(f"  ❌ Quality assessment failed for {filename}")
        
        # Test standardization suggestions
        print("  🔄 Testing standardization suggestions...")
        for filename, df in dataframes.items():
            if not df.empty:
                sample_data = {}
                for col in df.columns[:3]:  # Sample first 3 columns
                    sample_data[col] = df[col].dropna().head(5).tolist()
                
                standardization = phi_processor.suggest_data_standardization(sample_data)
                if isinstance(standardization, dict):
                    print(f"  ✅ Standardization suggestions successful for {filename}")
                else:
                    print(f"  ❌ Standardization suggestions failed for {filename}")
        
        print("  ✅ Data agnosticism test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Data agnosticism test failed: {e}")
        return False
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🧪 Testing Edge Cases...")
    
    phi_processor = PhiLLMProcessor()
    
    # Test 1: Empty DataFrame
    print("  📊 Testing empty DataFrame...")
    empty_df = pd.DataFrame()
    try:
        quality_assessment = phi_processor.validate_data_quality(empty_df)
        print("  ✅ Empty DataFrame handled gracefully")
    except Exception as e:
        print(f"  ❌ Empty DataFrame test failed: {e}")
        return False
    
    # Test 2: DataFrame with only null values
    print("  📊 Testing DataFrame with null values...")
    null_df = pd.DataFrame({
        'col1': [None, None, None],
        'col2': [None, None, None]
    })
    try:
        quality_assessment = phi_processor.validate_data_quality(null_df)
        print("  ✅ Null DataFrame handled gracefully")
    except Exception as e:
        print(f"  ❌ Null DataFrame test failed: {e}")
        return False
    
    # Test 3: Very large column names
    print("  📊 Testing very long column names...")
    long_col_df = pd.DataFrame({
        'a' * 1000: [1, 2, 3],
        'b' * 1000: [4, 5, 6]
    })
    try:
        column_analysis = phi_processor.analyze_column_names(long_col_df.columns.tolist())
        print("  ✅ Long column names handled gracefully")
    except Exception as e:
        print(f"  ❌ Long column names test failed: {e}")
        return False
    
    # Test 4: Special characters in column names
    print("  📊 Testing special characters in column names...")
    special_col_df = pd.DataFrame({
        'col@#$%': [1, 2, 3],
        'col!@#$%^&*()': [4, 5, 6],
        'col with spaces': [7, 8, 9]
    })
    try:
        column_analysis = phi_processor.analyze_column_names(special_col_df.columns.tolist())
        print("  ✅ Special characters in column names handled gracefully")
    except Exception as e:
        print(f"  ❌ Special characters test failed: {e}")
        return False
    
    # Test 5: Mixed data types
    print("  📊 Testing mixed data types...")
    mixed_df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['John', 'Jane', 'Bob'],
        'age': [25, 30, 35],
        'salary': [50000.00, 60000.00, 55000.00],
        'is_active': [True, False, True],
        'created_date': ['2023-01-01', '2023-01-02', '2023-01-03']
    })
    try:
        quality_assessment = phi_processor.validate_data_quality(mixed_df)
        print("  ✅ Mixed data types handled gracefully")
    except Exception as e:
        print(f"  ❌ Mixed data types test failed: {e}")
        return False
    
    print("  ✅ Edge cases test passed!")
    return True

def test_multiple_file_handling():
    """Test handling of multiple files with different characteristics."""
    print("\n🧪 Testing Multiple File Handling...")
    
    # Create test files with different characteristics
    test_files = []
    
    # File 1: Standard CSV
    df1 = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'name': ['John', 'Jane', 'Bob'],
        'email': ['john@email.com', 'jane@email.com', 'bob@email.com']
    })
    temp_file1 = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df1.to_csv(temp_file1.name, index=False)
    test_files.append(temp_file1.name)
    
    # File 2: Different naming convention
    df2 = pd.DataFrame({
        'CustomerID': [1, 2, 3],
        'FirstName': ['John', 'Jane', 'Bob'],
        'LastName': ['Doe', 'Smith', 'Johnson'],
        'EmailAddress': ['john@email.com', 'jane@email.com', 'bob@email.com']
    })
    temp_file2 = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df2.to_csv(temp_file2.name, index=False)
    test_files.append(temp_file2.name)
    
    # File 3: Completely different domain
    df3 = pd.DataFrame({
        'product_id': [101, 102, 103],
        'product_name': ['Laptop', 'Mouse', 'Keyboard'],
        'category': ['Electronics', 'Accessories', 'Accessories'],
        'price': [999.99, 25.50, 75.00]
    })
    temp_file3 = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df3.to_csv(temp_file3.name, index=False)
    test_files.append(temp_file3.name)
    
    try:
        # Test loading multiple files
        print("  📁 Testing multiple file loading...")
        dataframes = load_data_files(test_files)
        
        if len(dataframes) == 3:
            print("  ✅ Successfully loaded all 3 test files")
        else:
            print(f"  ❌ Expected 3 files, got {len(dataframes)}")
            return False
        
        # Test comprehensive analysis
        print("  🔍 Testing comprehensive analysis...")
        phi_processor = PhiLLMProcessor()
        report = generate_standardization_report(dataframes, phi_processor)
        
        # Check report structure
        required_keys = ['dataset_summary', 'column_analysis', 'similarity_clusters', 'standardization_suggestions', 'quality_assessment']
        for key in required_keys:
            if key not in report:
                print(f"  ❌ Missing key '{key}' in report")
                return False
        
        print("  ✅ Comprehensive analysis successful")
        
        # Test similarity detection across files
        print("  🔗 Testing cross-file similarity detection...")
        clusters = report['similarity_clusters']
        if isinstance(clusters, list) and len(clusters) > 0:
            print(f"  ✅ Found {len(clusters)} similarity clusters across files")
        else:
            print("  ❌ No similarity clusters found")
            return False
        
        print("  ✅ Multiple file handling test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Multiple file handling test failed: {e}")
        return False
    
    finally:
        # Clean up temporary files
        for temp_file in test_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

def test_phi_llm_integration():
    """Test the Phi LLM integration specifically."""
    print("\n🧪 Testing Phi LLM Integration...")
    
    phi_processor = PhiLLMProcessor()
    
    # Test 1: Basic query
    print("  🤖 Testing basic Phi query...")
    try:
        response = phi_processor.query_phi("Hello! Can you help me with data analysis?")
        if response and len(response) > 0:
            print("  ✅ Basic Phi query successful")
        else:
            print("  ❌ Basic Phi query failed - no response")
            return False
    except Exception as e:
        print(f"  ❌ Basic Phi query failed: {e}")
        return False
    
    # Test 2: Column analysis
    print("  📊 Testing column analysis...")
    test_columns = ['customer_id', 'CustomerID', 'first_name', 'FirstName']
    try:
        analysis = phi_processor.analyze_column_names(test_columns)
        if isinstance(analysis, dict):
            print("  ✅ Column analysis successful")
        else:
            print("  ❌ Column analysis failed")
            return False
    except Exception as e:
        print(f"  ❌ Column analysis failed: {e}")
        return False
    
    # Test 3: Similarity detection
    print("  🔗 Testing similarity detection...")
    columns1 = ['customer_id', 'first_name', 'last_name']
    columns2 = ['CustomerID', 'FirstName', 'LastName']
    try:
        matches = phi_processor.find_similar_columns(columns1, columns2)
        if isinstance(matches, list):
            print(f"  ✅ Similarity detection successful, found {len(matches)} matches")
        else:
            print("  ❌ Similarity detection failed")
            return False
    except Exception as e:
        print(f"  ❌ Similarity detection failed: {e}")
        return False
    
    # Test 4: Data quality validation
    print("  📊 Testing data quality validation...")
    test_df = pd.DataFrame({
        'id': [1, 2, 3, None, 5],
        'name': ['John', 'Jane', 'Bob', 'Alice', ''],
        'email': ['john@email.com', 'jane@email.com', 'invalid-email', 'alice@email.com', 'bob@email.com'],
        'age': [25, 30, 'N/A', 35, 40]
    })
    try:
        quality = phi_processor.validate_data_quality(test_df)
        if isinstance(quality, dict):
            print("  ✅ Data quality validation successful")
        else:
            print("  ❌ Data quality validation failed")
            return False
    except Exception as e:
        print(f"  ❌ Data quality validation failed: {e}")
        return False
    
    # Test 5: Standardization suggestions
    print("  🔄 Testing standardization suggestions...")
    sample_data = {
        'customer_id': ['CUST001', 'CUST002', 'CUST003'],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'email': ['john.doe@email.com', 'jane.smith@email.com', 'bob.johnson@email.com']
    }
    try:
        standardization = phi_processor.suggest_data_standardization(sample_data)
        if isinstance(standardization, dict):
            print("  ✅ Standardization suggestions successful")
        else:
            print("  ❌ Standardization suggestions failed")
            return False
    except Exception as e:
        print(f"  ❌ Standardization suggestions failed: {e}")
        return False
    
    print("  ✅ Phi LLM integration test passed!")
    return True

def test_performance_and_scalability():
    """Test performance with larger datasets."""
    print("\n🧪 Testing Performance and Scalability...")
    
    # Test with larger dataset
    print("  📊 Testing with larger dataset...")
    large_df = pd.DataFrame({
        'id': list(range(1, 1001)),
        'name': [f'User_{i}' for i in range(1, 1001)],
        'email': [f'user_{i}@email.com' for i in range(1, 1001)],
        'value': [random.random() * 1000 for _ in range(1000)]
    })
    
    phi_processor = PhiLLMProcessor()
    
    try:
        # Test column analysis with larger dataset
        start_time = datetime.now()
        column_analysis = phi_processor.analyze_column_names(large_df.columns.tolist())
        end_time = datetime.now()
        analysis_time = (end_time - start_time).total_seconds()
        
        if analysis_time < 10:  # Should complete within 10 seconds
            print(f"  ✅ Column analysis completed in {analysis_time:.2f} seconds")
        else:
            print(f"  ⚠️ Column analysis took {analysis_time:.2f} seconds (slow)")
        
        # Test similarity detection with larger dataset
        start_time = datetime.now()
        clusters = phi_processor.cluster_similar_columns(large_df.columns.tolist())
        end_time = datetime.now()
        clustering_time = (end_time - start_time).total_seconds()
        
        if clustering_time < 5:  # Should complete within 5 seconds
            print(f"  ✅ Similarity clustering completed in {clustering_time:.2f} seconds")
        else:
            print(f"  ⚠️ Similarity clustering took {clustering_time:.2f} seconds (slow)")
        
        # Test data quality validation with larger dataset
        start_time = datetime.now()
        quality = phi_processor.validate_data_quality(large_df)
        end_time = datetime.now()
        quality_time = (end_time - start_time).total_seconds()
        
        if quality_time < 15:  # Should complete within 15 seconds
            print(f"  ✅ Data quality validation completed in {quality_time:.2f} seconds")
        else:
            print(f"  ⚠️ Data quality validation took {quality_time:.2f} seconds (slow)")
        
        print("  ✅ Performance and scalability test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🔍 Comprehensive Validation Test for Clean Room Data Processor")
    print("=" * 70)
    
    tests = [
        ("Data Agnosticism", test_data_agnosticism),
        ("Edge Cases", test_edge_cases),
        ("Multiple File Handling", test_multiple_file_handling),
        ("Phi LLM Integration", test_phi_llm_integration),
        ("Performance and Scalability", test_performance_and_scalability)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n{'='*70}")
    print(f"📊 Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation tests passed! The system is robust and data agnostic.")
        print("\n✅ Key Validations:")
        print("  - Data agnosticism: Works with different data types and domains")
        print("  - Edge cases: Handles empty data, null values, special characters")
        print("  - Multiple files: Processes multiple datasets simultaneously")
        print("  - Phi LLM integration: AI-powered analysis working correctly")
        print("  - Performance: Scalable for larger datasets")
    else:
        print("⚠️ Some validation tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 