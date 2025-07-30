#!/usr/bin/env python3
"""
Test runner script for the Clean Room Data Processor.
Runs all tests from the tests directory.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_test(test_file: str, description: str):
    """Run a specific test file."""
    print(f"\n{'='*60}")
    print(f"🧪 Running {description}")
    print(f"{'='*60}")
    
    try:
        # Run the test file
        result = subprocess.run([
            sys.executable, 
            os.path.join(os.path.dirname(__file__), test_file)
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        # Return success status
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running {test_file}: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Clean Room Data Processor - Test Runner")
    print("=" * 60)
    
    # Define tests to run
    tests = [
        ("test_phi_functions.py", "Phi LLM Functions Test"),
        ("test_streamlit_app.py", "Streamlit App Test"),
        ("validation_test.py", "Comprehensive Validation Test")
    ]
    
    passed = 0
    total = len(tests)
    
    for test_file, description in tests:
        if run_test(test_file, description):
            passed += 1
            print(f"✅ {description} PASSED")
        else:
            print(f"❌ {description} FAILED")
    
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready for use.")
        print("\n🚀 To start the application:")
        print("   streamlit run streamlit_app.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 