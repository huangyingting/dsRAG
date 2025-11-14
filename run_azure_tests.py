#!/usr/bin/env python3
"""
Test runner for Azure integration components.

This script provides a convenient way to run Azure tests with different options.
"""

import sys
import os
import subprocess
import argparse


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def check_syntax():
    """Check Python syntax of Azure module files."""
    print_header("Checking Python Syntax")
    
    azure_files = [
        "dsrag/azure/__init__.py",
        "dsrag/azure/blob_storage.py",
        "dsrag/azure/azure_openai_chat.py",
        "dsrag/azure/azure_openai_embedding.py",
        "dsrag/azure/azure_openai_vlm.py",
    ]
    
    try:
        cmd = ["python3", "-m", "py_compile"] + azure_files
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ All Azure module files have valid Python syntax")
            return True
        else:
            print("✗ Syntax errors found:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ Error checking syntax: {e}")
        return False


def run_unit_tests():
    """Run unit tests for Azure components."""
    print_header("Running Unit Tests")
    
    try:
        cmd = ["python3", "tests/unit/test_azure_blob_storage.py"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✓ All unit tests passed")
            return True
        else:
            print("\n✗ Some unit tests failed")
            return False
    except Exception as e:
        print(f"✗ Error running unit tests: {e}")
        return False


def run_integration_tests():
    """Run integration tests for Azure components."""
    print_header("Running Integration Tests")
    
    # Check for required environment variables
    required_vars = [
        "AZURE_STORAGE_CONTAINER_NAME",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print("✗ Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables and try again.")
        return False
    
    # Check for storage credentials
    has_connection_string = bool(os.environ.get("AZURE_STORAGE_CONNECTION_STRING"))
    has_account_creds = bool(
        os.environ.get("AZURE_STORAGE_ACCOUNT_NAME") and 
        os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
    )
    
    if not has_connection_string and not has_account_creds:
        print("✗ Missing Azure Storage credentials")
        print("  Set either AZURE_STORAGE_CONNECTION_STRING")
        print("  or both AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY")
        return False
    
    try:
        cmd = ["python3", "tests/integration/test_azure_integration.py"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✓ All integration tests passed")
            return True
        else:
            print("\n✗ Some integration tests failed")
            return False
    except Exception as e:
        print(f"✗ Error running integration tests: {e}")
        return False


def run_example():
    """Run the Azure example."""
    print_header("Running Azure Example")
    
    try:
        cmd = ["python3", "examples/azure_example.py"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✓ Example completed successfully")
            return True
        else:
            print("\n✗ Example failed")
            return False
    except Exception as e:
        print(f"✗ Error running example: {e}")
        return False


def cleanup_example():
    """Clean up the example knowledge base."""
    print_header("Cleaning Up Example")
    
    try:
        cmd = ["python3", "examples/azure_example.py", "--cleanup"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✓ Cleanup completed successfully")
            return True
        else:
            print("\n✗ Cleanup failed")
            return False
    except Exception as e:
        print(f"✗ Error during cleanup: {e}")
        return False


def check_environment():
    """Check environment setup."""
    print_header("Checking Environment")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check for Azure packages
    print("\nChecking Azure dependencies...")
    
    try:
        import azure.storage.blob
        print("✓ azure-storage-blob is installed")
    except ImportError:
        print("✗ azure-storage-blob is NOT installed")
        print("  Install with: pip install 'dsrag[azure-storage]'")
    
    try:
        import openai
        print("✓ openai is installed")
    except ImportError:
        print("✗ openai is NOT installed")
        print("  Install with: pip install 'dsrag[azure-openai]'")
    
    # Check environment variables
    print("\nChecking environment variables...")
    
    env_vars = [
        "AZURE_STORAGE_CONNECTION_STRING",
        "AZURE_STORAGE_ACCOUNT_NAME",
        "AZURE_STORAGE_ACCOUNT_KEY",
        "AZURE_STORAGE_CONTAINER_NAME",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    ]
    
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            # Mask sensitive values
            if "KEY" in var or "CONNECTION" in var:
                display_value = value[:10] + "..." if len(value) > 10 else "***"
            else:
                display_value = value
            print(f"✓ {var} = {display_value}")
        else:
            print(f"  {var} = (not set)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner for Azure integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_azure_tests.py --all              # Run all tests
  python3 run_azure_tests.py --syntax           # Check syntax only
  python3 run_azure_tests.py --unit             # Run unit tests
  python3 run_azure_tests.py --integration      # Run integration tests
  python3 run_azure_tests.py --example          # Run example
  python3 run_azure_tests.py --check            # Check environment
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests (syntax, unit, integration)"
    )
    parser.add_argument(
        "--syntax",
        action="store_true",
        help="Check Python syntax"
    )
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests"
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests (requires Azure credentials)"
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run the Azure example"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up example resources"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check environment setup"
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    results = []
    
    if args.check:
        check_environment()
    
    if args.syntax or args.all:
        results.append(("Syntax Check", check_syntax()))
    
    if args.unit or args.all:
        results.append(("Unit Tests", run_unit_tests()))
    
    if args.integration or args.all:
        results.append(("Integration Tests", run_integration_tests()))
    
    if args.example:
        results.append(("Example", run_example()))
    
    if args.cleanup:
        results.append(("Cleanup", cleanup_example()))
    
    # Print summary
    if results:
        print_header("Test Summary")
        for name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{name:25} {status}")
        
        all_passed = all(passed for _, passed in results)
        exit_code = 0 if all_passed else 1
        
        print()
        if all_passed:
            print("All tests passed! ✓")
        else:
            print("Some tests failed. ✗")
        
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
