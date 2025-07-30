#!/usr/bin/env python3
"""
Simple script to check if dashboard data is accessible and up-to-date
"""

import json
import os
from datetime import datetime
from pathlib import Path

def check_data_files():
    """Check if all required data files exist and are up-to-date"""
    print("🔍 Checking Dashboard Data Files...")
    print("=" * 50)
    
    # Check manifest files
    manifest_files = [
        "public/enhanced_analysis_output/analysis_manifest.json",
        "enhanced_analysis_output/analysis_manifest.json"
    ]
    
    for manifest_file in manifest_files:
        if os.path.exists(manifest_file):
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
                print(f"✅ {manifest_file}")
                print(f"   - Timestamp: {manifest.get('timestamp', 'N/A')}")
                print(f"   - Products: {len(manifest.get('products', []))}")
                
                # Check individual product files
                for product in manifest.get('products', []):
                    product_file = f"public/enhanced_analysis_output/{product}/enhanced_analysis_{product}.json"
                    if os.path.exists(product_file):
                        with open(product_file, 'r') as pf:
                            product_data = json.load(pf)
                            print(f"   ✅ {product}: {product_data.get('analysis_timestamp', 'N/A')}")
                    else:
                        print(f"   ❌ {product}: File not found")
        else:
            print(f"❌ {manifest_file}: Not found")
        print()

def check_server_status():
    """Check if the React development server is running"""
    print("🌐 Server Status Check...")
    print("=" * 50)
    
    try:
        import requests
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("✅ React development server is running on http://localhost:3000")
        else:
            print(f"⚠️  Server responded with status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to React server: {e}")
        print("💡 Try running: npm start")
    except ImportError:
        print("⚠️  'requests' module not available. Cannot check server status.")
    print()

def generate_cache_busting_urls():
    """Generate cache-busting URLs for testing"""
    print("🔄 Cache-Busting URLs for Testing...")
    print("=" * 50)
    
    timestamp = int(datetime.now().timestamp())
    
    urls = [
        f"http://localhost:3000/enhanced_analysis_output/analysis_manifest.json?t={timestamp}",
        f"http://localhost:3000/enhanced_analysis_output/Desk/enhanced_analysis_Desk.json?t={timestamp}",
        f"http://localhost:3000/enhanced_analysis_output/Iphone16/enhanced_analysis_Iphone16.json?t={timestamp}",
    ]
    
    print("Test these URLs in your browser to check if data is loading:")
    for url in urls:
        print(f"- {url}")
    print()

def main():
    """Main function"""
    print("🚀 Dashboard Data Check Tool")
    print("=" * 50)
    print()
    
    check_data_files()
    check_server_status()
    generate_cache_busting_urls()
    
    print("💡 Troubleshooting Tips:")
    print("1. Clear your browser cache (Ctrl+F5 or Cmd+Shift+R)")
    print("2. Check browser console for any errors")
    print("3. Ensure React development server is running (npm start)")
    print("4. Try opening browser in incognito/private mode")
    print("5. Check if data files have recent timestamps")

if __name__ == "__main__":
    main()
