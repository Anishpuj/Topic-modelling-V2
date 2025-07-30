#!/usr/bin/env python3
"""
Utility script to force refresh dashboard data and clear caches
"""

import json
import os
import shutil
from datetime import datetime

def sync_data_to_public():
    """Ensure all data files are properly synced to public folder"""
    print("🔄 Syncing Data Files to Public Folder...")
    print("=" * 50)
    
    # Copy enhanced analysis data
    source_dir = "enhanced_analysis_output"
    dest_dir = "public/enhanced_analysis_output"
    
    if os.path.exists(source_dir):
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(source_dir, dest_dir)
        print(f"✅ Copied {source_dir} to {dest_dir}")
    
    # Copy intelligent analysis data if exists
    source_dir = "intelligent_analysis_output"
    dest_dir = "public/intelligent_analysis_output"
    
    if os.path.exists(source_dir):
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(source_dir, dest_dir)
        print(f"✅ Copied {source_dir} to {dest_dir}")
    
    print("📁 Data sync completed!")
    print()

def update_manifest_timestamp():
    """Update manifest timestamp to force refresh"""
    print("⏰ Updating Manifest Timestamps...")
    print("=" * 50)
    
    manifest_files = [
        "public/enhanced_analysis_output/analysis_manifest.json",
        "enhanced_analysis_output/analysis_manifest.json"
    ]
    
    current_time = datetime.now().isoformat()
    
    for manifest_file in manifest_files:
        if os.path.exists(manifest_file):
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            
            manifest['timestamp'] = current_time
            manifest['cache_buster'] = int(datetime.now().timestamp())
            
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            print(f"✅ Updated {manifest_file}")
    
    print()

def generate_dashboard_info():
    """Generate useful dashboard information"""
    print("📊 Dashboard Information...")
    print("=" * 50)
    
    # Read manifest
    manifest_file = "public/enhanced_analysis_output/analysis_manifest.json"
    if os.path.exists(manifest_file):
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        print(f"🔸 Products Available: {len(manifest.get('products', []))}")
        for product in manifest.get('products', []):
            print(f"  - {product}")
        
        print(f"🔸 Last Analysis: {manifest.get('timestamp', 'N/A')}")
        print(f"🔸 Model: {manifest.get('model_info', {}).get('llm_model', 'N/A')}")
    
    print(f"🔸 Dashboard URL: http://localhost:3000")
    print()

def create_cache_buster_html():
    """Create a simple HTML file to test data loading"""
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard Data Test</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .test-button {{ padding: 10px 20px; margin: 10px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }}
        .result {{ margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Dashboard Data Tester</h1>
    <p>Use this page to test if your dashboard data is loading correctly.</p>
    
    <button class="test-button" onclick="testManifest()">Test Manifest</button>
    <button class="test-button" onclick="testProductData()">Test Product Data</button>
    <button class="test-button" onclick="clearCache()">Clear Cache & Reload</button>
    
    <div id="results"></div>
    
    <script>
        const timestamp = Date.now();
        
        async function testManifest() {{
            try {{
                const response = await fetch(`/enhanced_analysis_output/analysis_manifest.json?t=${{timestamp}}`);
                const data = await response.json();
                displayResult('Manifest Test', 'success', JSON.stringify(data, null, 2));
            }} catch (error) {{
                displayResult('Manifest Test', 'error', error.message);
            }}
        }}
        
        async function testProductData() {{
            try {{
                const response = await fetch(`/enhanced_analysis_output/Desk/enhanced_analysis_Desk.json?t=${{timestamp}}`);
                const data = await response.json();
                displayResult('Product Data Test', 'success', `Loaded data for ${{data.product_name}} with ${{data.overall_stats?.total_reviews || 0}} reviews`);
            }} catch (error) {{
                displayResult('Product Data Test', 'error', error.message);
            }}
        }}
        
        function clearCache() {{
            if ('caches' in window) {{
                caches.keys().then(names => {{
                    names.forEach(name => caches.delete(name));
                }});
            }}
            location.reload(true);
        }}
        
        function displayResult(test, status, message) {{
            const results = document.getElementById('results');
            const resultDiv = document.createElement('div');
            resultDiv.className = 'result';
            resultDiv.style.background = status === 'success' ? '#d4edda' : '#f8d7da';
            resultDiv.innerHTML = `<strong>${{test}}:</strong> ${{status === 'success' ? '✅' : '❌'}} ${{message}}`;
            results.appendChild(resultDiv);
        }}
    </script>
</body>
</html>
    """
    
    with open("public/test-data.html", "w", encoding='utf-8') as f:
        f.write(html_content)
    
    print("🧪 Created test page: http://localhost:3000/test-data.html")
    print()

def main():
    """Main function"""
    print("🚀 Dashboard Refresh Tool")
    print("=" * 60)
    print()
    
    sync_data_to_public()
    update_manifest_timestamp()
    generate_dashboard_info()
    create_cache_buster_html()
    
    print("✨ Dashboard refresh completed!")
    print()
    print("📋 Next Steps:")
    print("1. Open http://localhost:3000 in your browser")
    print("2. Clear browser cache (Ctrl+F5 or Cmd+Shift+R)")
    print("3. If issues persist, try incognito/private mode")
    print("4. Use the test page: http://localhost:3000/test-data.html")
    print("5. Check browser console for any errors")

if __name__ == "__main__":
    main()
