"""
Debug ESConv data format
"""

import modal
import json

app = modal.App("esconv-debug")

image = modal.Image.debian_slim(python_version="3.11").pip_install("datasets")

@app.function(image=image, timeout=600)
def debug_esconv():
    from datasets import load_dataset
    
    print("Loading ESConv...")
    ds = load_dataset("thu-coai/esconv")
    
    print(f"Keys: {ds.keys()}")
    print(f"Train size: {len(ds['train'])}")
    
    # Look at first few items
    for i in range(3):
        item = ds['train'][i]
        print(f"\n{'='*70}")
        print(f"ITEM {i}")
        print(f"{'='*70}")
        print(f"Keys: {item.keys()}")
        
        for key in item.keys():
            val = item[key]
            if isinstance(val, str):
                print(f"  {key}: {val[:100]}..." if len(val) > 100 else f"  {key}: {val}")
            elif isinstance(val, list):
                print(f"  {key}: list with {len(val)} items")
                if len(val) > 0:
                    print(f"    First item type: {type(val[0])}")
                    if isinstance(val[0], dict):
                        print(f"    First item keys: {val[0].keys()}")
                        print(f"    First item: {val[0]}")
                    else:
                        print(f"    First item: {val[0][:100] if isinstance(val[0], str) and len(val[0]) > 100 else val[0]}")
            else:
                print(f"  {key}: {val}")


@app.local_entrypoint()
def main():
    debug_esconv.remote()
