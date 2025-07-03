#!/usr/bin/env python3
"""Test OpenSearch connectivity and available indices."""

import sys
import os
from pathlib import Path

# Add shared package to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opensearchpy import OpenSearch
from shared.config import app_config

def test_opensearch():
    print('Testing OpenSearch connection...')
    print(f'Connecting to: {app_config.OPENSEARCH_URL}')
    
    client = OpenSearch(
        hosts=[app_config.OPENSEARCH_URL],
        http_auth=None,
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False,
    )

    try:
        if client.ping():
            print('✅ OpenSearch is responding')
            
            # Check available indices
            try:
                indices = client.indices.get_alias(index='*')
                print(f'Available indices: {list(indices.keys())}')
            except Exception as e:
                print(f'Could not list indices: {e}')
            
            # Check recipes index specifically
            try:
                if client.indices.exists(index='recipes'):
                    count = client.count(index='recipes')
                    print(f'Recipes index exists with {count["count"]} documents')
                else:
                    print('❌ Recipes index does not exist')
            except Exception as e:
                print(f'Error checking recipes index: {e}')
                
        else:
            print('❌ OpenSearch ping failed')
            
    except Exception as e:
        print(f'❌ OpenSearch connection error: {e}')
        return False
        
    return True

if __name__ == "__main__":
    test_opensearch() 