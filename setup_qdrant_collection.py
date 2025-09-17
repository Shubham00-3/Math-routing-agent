#!/usr/bin/env python3
"""
Script to create the Qdrant collection for the knowledge base
"""

import sys
from pathlib import Path

# Add backend to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "backend"))

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

def setup_qdrant_collection():
    """Create the math_knowledge_base collection in Qdrant"""
    
    print("Setting up Qdrant collection...")
    
    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)
    
    collection_name = "math_knowledge_base"
    
    # Check if collection exists
    try:
        collections = client.get_collections()
        existing_collections = [col.name for col in collections.collections]
        
        if collection_name in existing_collections:
            print(f"✅ Collection '{collection_name}' already exists!")
            return True
            
    except Exception as e:
        print(f"Error checking collections: {e}")
    
    # Create collection
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384,  # sentence-transformers/all-MiniLM-L6-v2 embedding size
                distance=Distance.COSINE
            )
        )
        print(f"✅ Created collection '{collection_name}' successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        return False

if __name__ == "__main__":
    success = setup_qdrant_collection()
    if success:
        print("🎉 Qdrant setup complete!")
    else:
        print("💥 Qdrant setup failed!")
        sys.exit(1)