import os
import shutil
import numpy as np
import pytest
from adve.search.index import ADVESearchIndex

def test_two_stage_similarity_boost():
    index_dir = "test_two_stage_boost_idx"
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
        
    try:
        # 1. Initialize search index with dim=512 (reconstruction MLP size)
        index = ADVESearchIndex(index_dir, dim=512)
        
        # Create dummy normalized embeddings
        emb1 = np.random.randn(512)
        emb1 /= np.linalg.norm(emb1) + 1e-8
        
        emb2 = np.random.randn(512)
        emb2 /= np.linalg.norm(emb2) + 1e-8
        
        # 2. Add two frames: one with "person, car" and one with "dog, cat"
        index.add(
            video_path="test_video.mp4",
            camera_id="cam_01",
            timestamp=1.0,
            frame_idx=30,
            embedding=emb1,
            is_anchor=True,
            text="person, car"
        )
        
        index.add(
            video_path="test_video.mp4",
            camera_id="cam_01",
            timestamp=2.0,
            frame_idx=60,
            embedding=emb2,
            is_anchor=True,
            text="dog, cat"
        )
        
        # 3. Perform a query where we search using _search with keyword query
        # Find the result for frame_idx=30 and frame_idx=60 using exact match query vector
        results = index._search(emb1, k=5, query="person")
        res30 = next(r for r in results if r.frame_idx == 30)
        res60 = next(r for r in results if r.frame_idx == 60)
        
        # Verify that objects are parsed correctly
        assert sorted(res30.objects) == ["car", "person"]
        assert sorted(res60.objects) == ["cat", "dog"]
        
        # Create a vector v_orth orthogonal to emb1
        v = np.random.randn(512)
        v_orth = v - np.dot(v, emb1) * emb1
        v_orth /= np.linalg.norm(v_orth) + 1e-8
        
        # Search with v_orth, query="person"
        results_orth = index._search(v_orth, k=5, query="person")
        res30_orth = next(r for r in results_orth if r.frame_idx == 30)
        
        # Base similarity of res30_orth should be 0.0 (since orthogonal).
        # Boost should be +0.08, so similarity is 0.08.
        assert abs(res30_orth.similarity - 0.08) < 1e-5
        
        # Search with v_orth, query="person car" (2 keywords matched)
        results_2 = index._search(v_orth, k=5, query="person car")
        res30_2 = next(r for r in results_2 if r.frame_idx == 30)
        
        # Boost should be min(0.15, 0.08 * 2) = 0.15.
        assert abs(res30_2.similarity - 0.15) < 1e-5
        
        print("\n[SUCCESS] Two-Stage Similarity Boost logic test PASSED.")
        
    finally:
        # Close the SQLite database connection so the file lock is released
        if 'index' in locals() and hasattr(index, 'db'):
            try:
                index.db.close()
            except Exception:
                pass
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)

if __name__ == "__main__":
    test_two_stage_similarity_boost()
