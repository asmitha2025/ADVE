"""
ADVE v2 Test Suite
Run: pytest tests/ -v
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Spatial Graph Tests ───────────────────────────────────────────────────────

class TestSpatialGraph:
    def setup_method(self):
        from adve.core.spatial_graph import SpatialGraph, ObjectState

        self.SpatialGraph  = SpatialGraph
        self.ObjectState   = ObjectState

    def make_object(self, obj_id, cx, cy, area=1000, emb=None):
        emb = emb if emb is not None else np.random.randn(512).astype(np.float32)
        return self.ObjectState(
            obj_id=obj_id, class_name="person",
            bbox=(int(cx-20), int(cy-30), int(cx+20), int(cy+30)),
            center=(cx, cy), area=area, embedding=emb,
        )

    def test_build_relations_two_objects(self):
        g = self.SpatialGraph()
        g.add_object(self.make_object(1, 100, 100))
        g.add_object(self.make_object(2, 300, 100))
        g.build_relations()

        assert (1, 2) in g.relations
        assert abs(g.relations[(1, 2)].distance - 200.0) < 1.0

    def test_delta_zero_when_identical(self):
        g = self.SpatialGraph()
        g.add_object(self.make_object(1, 100, 100))
        g.add_object(self.make_object(2, 300, 200))
        g.build_relations()

        delta = g.compute_delta(g)
        assert delta["total_magnitude"] < 1e-6

    def test_delta_detects_movement(self):
        g1 = self.SpatialGraph()
        g1.add_object(self.make_object(1, 100, 100))
        g1.add_object(self.make_object(2, 300, 100))
        g1.build_relations()

        g2 = self.SpatialGraph()
        g2.add_object(self.make_object(1, 100, 100))
        g2.add_object(self.make_object(2, 400, 100))  # obj2 moved right
        g2.build_relations()

        delta = g1.compute_delta(g2)
        assert delta["total_magnitude"] > 0.1

    def test_new_object_detected(self):
        g1 = self.SpatialGraph()
        g1.add_object(self.make_object(1, 100, 100))
        g1.build_relations()

        g2 = self.SpatialGraph()
        g2.add_object(self.make_object(1, 100, 100))
        g2.add_object(self.make_object(2, 300, 200))  # new object
        g2.build_relations()

        delta = g1.compute_delta(g2)
        assert 2 in delta["new_objects"]

    def test_lost_object_detected(self):
        g1 = self.SpatialGraph()
        g1.add_object(self.make_object(1, 100, 100))
        g1.add_object(self.make_object(2, 300, 200))
        g1.build_relations()

        g2 = self.SpatialGraph()
        g2.add_object(self.make_object(1, 100, 100))  # obj2 gone
        g2.build_relations()

        delta = g1.compute_delta(g2)
        assert 2 in delta["lost_objects"]


# ── Reconstructor Tests ───────────────────────────────────────────────────────

class TestReconstructor:
    def setup_method(self):
        from adve.core.reconstructor import EmbeddingReconstructor
        from adve.core.spatial_graph import SpatialGraph, ObjectState

        self.reconstructor = EmbeddingReconstructor()
        self.SpatialGraph  = SpatialGraph
        self.ObjectState   = ObjectState

    def _make_graph_with_emb(self, positions, embeddings):
        g = self.SpatialGraph()
        for i, ((cx, cy), emb) in enumerate(zip(positions, embeddings)):
            g.add_object(self.ObjectState(
                obj_id=i, class_name="person",
                bbox=(int(cx-20), int(cy-20), int(cx+20), int(cy+20)),
                center=(cx, cy), area=1000.0, embedding=emb,
            ))
        g.build_relations()
        return g

    def test_output_is_unit_normalized(self):
        embs  = [np.random.randn(512).astype(np.float32) for _ in range(3)]
        g     = self._make_graph_with_emb([(100,100),(200,200),(300,100)], embs)
        delta = g.compute_delta(g)
        anchor_emb = np.random.randn(512).astype(np.float32)
        anchor_emb /= np.linalg.norm(anchor_emb)

        result = self.reconstructor.reconstruct(g, g, delta, anchor_emb)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_static_scene_returns_close_to_anchor(self):
        """When ΔG ≈ 0, result should be very close to anchor embedding."""
        embs  = [np.random.randn(512).astype(np.float32) for _ in range(2)]
        g     = self._make_graph_with_emb([(100,100),(300,100)], embs)
        delta = {"total_magnitude": 0.0, "relation_deltas": {}, "new_objects": [], "lost_objects": []}
        anchor_emb = np.random.randn(512).astype(np.float32)
        anchor_emb /= np.linalg.norm(anchor_emb)

        result = self.reconstructor.reconstruct(g, g, delta, anchor_emb)
        sim = float(np.dot(result, anchor_emb))
        assert sim > 0.95, f"Expected sim>0.95, got {sim:.4f}"

    def test_no_objects_returns_anchor(self):
        """If no objects have embeddings, return anchor unchanged."""
        g = self.SpatialGraph()
        delta = {"total_magnitude": 0.0, "relation_deltas": {}, "new_objects": [], "lost_objects": []}
        anchor_emb = np.random.randn(512).astype(np.float32)
        anchor_emb /= np.linalg.norm(anchor_emb)

        result = self.reconstructor.reconstruct(g, g, delta, anchor_emb)
        assert np.allclose(result, anchor_emb)


# ── Search Index Tests ────────────────────────────────────────────────────────

class TestSearchIndex:
    def setup_method(self, tmp_path=None):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        from adve.search.index import ADVESearchIndex
        self.index = ADVESearchIndex(self.tmpdir)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_and_stats(self):
        emb = np.random.randn(512).astype(np.float32)
        self.index.add("video1.mp4", "cam1", 1.5, 45, emb)
        stats = self.index.stats()
        assert stats["total_embeddings"] == 1

    def test_search_returns_results(self):
        for i in range(10):
            emb = np.random.randn(512).astype(np.float32)
            self.index.add("video1.mp4", "cam1", float(i), i*30, emb)

        query = np.random.randn(512).astype(np.float32)
        results = self.index._search(query, k=5)
        assert len(results) == 5

    def test_batch_add(self):
        records = [
            {
                "video_path": "v.mp4",
                "camera_id":  "c1",
                "timestamp":  float(i),
                "frame_idx":  i,
                "embedding":  np.random.randn(512).astype(np.float32),
                "is_anchor":  i % 30 == 0,
            }
            for i in range(100)
        ]
        self.index.add_batch(records)
        stats = self.index.stats()
        assert stats["total_embeddings"] == 100
        assert stats["anchor_frames"]    >  0


# ── Validator Tests ───────────────────────────────────────────────────────────

class TestValidator:
    def setup_method(self):
        from adve.core.validator import Validator
        from adve.core.config    import Config
        self.validator = Validator(Config())

    def test_cosine_sim_identical(self):
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        sim = self.validator.log(0, emb, emb, True, 0.0, True)
        assert abs(sim - 1.0) < 1e-5

    def test_summary_savings_calculation(self):
        for i in range(10):
            emb = np.random.randn(512).astype(np.float32)
            emb /= np.linalg.norm(emb)
            self.validator.log(i, emb, emb, i == 0, 0.0, i == 0)

        summary = self.validator.summarize()
        assert summary["encoder_savings_pct"] == pytest.approx(90.0)
        assert summary["total_frames"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
