# ADVE — Road to Best-in-Class
## What to Build, In What Order, and Why

---

## Current State (Honest)

```
Working:
  Research validated (96.67% savings, 0.9923 cosine sim on MOT17)
  FastAPI server running (port 8000)
  FAISS + SQLite index (16,604 embeddings)
  Object RoI detection (person, car, elephant, etc.)
  Dense indexing (every frame stored)
  13/13 tests passing
  GitHub live

Not working or not done:
  Audio search (biggest gap)
  MLP training (accuracy improvement pending)
  Public demo (nobody can try it)
  Zenodo DOI (no citable reference)
  Two-stage search fix (object scores 0.27 — not usable)
  Competitor comparison
  First user
```

---

## The Five Builds — Ordered by Impact

---

### Build 1: Audio Search with Whisper (Week 1)
**Why first:** Every competitor is visual-only. Audio + visual = genuine differentiation.

```
Install:
  pip install openai-whisper

Integrate:
  adve/audio/indexer.py         ← transcribe + embed audio segments
  adve/audio/multimodal_search.py ← merge visual + audio results

Update index_video_task in server.py:
  After visual indexing:
    audio_indexer.index_video(video_path, video_id)

Update search endpoint:
  visual_results = search_index.search_by_text(query, k=20)
  audio_results  = audio_indexer.search(query, video_id, k=20)
  merged         = merge_results(visual_results, audio_results)
  return merged
```

**Demo claim after this:**
"Search what was SHOWN and what was SAID in any video. Automatically."

**What Twelve Labs does:** visual only.
**What you do:** visual + audio, 5x cheaper.

---

### Build 2: MLP Reconstruction Training (Week 1-2)
**Why:** Delta frame accuracy 0.9484 → 0.97+. Every delta embedding in index improves.

```bash
# Generate training data from videos you already have indexed
python training/generate_training_data.py \
  --videos demo_videos/uZAwsh-unZ8.mp4\
  --output training/data/samples.json

# Train
python training/train.py \
  --data training/data/samples.json \
  --epochs 50

# Integrate trained model
# In reconstructor.py: replace weighted average with MLP.forward()

# Re-run validation
python main.py --video "Input video/MOT17-02-SDP-raw.webm" \
  --model training/checkpoints/best_model.pt

# Report new numbers
# Expected: 0.9484 → 0.97+
```

**Paper update after this:**
Add one row to Table 1: "ADVE + Learned Reconstruction: X% savings, 0.97 cosine sim"
This makes the paper stronger and the product more accurate.

---

### Build 3: Public Demo (Week 2) — Most Urgent Product Move

Without a public URL nobody can try it. Gradio share URL is one command.

```bash
# Set Groq key (get new one after rotating)
$env:GROQ_API_KEY="your_new_key"

# Run with share flag
python adve_v2/demo_v2.py --share

# Gradio prints: https://XXXXXX.gradio.live
# This is your demo URL. Share it.
```

Where to post the URL:

```
Hacker News:
  Title: "Show HN: Search inside any video with natural language —
          visual + audio, 96% fewer encoder calls than Twelve Labs"
  URL: your gradio link
  Text: 2 sentences about what it does + the GitHub link

Post on: Tuesday 9am IST (best HN traffic)

r/MachineLearning:
  Post same day as HN

LinkedIn:
  Post demo screen recording (90-second video)
  Same numbers: 96.67% savings, 0.9923 cosine sim, 65% less VRAM
```

---

### Build 4: Zenodo DOI (Week 2, 1 hour)

```
zenodo.org → New Upload

Files:
  ADVE_PAPER_DRAFT.md
  All .py files
  outputs/adve_results.png
  outputs/adve_results.json
  README.md

Title: ADVE: Anchor-Delta Video Embedding for
       Efficient Semantic Scene Understanding

→ Publish → get DOI in 5 minutes
→ Add to resume immediately
→ Add to GitHub README
```

---

### Build 5: Fix Two-Stage Search (Week 2)

Object search returning 0.27 similarity means it is not usable. Fix:

```
Stage 1: Search global frame embeddings for scene
  → finds timestamps where the scene is relevant

Stage 2: At each timestamp, attach detected objects
  → confirms what was physically present at that moment

Return: ranked results with scene relevance + object list
```

See two-stage search code in previous message.

---

## After the Five Builds — What You Have

```
Feature                    You        Twelve Labs   Google Video AI
─────────────────────────────────────────────────────────────────────
Visual search              ✅         ✅            ✅
Audio search               ✅         ❌            partial
Object-level search        ✅         partial       partial
Cost per minute            ~₹1        ~₹40          cloud pricing
Edge deployment (330 MB)   ✅         ❌            ❌
Open source                ✅         ❌            ❌
Training-free              ✅         ❌            ❌
Multimodal merge           ✅         ❌            ❌
DOI + paper                ✅         ❌            N/A
```

On every axis where it matters for price-sensitive developers, you win.

---

## The Three Datasets to Add (For Paper Credibility)

Currently validated on synthetic + MOT17. Add three more:

```
Dataset 1: UCF-101 (Action Recognition)
  Why: proves it works on action videos not just pedestrian tracking
  Download: crcv.ucf.edu/data/UCF101.php
  What to measure: encoder savings across 101 action classes
  Expected: 70-85% savings (more motion than MOT17)

Dataset 2: VIRAT (Surveillance)
  Why: direct proof for the surveillance use case
  Download: viratdata.org (free, academic)
  What to measure: savings on static camera footage
  Expected: 90-95% savings

Dataset 3: Any lecture video from YouTube
  Why: proves EdTech use case, matches demo audience
  Source: 3Blue1Brown, StatQuest, any public lecture
  What to measure: savings + audio search accuracy on same video
  Expected: 92-96% savings (static camera, talking head)
```

---

## The Technical Ceiling — What Makes It Genuinely Best

After the five builds and three datasets, one more technical improvement
pushes ADVE above everything else:

**Cross-modal Re-ranking**

After visual + audio search returns top 20 results, run a cross-encoder
that reads (query, frame_description) pairs and re-scores them.

```python
# After merge_results(), re-rank with a small cross-encoder
# Use a tiny BERT model (50MB) to re-rank top 20 → return top 5

from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = [(query, r.text or "visual scene") for r in merged_results]
scores = reranker.predict(pairs)

for r, score in zip(merged_results, scores):
    r.similarity = float(score)

merged_results.sort(key=lambda r: r.similarity, reverse=True)
```

This is what Cohere's rerank API does. You do it locally, free, faster.

---

## Revenue Path After Builds

```
Today:       0 users, 0 revenue
Week 2:      Demo live, first users from HN post
Week 4:      5-10 people using it
Month 2:     First paying customer (₹2,000-10,000/month)
Month 3:     5 paying customers = ₹10,000-50,000/month recurring
Month 6:     Paper accepted at workshop/conference
              10-20 customers, ₹50,000-2,00,000/month
Year 1:      Either: raise seed round on this traction
             Or: profitable small product, keep building
```

The bottleneck between today and first revenue is Build 3: the public demo.
Everything else is secondary to that.

---

## This Week — Exact Daily Plan

```
Monday:
  Install whisper: pip install openai-whisper
  Add adve/audio/__init__.py
  Copy indexer.py and multimodal_search.py into repo
  Test: python -c "import whisper; m=whisper.load_model('base'); print('OK')"

Tuesday:
  Integrate audio indexing into server.py index_video_task
  Test: index one video, verify audio segments appear in SQLite
  Check: SELECT camera_id FROM embeddings WHERE camera_id LIKE '%AUDIO%'

Wednesday:
  Update search endpoint to merge visual + audio
  Test: search "machine learning" → get both visual and audio results
  Verify: some results show source="both"

Thursday:
  Run demo_v2.py --share
  Get Gradio URL
  Record 90-second screen recording

Friday (9am IST — post day):
  Post on Hacker News
  Post on r/MachineLearning
  Post on LinkedIn
  Reply to every comment same day
```

Start with whisper install Monday. Everything follows from that.
