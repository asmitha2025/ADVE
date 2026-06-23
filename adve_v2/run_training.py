import os
import sys
import json
import time
from pathlib import Path

# Add the adve_v2 directory to Python path to resolve imports correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from training.generate_training_data import TrainingDataGenerator
from training.train import train
from adve.core.pipeline import ADVEPipeline
from adve.core.config import Config


def run():
    print("=" * 60)
    print("         ADVE v2.0 MLP Training Pipeline")
    print("=" * 60)

    # 1. Paths Setup
    video_path = os.path.join(current_dir, "..", "Input video", "MOT17-02-SDP-raw.webm")
    video_path = os.path.normpath(video_path)
    output_samples = os.path.join(current_dir, "training", "data", "samples.json")
    checkpoint_dir = os.path.join(current_dir, "training", "checkpoints")

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        print("Please check the video path and try again.")
        sys.exit(1)

    print(f"Using video source: {video_path}")
    print(f"Output samples path: {output_samples}")

    # 2. Generate Training Data
    print("\n--- Step 1: Generating Training Samples ---")
    if os.path.exists(output_samples) and os.path.getsize(output_samples) > 1000:
        print(f"Found existing training samples at {output_samples}. Skipping data generation.")
        with open(output_samples, "r") as f:
            samples = json.load(f)
    else:
        start_time = time.time()
        gen = TrainingDataGenerator(device="cpu")  # CPU generation is highly stable
        samples = gen.generate_from_video(video_path, output_samples, max_frames=600)
        print(f"Generated {len(samples)} samples in {time.time() - start_time:.2f} seconds.")

    if len(samples) == 0:
        print("Error: No training samples were generated. Please check if the video has tracked objects.")
        sys.exit(1)

    # 3. Train MLP Model
    print("\n--- Step 2: Training Reconstruction MLP ---")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Run training for 30 epochs
    epochs = 30
    best_model = train(output_samples, epochs=epochs, device="cpu", checkpoint_path=os.path.join(checkpoint_dir, "best_model.pt"))
    print(f"Training complete. Weights saved in {checkpoint_dir}")

    # Free up memory from training before running verification
    import gc
    import torch
    if "best_model" in locals():
        del best_model
    if "samples" in locals():
        del samples
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Verify Reconstruction Similarity Improvement
    print("\n--- Step 3: Verifying Reconstruction Quality ---")
    print("Running pipeline verification...")

    # Load configuration
    config = Config()
    config.MLP_MODEL_PATH = os.path.join(checkpoint_dir, "best_model.pt")

    # Run the pipeline with the trained model
    pipeline = ADVEPipeline(config)
    summary = pipeline.process_video(video_path, no_validation=False, max_frames=300)

    print("\n" + "=" * 60)
    print("            Verification Summary")
    print("=" * 60)
    print(f"  Total processed frames  : {summary['total_frames']}")
    print(f"  Encoder savings         : {summary['encoder_savings_pct']}%")
    print(f"  Mean Cosine Similarity  : {summary['mean_delta_cosine_sim']:.4f}")
    print(f"  Min Cosine Similarity   : {summary['min_delta_cosine_sim']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run()
