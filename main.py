import argparse
from config import Config
from pipeline import ADVEPipeline


def parse_args():
    p = argparse.ArgumentParser(
        description="ADVE — Anchor-Delta Video Embedding | PoC Validator"
    )
    p.add_argument("--video",               type=str,   required=True,
                   help="Path to input video file")
    p.add_argument("--spatial-threshold",   type=float, default=0.30,
                   help="ΔG magnitude that triggers anchor refresh (default: 0.30)")
    p.add_argument("--appearance-threshold",type=float, default=0.15,
                   help="Histogram diff that triggers anchor refresh (default: 0.15)")
    p.add_argument("--max-delta-frames",    type=int,   default=30,
                   help="Force anchor refresh every N frames (default: 30)")
    p.add_argument("--output-dir",          type=str,   default="outputs",
                   help="Where to save results (default: ./outputs)")
    p.add_argument("--no-validation",       action="store_true",
                   help="Skip ground-truth CLIP encoding on delta frames to measure true inference speed")
    return p.parse_args()


def main():
    args = parse_args()

    config = Config(
        SPATIAL_THRESHOLD    = args.spatial_threshold,
        APPEARANCE_THRESHOLD = args.appearance_threshold,
        MAX_DELTA_FRAMES     = args.max_delta_frames,
        OUTPUT_DIR           = args.output_dir,
    )

    pipeline = ADVEPipeline(config)
    pipeline.process_video(args.video, no_validation=args.no_validation)


if __name__ == "__main__":
    main()
