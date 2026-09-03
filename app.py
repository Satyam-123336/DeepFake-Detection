import gradio as gr
import spaces
import json
from pathlib import Path
from src.pipeline.optimized_inference import analyze_video_optimized
from src.utils.io import ensure_dir

# Setup directories
PROCESSED_DIR = ensure_dir(Path("data/processed"))

@spaces.GPU(duration=120)
def process_video(video_path):
    """
    Core deepfake detection API endpoint.
    Automatically routed to a free A100 GPU by the @spaces.GPU decorator.
    """
    if not video_path:
        return {"error": "No video provided."}
    
    try:
        # The Gradio video component passes the local temporary file path of the uploaded video
        result = analyze_video_optimized(video_path, PROCESSED_DIR)
        return result
    except Exception as e:
        return {"error": str(e)}

# Define the Gradio Interface (which auto-generates the public API)
demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Video to Analyze"),
    outputs=gr.JSON(label="Deepfake Analysis Results"),
    title="RealityGuard AI - Public API",
    description="Send a video to this endpoint to receive a full multi-modal deepfake analysis.",
    allow_flagging="never"
)

if __name__ == "__main__":
    # Hugging Face Spaces will automatically run this
    demo.launch()
