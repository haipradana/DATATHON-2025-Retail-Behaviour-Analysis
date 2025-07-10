import os
import gradio as gr
import tempfile
import time
from utils.llm import generate_llm_insights
from pipeline import full_video_analysis, get_key_metrics

# Add at the top with other imports
import shutil

# Add this after your other imports
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache model loading
model_cache = {}

def process_video(video_file, max_duration=30):
    """Process video and return analysis results"""
    if video_file is None:
        return None, None, None, None, None
    
    # Create temp directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Gradio sometimes uploads video as .mp4 or as .webm
        temp_video = os.path.join(temp_dir, "input.mp4")
        
        # Handle both string paths and file-like objects
        if isinstance(video_file, str):
            with open(temp_video, "wb") as f:
                f.write(open(video_file, "rb").read())
        else:
            # Handle file-like object with .name attribute
            with open(temp_video, "wb") as f:
                f.write(open(video_file.name, "rb").read())

        # Process the video
        try:
            processed_video, metrics = full_video_analysis(temp_video, temp_dir, max_duration=max_duration)
            
            # Generate visualizations from metrics
            heatmap, dwell_chart, action_chart, archetype_table = get_key_metrics(temp_dir)
            
            insights = generate_llm_insights(metrics, archetype_table, temp_dir)
            
            # After processing, copy important files to persistent location
            persistent_video_path = os.path.join(OUTPUT_DIR, f"video_output_{int(time.time())}.mp4")
            persistent_heatmap_path = os.path.join(OUTPUT_DIR, f"heatmap_{int(time.time())}.png")
            persistent_dwell_path = os.path.join(OUTPUT_DIR, f"dwell_{int(time.time())}.png")
            persistent_journey_path = os.path.join(OUTPUT_DIR, f"journey_{int(time.time())}.png")
            
            shutil.copy(processed_video, persistent_video_path)
            shutil.copy(metrics['heatmap'], persistent_heatmap_path)
            shutil.copy(metrics['dwell_time'], persistent_dwell_path)
            shutil.copy(metrics['journey'], persistent_journey_path)
            
            # In process_video function after copying files
            # return persistent_video_path, persistent_heatmap_path, persistent_dwell_path, persistent_journey_path, archetype_table
            return persistent_video_path, persistent_heatmap_path, persistent_dwell_path, persistent_journey_path, archetype_table, insights
            # return processed_video, heatmap, dwell_chart, action_chart, archetype_table
            
        except Exception as e:
            # Return appropriate data types for each output
            error_message = f"Error processing video: {str(e)}"
            print(error_message)  # For debugging
            return None, None, None, None, None, None  # Return None for all outputs

# Define Gradio Interface
with gr.Blocks(title="Retail Behaviour Analysis") as demo:
    gr.Markdown("# 🛒 Retail Behaviour Analysis")
    gr.Markdown("""
    Upload a video of people shopping in a supermarket, and the AI will:
    1. Detect shelves and people
    2. Track people movement
    3. Classify their actions
    4. Analyze shelf interactions
    5. Produce insights on shelf effectiveness
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(label="Upload Video (max 30s)")
            duration_slider = gr.Slider(1, 60, value=30, step=1, label="Max Duration (seconds)")
            submit_btn = gr.Button("Analyze", variant="primary")
            
        with gr.Column(scale=2):
            output_video = gr.Video(label="Processed Video")
    
    with gr.Row():
        with gr.Column():
            heatmap_img = gr.Image(label="Customer Traffic Heatmap")
        with gr.Column():
            dwell_chart = gr.Image(label="Dwell Time per Shelf (seconds)")
    
    with gr.Row():
        with gr.Column():
            action_chart = gr.Image(label="Customer Journey Analysis")
        with gr.Column():
            archetype_table = gr.DataFrame(label="Shelf Behavioral Archetypes")
    
    with gr.Row():
        insights_text = gr.Textbox(
            label="Rekomendasi dari Insight", 
            placeholder="Insight analisis akan muncul di sini setelah pemrosesan selesai",
            lines=10
        )
    submit_btn.click(
        process_video,
        inputs=[input_video, duration_slider],
        outputs=[output_video, heatmap_img, dwell_chart, action_chart, archetype_table, insights_text]
    )

if __name__ == "__main__":
    demo.launch()