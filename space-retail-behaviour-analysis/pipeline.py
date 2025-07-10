import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from decord import VideoReader, cpu
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForVideoClassification
import supervision as sv
from shapely.geometry import box as shp_box
from huggingface_hub import snapshot_download

# Add at the top with other imports
import shutil

# Add this after your other imports
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global model cache
MODELS = {}

def load_models():
    """Load all required models"""
    global MODELS
    
    if not MODELS:
        print("Loading models...")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Download shelf segmentation model
        snapshot_download(repo_id="cheesecz/shelf-segmentation", local_dir="models/shelf_model", local_dir_use_symlinks=False)
        
        # Load models with explicit device setting
        MODELS["person_model"] = YOLO('yolo11s.pt').to(device)
        MODELS["shelf_model"] = YOLO("models/shelf_model/best.pt").to(device)
        MODELS["action_model"] = AutoModelForVideoClassification.from_pretrained('haipradana/s-h-o-p-domain-adaptation').to(device)
        MODELS["image_processor"] = AutoImageProcessor.from_pretrained('haipradana/s-h-o-p-domain-adaptation')
        
        # Store device info
        MODELS["device"] = device
        MODELS["action_model"].eval()  # Set model to evaluation mode
        MODELS["id2label"] = MODELS["action_model"].config.id2label
        
        print("Models loaded successfully")
    
    return MODELS

def merge_consecutive_predictions(preds, min_duration_frames=0):
    """Merge consecutive predictions of the same class"""
    if not preds: return []
    merged = []
    current = preds[0].copy()
    for nxt in preds[1:]:
        if nxt['pred'] == current['pred']:
            current['end'] = nxt['end']
        else:
            merged.append(current)
            current = nxt.copy()
    merged.append(current)
    return [e for e in merged if (e['end'] - e['start']) >= min_duration_frames]

def iou_xyxy(a, b):
    """Calculate IoU between two bounding boxes in (x1,y1,x2,y2) format"""
    inter = shp_box(*a).intersection(shp_box(*b)).area
    union = shp_box(*a).union(shp_box(*b)).area
    return inter / union if union else 0

def full_video_analysis(video_path, output_dir, max_duration=30):
    """
    Process a video file and generate analysis outputs
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save outputs
        max_duration: Maximum video duration to process (in seconds)
        
    Returns:
        Path to processed video
        Dictionary of metrics
    """
    # Load models if not already loaded
    models = load_models()
    person_model = models["person_model"]
    shelf_model = models["shelf_model"]
    action_model = models["action_model"]
    image_processor = models["image_processor"]
    device = models["device"]
    id2label = models["id2label"]
    
    # Load video
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    print(f"FPS video = {fps:.2f}")
    H, W, _ = vr[0].shape
    
    # Limit video length
    max_frames = min(len(vr), int(max_duration * fps))
    
    # Output video path
    out_path = os.path.join(output_dir, 'video_output.mp4')
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    
    # Initialize tracking
    # Replace the tracker initialization with:
    # tracker = person_model.track(source=video_path, persist=True, tracker='bytetrack.yaml',
    #                         classes=[0], stream=True)

    tracker = person_model.track(source=video_path, persist=True, tracker='bytetrack.yaml',
                            classes=[0], stream=True, device=device)
    
    # Rest of your frame processing code
    # Data containers
    tracks = defaultdict(list)
    raw_actions = defaultdict(list)
    heatmap_grid = np.zeros((20, 20))
    shelf_boxes_per_frame = {}
    shelf_last_box = {}
    next_shelf_idx = 1
    IOU_TH = 0.5  # IoU threshold for considering same shelf
    
    # ---------- PASS 1: Detection + Tracking ----------
    # Then limit frames manually in your processing loop
    for f_idx, result in enumerate(tracker):
        if f_idx >= max_frames:
            break
            
        frame = vr[f_idx].asnumpy()
        # res_shelf = shelf_model(frame)
        res_shelf = shelf_model(frame, device=device)
        
        # Process shelf detections
        assigned = []
        raw_boxes = [b.xyxy[0].cpu().numpy() for b in res_shelf[0].boxes] if res_shelf[0].boxes else []
        
        for box in raw_boxes:
            cur = tuple(map(int, box))
            best_iou, best_id = 0, None
            for sid, prev in shelf_last_box.items():
                val = iou_xyxy(cur, prev)
                if val > best_iou:
                    best_iou, best_id = val, sid
            if best_iou >= IOU_TH:
                shelf_last_box[best_id] = cur
                assigned.append((best_id, cur))
            else:
                sid = f"shelf_{next_shelf_idx}"
                next_shelf_idx += 1
                shelf_last_box[sid] = cur
                assigned.append((sid, cur))
        
        shelf_boxes_per_frame[f_idx] = assigned
        
        # Process person detections
        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            for box, pid in zip(boxes, ids):
                tracks[pid].append({'frame': f_idx, 'bbox': box, 'pid': pid})
                cx, cy = (box[0] + box[2])/2, (box[1] + box[3])/2
                gx, gy = min(int(cx/W*20), 19), min(int(cy/H*20), 19)
                heatmap_grid[gy, gx] += 1
    
    # ---------- Action Recognition ----------
    for pid, dets in tracks.items():
        if len(dets) < 16: continue
        for i in range(0, len(dets)-15, 8):
            clip_frames = [d['frame'] for d in dets[i:i+16]]
            imgs = vr.get_batch(clip_frames).asnumpy()
            crops = [img[int(d['bbox'][1]):int(d['bbox'][3]),
                       int(d['bbox'][0]):int(d['bbox'][2])] for img, d in zip(imgs, dets[i:i+16])]
            if not crops: continue
            try:
                inp = image_processor(crops, return_tensors='pt').to(device)
                pred = action_model(**inp).logits.argmax(-1).item()
                raw_actions[pid].append({'start': dets[i]['frame'], 'end': dets[i+15]['frame'], 'pred': pred})
            except Exception as e:
                print(f"Error processing action for pid {pid}: {e}")
    
    action_preds = {pid: merge_consecutive_predictions(v, int(fps*0.4))
                   for pid, v in raw_actions.items()}
    
    # ---------- Calculate Shelf Interactions ----------
    shelf_interaksi = defaultdict(int)
    for pid, dets in tracks.items():
        for d in dets:
            f = d['frame']
            x1, y1, x2, y2 = d['bbox']
            cx, cy = (x1+x2)/2, (y1+y2)/2
            for sid, (sx1, sy1, sx2, sy2) in shelf_boxes_per_frame.get(f, []):
                if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                    shelf_interaksi[sid] += 1
    
    # Save interaction summary
    pd.DataFrame(list(shelf_interaksi.items()),
                columns=['shelf_id', 'interaksi']).to_csv(
                os.path.join(output_dir, 'rak_interaksi.csv'), index=False)
    
    # ---------- Generate Heatmap ----------
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_grid, cmap='hot', interpolation='nearest')
    plt.title('Heatmap Flow Pengunjung')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap.png'))
    plt.close()
    
    # ---------- Action Summary ----------
    all_actions = []
    for pid, acts in action_preds.items():
        for a in acts:
            all_actions.append([pid, a['start'], a['end'], id2label[a['pred']]])
    
    pd.DataFrame(all_actions,
                columns=['id', 'start', 'end', 'action']).to_csv(
                os.path.join(output_dir, 'action_log.csv'), index=False)
    
    pd.DataFrame(pd.Series([row[3] for row in all_actions])
                .value_counts()).to_csv(
                os.path.join(output_dir, 'action_summary.csv'))
    
    # ---------- Action ↔ Shelf Mapping ----------
    action_shelf = []
    shelf_action_counter = defaultdict(int)
    
    for pid, acts in action_preds.items():
        for seg in acts:
            s, e, act_id = seg['start'], seg['end'], seg['pred']
            act_label = id2label[act_id]
            
            for f in range(s, e+1):
                det = next((d for d in tracks[pid] if d['frame'] == f), None)
                if det is None: continue
                x1, y1, x2, y2 = det['bbox']
                cx, cy = (x1+x2)/2, (y1+y2)/2
                
                for sid, (sx1, sy1, sx2, sy2) in shelf_boxes_per_frame.get(f, []):
                    if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                        action_shelf.append([pid, f, sid, act_label])
                        shelf_action_counter[(sid, act_label)] += 1
                        break
    
    # Save detailed action-shelf mapping
    pd.DataFrame(action_shelf,
                columns=['pid', 'frame', 'shelf_id', 'action']).to_csv(
                os.path.join(output_dir, 'action_shelf_log.csv'), index=False)
    
    pd.DataFrame([{'shelf_id': k[0], 'action': k[1], 'count': v}
                 for k, v in shelf_action_counter.items()]).to_csv(
                 os.path.join(output_dir, 'action_shelf_summary.csv'), index=False)
    
    # ---------- Layout Recommendations ----------
    pd.DataFrame(sorted(shelf_interaksi.items(),
                        key=lambda x: -x[1]),
                columns=['shelf_id', 'interaksi']).to_csv(
                os.path.join(output_dir, 'rekomendasi_layout.csv'), index=False)
    
    # ---------- Create Video with Overlay ----------
    heatmap_ann = sv.HeatMapAnnotator(position=sv.Position.BOTTOM_CENTER,
                                     opacity=0.3, radius=20, kernel_size=25)
    
    # For web performance, we can skip frames if needed
    render_every = max(1, int(len(vr) / 300))  # Aim for ~300 frames max
    
    for f_idx in range(min(max_frames, len(vr))):
        if f_idx % render_every != 0 and f_idx != min(max_frames, len(vr))-1:  # Always render last frame
            continue
            
        frame = vr[f_idx].asnumpy()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Draw shelves
        for sid, (x1, y1, x2, y2) in shelf_boxes_per_frame.get(f_idx, []):
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame_bgr, sid, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw persons
        cur_tracks = [t for pid, v in tracks.items() for t in v if t['frame'] == f_idx]
        for t in cur_tracks:
            x1, y1, x2, y2 = map(int, t['bbox'])
            pid = t['pid']
            label = f"ID {pid}"
            for a in action_preds.get(pid, []):
                if a['start'] <= f_idx <= a['end']:
                    label += f" | {id2label[a['pred']]}"
                    break
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_bgr, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        

        # Add heatmap if there are tracks
        cur_tracks = [t for pid, v in tracks.items() for t in v if t['frame'] == f_idx]
        if cur_tracks:
            dets = sv.Detections(xyxy=np.array([t['bbox'] for t in cur_tracks]),
                            confidence=np.ones(len(cur_tracks)),
                            class_id=np.zeros(len(cur_tracks)))
            frame_bgr = heatmap_ann.annotate(scene=frame_bgr.copy(), detections=dets)
        
        vw.write(frame_bgr)
        
    vw.release()
    
    # Generate additional analytics
    generate_dwell_time_analysis(os.path.join(output_dir, 'action_shelf_log.csv'), 
                              output_dir, fps)
    generate_journey_analysis(os.path.join(output_dir, 'action_shelf_log.csv'), 
                           output_dir)
    generate_behavioral_archetypes(output_dir)
    
    # Return paths and metrics
    return out_path, {
        'heatmap': os.path.join(output_dir, 'heatmap.png'),
        'dwell_time': os.path.join(output_dir, 'dwell_time_chart.png'),
        'journey': os.path.join(output_dir, 'journey_chart.png'),
        'archetypes': os.path.join(output_dir, 'behavioral_archetypes.csv')
    }

def generate_dwell_time_analysis(action_shelf_log_path, output_dir, fps):
    """Generate dwell time analysis chart"""
    df = pd.read_csv(action_shelf_log_path)
    
    # Calculate dwell time
    dwell_time_data = []
    for pid, group in df.groupby('pid'):
        start_frame, current_shelf = 0, -1
        for i, row in group.iterrows():
            if row['shelf_id'] != current_shelf:
                if current_shelf != -1:
                    dwell_seconds = (end_frame - start_frame) / fps
                    dwell_time_data.append({'pid': pid, 'shelf_id': current_shelf, 'dwell_time': dwell_seconds})
                current_shelf, start_frame = row['shelf_id'], row['frame']
            end_frame = row['frame']
        if current_shelf != -1:
            dwell_seconds = (end_frame - start_frame) / fps
            dwell_time_data.append({'pid': pid, 'shelf_id': current_shelf, 'dwell_time': dwell_seconds})
    
    dwell_df = pd.DataFrame(dwell_time_data)
    avg_dwell_time = dwell_df.groupby('shelf_id')['dwell_time'].mean().sort_values(ascending=False)
    
    # Save data
    avg_dwell_time.to_csv(os.path.join(output_dir, 'average_dwell_time.csv'))
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    avg_dwell_time.plot(kind='bar', color='purple')
    plt.title('Dwell Time (Rata-rata) Tiap Rak', fontsize=14)
    plt.xlabel('Shelf ID')
    plt.ylabel('Seconds')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dwell_time_chart.png'))
    plt.close()

def generate_journey_analysis(action_shelf_log_path, output_dir):
    """Generate customer journey analysis chart"""
    df = pd.read_csv(action_shelf_log_path)
    
    # Find key events
    reach_events = df[df['action'] == 'Reach To Shelf'][['pid', 'shelf_id']].drop_duplicates().assign(did_reach=True)
    inspect_events = df[df['action'] == 'Inspect Product'][['pid', 'shelf_id']].drop_duplicates().assign(did_inspect=True)
    return_events = df[df['action'] == 'Hand In Shelf'][['pid', 'shelf_id']].drop_duplicates().assign(did_return=True)
    
    # Create analysis dataframe
    interactions = df[['pid', 'shelf_id']].drop_duplicates()
    analysis_df = pd.merge(interactions, reach_events, on=['pid', 'shelf_id'], how='left')
    analysis_df = pd.merge(analysis_df, inspect_events, on=['pid', 'shelf_id'], how='left')
    analysis_df = pd.merge(analysis_df, return_events, on=['pid', 'shelf_id'], how='left')
    analysis_df = analysis_df.fillna(False)
    
    # Categorize outcomes
    def categorize_outcome(row):
        if not row['did_reach']:
            return 'No Reach'
        if row['did_inspect'] and row['did_return']:
            return 'Keraguan & Pembatalan'
        elif row['did_inspect'] and not row['did_return']:
            return 'Konversi Sukses'
        else:
            return 'Kegagalan Menarik Minat'
            
    analysis_df['outcome'] = analysis_df.apply(categorize_outcome, axis=1)
    relevant_outcomes = analysis_df[analysis_df['outcome'] != 'No Reach']
    
    # Aggregate results
    outcome_summary = relevant_outcomes.groupby(['shelf_id', 'outcome']).size().unstack(fill_value=0)
    outcome_percentage = outcome_summary.div(outcome_summary.sum(axis=1), axis=0) * 100
    
    desired_order = ['Konversi Sukses', 'Keraguan & Pembatalan', 'Kegagalan Menarik Minat']
    for col in desired_order:
        if col not in outcome_percentage.columns:
            outcome_percentage[col] = 0
    outcome_percentage = outcome_percentage[desired_order]
    
    # Save data
    outcome_percentage.to_csv(os.path.join(output_dir, 'journey_analysis.csv'))
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    outcome_percentage.plot(
        kind='bar', 
        stacked=True,
        color=['#2ca02c', '#ff7f0e', '#d62728'] # Green, Orange, Red
    )
    plt.title('Analisis Perilaku Pengunjung tiap Rak', fontsize=14)
    plt.xlabel('Shelf ID')
    plt.ylabel('Outcome Distribution (%)')
    plt.legend(title='Interaction Outcome')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'journey_chart.png'))
    plt.close()

def generate_behavioral_archetypes(output_dir):
    """Generate behavioral archetypes analysis"""
    # Load previously calculated data
    try:
        df_raw = pd.read_csv(os.path.join(output_dir, 'action_shelf_log.csv'))
        df_dwell = pd.read_csv(os.path.join(output_dir, 'average_dwell_time.csv'))
        df_dwell.rename(columns={df_dwell.columns[0]: 'shelf_id'}, inplace=True)
        df_outcomes = pd.read_csv(os.path.join(output_dir, 'journey_analysis.csv')).set_index('shelf_id')
        
        # Calculate unique interactions
        unique_interactions = df_raw.groupby('shelf_id')['pid'].nunique().reset_index()
        unique_interactions.rename(columns={'pid': 'Interaksi Unik'}, inplace=True)
        
        # Merge data
        summary_table = pd.merge(unique_interactions, df_dwell, on='shelf_id')
        summary_table = summary_table.set_index('shelf_id')
        
        # Define behavioral archetypes
        def get_behavioral_archetype(row):
            shelf_id = row.name
            dwell_time = row['dwell_time']
            unique_visits = row['Interaksi Unik']
            
            # Check if shelf exists in outcomes data
            if shelf_id not in df_outcomes.index:
                if dwell_time > 3.0:
                    return 'Passive Attention (No Physical Engagement)'
                else:
                    return 'Low Engagement Zone'
                    
            outcomes = df_outcomes.loc[shelf_id]
            if 'Konversi Sukses' in outcomes and outcomes['Konversi Sukses'] > 10:
                return 'High Attention, Low Conversion'
                
            if 'Keraguan & Pembatalan' in outcomes.index:
                dominant_outcome = outcomes.idxmax()
                if dominant_outcome == 'Keraguan & Pembatalan':
                    return 'Interaksi Positif Namun Ragu'
            
            if unique_visits > 8:
                return 'Traffic Tinggi, Engagement Rendah'
            else:
                return 'Low Engagement Zone'
        
        # Apply archetypes
        summary_table['Arketipe Perilaku'] = summary_table.apply(get_behavioral_archetype, axis=1)
        
        # Format table
        summary_table.reset_index(inplace=True)
        summary_table.rename(columns={'shelf_id': 'Rak', 'dwell_time': 'Rata-rata Dwell (s)'}, inplace=True)
        summary_table['Rata-rata Dwell (s)'] = summary_table['Rata-rata Dwell (s)'].round(2)
        summary_table = summary_table.sort_values(by='Interaksi Unik', ascending=False)
        
        # Save results
        summary_table.to_csv(os.path.join(output_dir, 'behavioral_archetypes.csv'), index=False)
        return summary_table
    except Exception as e:
        print(f"Error generating behavioral archetypes: {e}")
        return pd.DataFrame({
            'Rak': ['N/A'], 
            'Interaksi Unik': [0], 
            'Rata-rata Dwell (s)': [0], 
            'Arketipe Perilaku': ['Error']
        })

def get_key_metrics(output_dir):
    """Collect key metric visualizations for Gradio interface"""
    heatmap_path = os.path.join(output_dir, 'heatmap.png')
    dwell_time_path = os.path.join(output_dir, 'dwell_time_chart.png')
    journey_path = os.path.join(output_dir, 'journey_chart.png')
    
    try:
        archetypes_df = pd.read_csv(os.path.join(output_dir, 'behavioral_archetypes.csv'))
    except:
        archetypes_df = pd.DataFrame({
            'Rak': ['N/A'], 
            'Interaksi Unik': [0], 
            'Rata-rata Dwell (s)': [0], 
            'Arketipe Perilaku': ['No Data']
        })
    
    return heatmap_path, dwell_time_path, journey_path, archetypes_df