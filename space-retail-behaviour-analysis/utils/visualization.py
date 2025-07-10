import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from collections import defaultdict
import numpy as np

def generate_heatmap_from_grid(heatmap_grid, output_path):
    """Generate a heatmap visualization from a grid of values"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_grid, cmap='hot', annot=False)
    plt.title('Customer Traffic Heatmap')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path

def generate_dwell_time_chart(dwell_df, output_path):
    """Generate a chart showing dwell time per shelf"""
    plt.figure(figsize=(12, 6))
    
    # Sort by dwell time descending
    sorted_df = dwell_df.sort_values('dwell_time', ascending=False)
    
    # Create bar chart
    plt.bar(sorted_df['shelf_id'], sorted_df['dwell_time'], color='purple')
    plt.title('Average Dwell Time per Shelf', fontsize=14)
    plt.xlabel('Shelf ID')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    return output_path

def generate_action_distribution_chart(action_summary_df, output_path):
    """Generate a chart showing distribution of customer actions"""
    plt.figure(figsize=(10, 6))
    
    # Create pie chart
    action_summary_df.plot.pie(y='count', autopct='%1.1f%%', startangle=90, 
                              labels=action_summary_df['action'])
    plt.ylabel('')
    plt.title('Distribution of Customer Actions', fontsize=14)
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    return output_path

def generate_journey_chart(journey_df, output_path):
    """Generate a stacked bar chart showing customer journey outcomes per shelf"""
    plt.figure(figsize=(12, 6))
    
    # Create stacked bar chart
    journey_df.plot(
        kind='bar',
        stacked=True,
        figsize=(12, 6),
        color=['#2ca02c', '#ff7f0e', '#d62728']  # Green, Orange, Red
    )
    plt.title('Customer Journey Analysis per Shelf', fontsize=14)
    plt.xlabel('Shelf ID')
    plt.ylabel('Percentage')
    plt.legend(title='Journey Outcome')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    return output_path

def generate_rak_timeline(interaction_csv_path, tracks, shelf_boxes_per_frame, fps, output_path):
    """Generate a timeline visualization of rack interactions"""
    # Load rack interaction data
    rak_df = pd.read_csv(interaction_csv_path)
    valid_raks = set(rak_df['shelf_id'].tolist())
    
    # Build timeline per rack
    rak_timeline = defaultdict(list)
    for pid, dets in tracks.items():
        for d in dets:
            f = d['frame']
            x1, y1, x2, y2 = d['bbox']
            px, py = (x1 + x2) / 2, (y1 + y2) / 2
            for sid, (sx1, sy1, sx2, sy2) in shelf_boxes_per_frame.get(f, []):
                if sid not in valid_raks:
                    continue
                if sx1 <= px <= sx2 and sy1 <= py <= sy2:
                    rak_timeline[sid].append(f)
    
    # Create visualization
    plt.figure(figsize=(12, max(4, len(rak_timeline) * 0.4)))
    for i, (rak_id, frames) in enumerate(sorted(rak_timeline.items())):
        if not frames:
            continue
        frames = sorted(frames)
        start = frames[0]
        for j in range(1, len(frames)):
            if frames[j] != frames[j-1] + 1:
                plt.plot([start / fps, frames[j-1] / fps], [i, i], linewidth=6)
                start = frames[j]
        plt.plot([start / fps, frames[-1] / fps], [i, i], linewidth=6)
        plt.text(-1, i, rak_id, verticalalignment='center', fontsize=8)
    
    plt.xlabel('Time (seconds)')
    plt.title('Timeline of Shelf Interactions')
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path