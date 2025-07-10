import pandas as pd
import numpy as np
import os
from collections import defaultdict

def calculate_dwell_time(action_shelf_log_df, fps):
    """Calculate dwell time per person per shelf"""
    dwell_time_data = []
    
    for (pid, shelf_id), group in action_shelf_log_df.groupby(['pid', 'shelf_id']):
        frames = group['frame'].sort_values().tolist()
        if not frames:
            continue
            
        segments = []
        segment_start = frames[0]
        prev_frame = frames[0]
        
        # Find continuous segments
        for frame in frames[1:]:
            if frame > prev_frame + 3:  # Allow small gaps
                segments.append((segment_start, prev_frame))
                segment_start = frame
            prev_frame = frame
        segments.append((segment_start, prev_frame))
        
        # Calculate total dwell time across segments
        total_frames = sum(end - start + 1 for start, end in segments)
        dwell_seconds = total_frames / fps
        
        dwell_time_data.append({
            'pid': pid, 
            'shelf_id': shelf_id, 
            'dwell_frames': total_frames,
            'dwell_time': dwell_seconds
        })
    
    return pd.DataFrame(dwell_time_data)

def analyze_customer_journey(action_shelf_log_df):
    """Analyze customer journey through shelves"""
    # Extract unique interactions
    interactions = action_shelf_log_df[['pid', 'shelf_id']].drop_duplicates()
    
    # Find key events
    reach_events = action_shelf_log_df[action_shelf_log_df['action'] == 'Reach To Shelf'][['pid', 'shelf_id']].drop_duplicates().assign(did_reach=True)
    inspect_events = action_shelf_log_df[action_shelf_log_df['action'] == 'Inspect Product'][['pid', 'shelf_id']].drop_duplicates().assign(did_inspect=True)
    return_events = action_shelf_log_df[action_shelf_log_df['action'] == 'Hand In Shelf'][['pid', 'shelf_id']].drop_duplicates().assign(did_return=True)
    
    # Combine into analysis dataframe
    analysis_df = pd.merge(interactions, reach_events, on=['pid', 'shelf_id'], how='left')
    analysis_df = pd.merge(analysis_df, inspect_events, on=['pid', 'shelf_id'], how='left')
    analysis_df = pd.merge(analysis_df, return_events, on=['pid', 'shelf_id'], how='left')
    analysis_df = analysis_df.fillna(False)
    
    # Categorize outcomes
    def categorize_outcome(row):
        if not row['did_reach']:
            return 'No Reach' # Ignore interactions without reach
        
        if row['did_inspect'] and row['did_return']:
            return 'Keraguan & Pembatalan'
        elif row['did_inspect'] and not row['did_return']:
            return 'Konversi Sukses'
        else:
            return 'Kegagalan Menarik Minat'
    
    analysis_df['outcome'] = analysis_df.apply(categorize_outcome, axis=1)
    
    return analysis_df

def calculate_behavioral_archetypes(dwell_df, journey_df, action_shelf_log_df):
    """Calculate behavioral archetypes for each shelf"""
    # Get unique visitors per shelf
    unique_visitors = action_shelf_log_df.groupby('shelf_id')['pid'].nunique().reset_index()
    unique_visitors.rename(columns={'pid': 'unique_visitors'}, inplace=True)
    
    # Get average dwell time
    avg_dwell = dwell_df.groupby('shelf_id')['dwell_time'].mean().reset_index()
    
    # Get outcome percentages
    relevant_journey = journey_df[journey_df['outcome'] != 'No Reach']
    outcome_counts = relevant_journey.groupby(['shelf_id', 'outcome']).size().unstack(fill_value=0)
    total_counts = outcome_counts.sum(axis=1)
    outcome_percentage = outcome_counts.div(total_counts, axis=0) * 100
    
    # Merge data
    merged_df = pd.merge(unique_visitors, avg_dwell, on='shelf_id', how='outer')
    
    # Define archetypes
    archetypes = []
    for _, row in merged_df.iterrows():
        shelf_id = row['shelf_id']
        visitors = row['unique_visitors']
        dwell = row['dwell_time']
        
        if shelf_id not in outcome_percentage.index:
            if dwell > 3.0:
                archetype = 'Passive Attention'
            else:
                archetype = 'Low Engagement Zone'
        else:
            outcomes = outcome_percentage.loc[shelf_id]
            
            if 'Konversi Sukses' in outcomes and outcomes['Konversi Sukses'] > 20:
                archetype = 'High Conversion'
            elif 'Keraguan & Pembatalan' in outcomes and outcomes['Keraguan & Pembatalan'] > 50:
                archetype = 'High Interest, Low Conversion'
            elif visitors > 10:
                archetype = 'High Traffic, Low Engagement'
            else:
                archetype = 'Low Engagement Zone'
        
        archetypes.append({
            'shelf_id': shelf_id,
            'unique_visitors': visitors,
            'avg_dwell_time': dwell,
            'archetype': archetype
        })
    
    return pd.DataFrame(archetypes)