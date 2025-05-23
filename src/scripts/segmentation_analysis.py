#!/usr/bin/env python3
"""
Analyze how comments would be segmented using the existing comment_parser
"""

import sqlite3
import os
from dotenv import load_dotenv
import sys
import json

# Add src to path to import comment_parser
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from kexp_processing_utils.comment_parser import parse_comment_to_prodigy_tasks
except ImportError:
    print("❌ Could not import comment_parser. Make sure src/ is in your path.")
    sys.exit(1)

load_dotenv()


def analyze_segmentation():
    """Analyze how comments would be segmented for LLM processing"""

    db_path = os.environ.get('KEXP_DB_PATH')
    if not db_path or not os.path.exists(db_path):
        print("❌ KEXP_DB_PATH not set or database not found")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    print("✂️  KEXP Comment Segmentation Analysis")
    print("=" * 50)

    # Get a sample of comments for analysis
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            p.id as play_id,
            p.comment,
            p.airdate,
            s.artist,
            s.song,
            s.album,
            sh.host_names
        FROM plays p
        LEFT JOIN songs s ON p.song_id = s.song_id
        LEFT JOIN shows sh ON p.show = sh.id
        WHERE p.comment IS NOT NULL 
            AND LENGTH(TRIM(p.comment)) >= 20
        ORDER BY RANDOM() 
        LIMIT 500
    """)

    sample_comments = cursor.fetchall()
    conn.close()

    segment_stats = {
        'total_comments': 0,
        'total_segments': 0,
        'comments_with_multiple_segments': 0,
        'segments_per_comment': [],
        'segment_lengths': [],
        'filtered_out_comments': 0
    }

    print(f"Analyzing {len(sample_comments)} sample comments...")

    for comment_row in sample_comments:
        comment_text = comment_row['comment']
        meta = {
            'play_id': comment_row['play_id'],
            'db_artist_name': comment_row['artist'],
            'db_song_title': comment_row['song'],
            'db_album_title': comment_row['album'],
            'dj_host_names': comment_row['host_names']
        }

        # Use the comment parser to segment
        segments = list(parse_comment_to_prodigy_tasks(comment_text, meta))

        segment_stats['total_comments'] += 1

        if not segments:
            segment_stats['filtered_out_comments'] += 1
            continue

        segment_count = len(segments)
        segment_stats['total_segments'] += segment_count
        segment_stats['segments_per_comment'].append(segment_count)

        if segment_count > 1:
            segment_stats['comments_with_multiple_segments'] += 1

        for segment in segments:
            segment_stats['segment_lengths'].append(len(segment['text']))

    # Calculate statistics
    valid_comments = segment_stats['total_comments'] - \
        segment_stats['filtered_out_comments']
    avg_segments_per_comment = segment_stats['total_segments'] / \
        valid_comments if valid_comments > 0 else 0

    if segment_stats['segment_lengths']:
        avg_segment_length = sum(
            segment_stats['segment_lengths']) / len(segment_stats['segment_lengths'])
        max_segment_length = max(segment_stats['segment_lengths'])
        min_segment_length = min(segment_stats['segment_lengths'])
    else:
        avg_segment_length = max_segment_length = min_segment_length = 0

    print(f"\n📊 SEGMENTATION RESULTS")
    print(f"Comments analyzed: {segment_stats['total_comments']}")
    print(f"Comments filtered out: {segment_stats['filtered_out_comments']}")
    print(f"Valid comments: {valid_comments}")
    print(f"Total segments created: {segment_stats['total_segments']}")
    print(
        f"Comments with multiple segments: {segment_stats['comments_with_multiple_segments']}")
    print(f"Average segments per comment: {avg_segments_per_comment:.2f}")
    print(f"Average segment length: {avg_segment_length:.1f} characters")
    print(
        f"Segment length range: {min_segment_length} - {max_segment_length} characters")

    # Estimate tokens for segments
    estimated_segment_tokens = sum(
        length / 4.5 for length in segment_stats['segment_lengths'])
    print(f"Estimated tokens in segments: {estimated_segment_tokens:,.0f}")

    # Show some example segmentations
    print(f"\n🔍 EXAMPLE SEGMENTATIONS")
    print("-" * 40)

    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            p.id as play_id,
            p.comment,
            s.artist,
            s.song
        FROM plays p
        LEFT JOIN songs s ON p.song_id = s.song_id
        WHERE p.comment IS NOT NULL 
            AND LENGTH(TRIM(p.comment)) >= 50
            AND p.comment LIKE '%---%' OR p.comment LIKE '%\n\n%'
        ORDER BY RANDOM() 
        LIMIT 3
    """)

    complex_comments = cursor.fetchall()

    for i, comment_row in enumerate(complex_comments, 1):
        comment_text = comment_row['comment']
        meta = {
            'play_id': comment_row['play_id'],
            'db_artist_name': comment_row['artist'],
            'db_song_title': comment_row['song']
        }

        segments = list(parse_comment_to_prodigy_tasks(comment_text, meta))

        print(f"\nExample {i} - {len(segments)} segments:")
        print(f"Original comment ({len(comment_text)} chars):")
        print(
            f"'{comment_text[:150]}{'...' if len(comment_text) > 150 else ''}'")
        print(f"Segments:")
        for j, segment in enumerate(segments, 1):
            segment_text = segment['text']
            preview = segment_text[:100] + \
                "..." if len(segment_text) > 100 else segment_text
            print(f"  {j}. ({len(segment_text)} chars) '{preview}'")

    conn.close()

    # Extrapolate to full database
    print(f"\n🔮 FULL DATABASE EXTRAPOLATION")
    print("-" * 40)

    # Get total comment count
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) as total_substantial_comments
        FROM plays 
        WHERE comment IS NOT NULL AND LENGTH(TRIM(comment)) >= 20
    """)
    total_comments = cursor.fetchone()['total_substantial_comments']
    conn.close()

    if valid_comments > 0:
        # Scale up the sample results
        scaling_factor = total_comments / len(sample_comments)
        estimated_total_segments = int(
            segment_stats['total_segments'] * scaling_factor)
        estimated_total_tokens = int(estimated_segment_tokens * scaling_factor)

        print(f"Total substantial comments in DB: {total_comments:,}")
        print(f"Estimated total segments: {estimated_total_segments:,}")
        print(f"Estimated total tokens: {estimated_total_tokens:,}")

        # Processing time estimates for segments
        print(f"\n⏱️  SEGMENT-BASED PROCESSING TIME ESTIMATES")
        models = {
            'Fast local (50 tok/s)': 50,
            'Medium local (25 tok/s)': 25,
            'Large local (5 tok/s)': 5
        }

        for model_name, tokens_per_sec in models.items():
            total_seconds = estimated_total_tokens / tokens_per_sec
            if total_seconds < 3600:
                time_str = f"{total_seconds/60:.1f} minutes"
            elif total_seconds < 86400:
                time_str = f"{total_seconds/3600:.1f} hours"
            else:
                time_str = f"{total_seconds/86400:.1f} days"
            print(f"{model_name}: {time_str}")

    return {
        'sample_size': len(sample_comments),
        'total_segments': segment_stats['total_segments'],
        'avg_segments_per_comment': avg_segments_per_comment,
        'avg_segment_length': avg_segment_length,
        'estimated_total_segments': estimated_total_segments if valid_comments > 0 else 0,
        'estimated_total_tokens': estimated_total_tokens if valid_comments > 0 else 0
    }


if __name__ == "__main__":
    try:
        results = analyze_segmentation()

        # Save results
        with open('segmentation_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Analysis saved to 'segmentation_analysis_results.json'")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
