#!/usr/bin/env python3
"""
KEXP Processing Feasibility Analyzer
Run this after executing the SQL queries to estimate processing time and requirements.
"""

import sqlite3
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta

load_dotenv()


def analyze_kexp_feasibility():
    """Analyze KEXP database for LLM processing feasibility"""

    db_path = os.environ.get('KEXP_DB_PATH')
    if not db_path or not os.path.exists(db_path):
        print("❌ KEXP_DB_PATH not set or database not found")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    print("🎵 KEXP Database Analysis for LLM Processing")
    print("=" * 60)

    # Basic metrics
    cursor = conn.cursor()

    # Get key statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as total_plays,
            COUNT(CASE WHEN comment IS NOT NULL AND TRIM(comment) != '' THEN 1 END) as plays_with_comments,
            COUNT(CASE WHEN comment IS NOT NULL AND LENGTH(TRIM(comment)) >= 20 THEN 1 END) as substantial_comments,
            MIN(airdate) as earliest_date,
            MAX(airdate) as latest_date
        FROM plays
    """)

    basic_stats = cursor.fetchone()

    # Comment length analysis
    cursor.execute("""
        SELECT 
            AVG(LENGTH(comment)) as avg_length,
            SUM(LENGTH(comment)) as total_chars,
            MAX(LENGTH(comment)) as max_length,
            COUNT(*) as comment_count
        FROM plays 
        WHERE comment IS NOT NULL AND LENGTH(TRIM(comment)) >= 20
    """)

    length_stats = cursor.fetchone()

    # Sample some comments for inspection
    cursor.execute("""
        SELECT 
            p.comment,
            s.artist,
            s.song,
            LENGTH(p.comment) as length
        FROM plays p
        LEFT JOIN songs s ON p.song_id = s.song_id
        WHERE p.comment IS NOT NULL AND LENGTH(TRIM(p.comment)) >= 20
        ORDER BY RANDOM() 
        LIMIT 5
    """)

    sample_comments = cursor.fetchall()

    conn.close()

    # Analysis and calculations
    total_chars = length_stats['total_chars'] or 0
    avg_length = length_stats['avg_length'] or 0
    substantial_comments = basic_stats['substantial_comments']

    # Token estimation (using 4.5 chars per token for English)
    estimated_tokens = int(total_chars / 4.5)

    # Processing time estimates
    tokens_per_second_estimates = {
        'Fast local model (Llama 3.2 3B)': 50,
        'Medium local model (Llama 3.1 8B)': 25,
        'Large local model (Llama 3.1 70B)': 5,
        'Very large local model (Llama 3.1 405B)': 1
    }

    print("📊 DATA OVERVIEW")
    print(f"Total plays: {basic_stats['total_plays']:,}")
    print(f"Plays with comments: {basic_stats['plays_with_comments']:,}")
    print(f"Substantial comments (20+ chars): {substantial_comments:,}")
    print(
        f"Date range: {basic_stats['earliest_date']} to {basic_stats['latest_date']}")
    print(f"Average comment length: {avg_length:.1f} characters")
    print(f"Total characters to process: {total_chars:,}")
    print(f"Estimated tokens: {estimated_tokens:,}")

    print("\n⏱️  PROCESSING TIME ESTIMATES")
    print("(Assuming single-threaded processing)")
    print("-" * 40)

    for model_name, tokens_per_sec in tokens_per_second_estimates.items():
        total_seconds = estimated_tokens / tokens_per_sec

        if total_seconds < 3600:  # Less than an hour
            time_str = f"{total_seconds/60:.1f} minutes"
        elif total_seconds < 86400:  # Less than a day
            time_str = f"{total_seconds/3600:.1f} hours"
        else:  # Days or more
            time_str = f"{total_seconds/86400:.1f} days"

        print(f"{model_name:.<35} {time_str}")

    print("\n💰 COST ESTIMATES (if using cloud APIs)")
    print("-" * 40)

    # Cloud API pricing estimates (per 1M tokens, as of 2024)
    cloud_pricing = {
        'OpenAI GPT-4': 30.0,
        'OpenAI GPT-3.5': 2.0,
        'Anthropic Claude-3 Haiku': 1.25,
        'Anthropic Claude-3 Sonnet': 15.0,
        'Google Gemini Pro': 7.0
    }

    millions_of_tokens = estimated_tokens / 1_000_000

    for service, price_per_million in cloud_pricing.items():
        cost = millions_of_tokens * price_per_million
        print(f"{service:.<35} ${cost:.2f}")

    print("\n🔍 SAMPLE COMMENTS")
    print("-" * 40)
    for i, comment in enumerate(sample_comments, 1):
        print(f"\nSample {i} ({comment['length']} chars):")
        print(f"Artist: {comment['artist'] or 'Unknown'}")
        print(f"Song: {comment['song'] or 'Unknown'}")
        preview = comment['comment'][:200] + \
            "..." if len(comment['comment']) > 200 else comment['comment']
        print(f"Comment: {preview}")

    print("\n🚀 RECOMMENDATIONS")
    print("-" * 40)

    if estimated_tokens < 1_000_000:  # Less than 1M tokens
        complexity = "LOW"
        recommendation = "Very feasible with any local model"
    elif estimated_tokens < 10_000_000:  # Less than 10M tokens
        complexity = "MEDIUM"
        recommendation = "Feasible with local models, consider batch processing"
    elif estimated_tokens < 100_000_000:  # Less than 100M tokens
        complexity = "HIGH"
        recommendation = "Requires careful planning, use efficient models"
    else:
        complexity = "VERY HIGH"
        recommendation = "Consider sampling or distributed processing"

    print(f"Complexity: {complexity}")
    print(f"Recommendation: {recommendation}")

    # Batching suggestions
    batch_sizes = [100, 500, 1000]
    print(f"\nSuggested batch sizes for processing:")
    for batch_size in batch_sizes:
        num_batches = (substantial_comments + batch_size - 1) // batch_size
        print(f"- {batch_size} comments per batch = {num_batches} batches")

    # Memory estimates
    print(f"\nEstimated memory requirements:")
    print(f"- Text data: ~{total_chars / (1024*1024):.1f} MB")
    # Rough estimate
    print(
        f"- With model context: ~{estimated_tokens * 2 / (1024*1024):.1f} MB")

    return {
        'total_comments': substantial_comments,
        'total_tokens': estimated_tokens,
        'total_chars': total_chars,
        'avg_length': avg_length,
        'complexity': complexity
    }


if __name__ == "__main__":
    try:
        results = analyze_kexp_feasibility()

        # Save results for later use
        if results:
            with open('kexp_analysis_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Analysis saved to 'kexp_analysis_results.json'")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Make sure KEXP_DB_PATH is set and database is accessible")
