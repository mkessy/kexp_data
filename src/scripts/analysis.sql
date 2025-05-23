-- KEXP Database Analysis Queries
-- Run these in SQLite to understand data size and processing requirements

-- 1. Basic counts and overview
SELECT 'Total plays' AS metric, COUNT(*) AS count FROM plays
UNION ALL
SELECT 'Plays with comments', COUNT(*) FROM plays WHERE comment IS NOT NULL AND TRIM(comment) != ''
UNION ALL
SELECT 'Plays with substantial comments (20+ chars)', COUNT(*) FROM plays WHERE comment IS NOT NULL AND LENGTH(TRIM(comment)) >= 20
UNION ALL
SELECT 'Total songs', COUNT(*) FROM songs
UNION ALL
SELECT 'Total artists', COUNT(DISTINCT artist) FROM songs
UNION ALL
SELECT 'Total shows', COUNT(*) FROM shows
UNION ALL
SELECT 'Date range plays', NULL;

-- 2. Date range of data
SELECT 
    'Earliest play' AS metric,
    MIN(airdate) AS value
FROM plays
UNION ALL
SELECT 
    'Latest play',
    MAX(airdate)
FROM plays
UNION ALL
SELECT 
    'Years of data',
    CAST((julianday(MAX(airdate)) - julianday(MIN(airdate))) / 365.25 AS INTEGER)
FROM plays;

-- 3. Comment length analysis
SELECT 
    'Average comment length (chars)' AS metric,
    CAST(AVG(LENGTH(comment)) AS INTEGER) AS value
FROM plays 
WHERE comment IS NOT NULL AND TRIM(comment) != ''
UNION ALL
SELECT 
    'Median comment length (approx)',
    (SELECT LENGTH(comment) FROM plays WHERE comment IS NOT NULL AND TRIM(comment) != '' ORDER BY LENGTH(comment) LIMIT 1 OFFSET (SELECT COUNT(*)/2 FROM plays WHERE comment IS NOT NULL AND TRIM(comment) != ''))
UNION ALL
SELECT 
    'Max comment length',
    MAX(LENGTH(comment))
FROM plays 
WHERE comment IS NOT NULL AND TRIM(comment) != ''
UNION ALL
SELECT 
    'Min comment length (substantial)',
    MIN(LENGTH(comment))
FROM plays 
WHERE comment IS NOT NULL AND LENGTH(TRIM(comment)) >= 20;

-- 4. Token estimation (rough approximation: chars/4.5 for English)
SELECT 
    'Total comment characters' AS metric,
    SUM(LENGTH(comment)) AS value
FROM plays 
WHERE comment IS NOT NULL AND TRIM(comment) != ''
UNION ALL
SELECT 
    'Estimated total tokens (chars/4.5)',
    CAST(SUM(LENGTH(comment)) / 4.5 AS INTEGER)
FROM plays 
WHERE comment IS NOT NULL AND TRIM(comment) != ''
UNION ALL
SELECT 
    'Estimated tokens (20+ char comments)',
    CAST(SUM(LENGTH(comment)) / 4.5 AS INTEGER)
FROM plays 
WHERE comment IS NOT NULL AND LENGTH(TRIM(comment)) >= 20;

-- 5. Comments by year (to see data distribution)
SELECT 
    strftime('%Y', airdate) AS year,
    COUNT(*) AS total_plays,
    COUNT(CASE WHEN comment IS NOT NULL AND TRIM(comment) != '' THEN 1 END) AS plays_with_comments,
    CAST(AVG(CASE WHEN comment IS NOT NULL AND TRIM(comment) != '' THEN LENGTH(comment) END) AS INTEGER) AS avg_comment_length
FROM plays 
GROUP BY strftime('%Y', airdate)
ORDER BY year;

-- 6. Top DJs by comment volume (to understand who writes most)
SELECT 
    COALESCE(sh.host_names, 'Unknown') AS dj_names,
    COUNT(*) AS plays_with_comments,
    CAST(AVG(LENGTH(p.comment)) AS INTEGER) AS avg_comment_length,
    SUM(LENGTH(p.comment)) AS total_comment_chars
FROM plays p
LEFT JOIN shows sh ON p.show = sh.id
WHERE p.comment IS NOT NULL AND TRIM(p.comment) != ''
GROUP BY sh.host_names
ORDER BY plays_with_comments DESC
LIMIT 10;

-- 7. Sample comments for manual inspection
SELECT 
    p.id as play_id,
    p.airdate,
    s.artist,
    s.song,
    s.album,
    sh.host_names,
    LENGTH(p.comment) as comment_length,
    SUBSTR(p.comment, 1, 200) || '...' as comment_preview
FROM plays p
LEFT JOIN songs s ON p.song_id = s.song_id
LEFT JOIN shows sh ON p.show = sh.id
WHERE p.comment IS NOT NULL 
    AND LENGTH(TRIM(p.comment)) >= 50
ORDER BY RANDOM()
LIMIT 20;

-- 8. Comment complexity indicators (URLs, mentions, formatting)
SELECT 
    'Comments with URLs' AS metric,
    COUNT(*) AS count
FROM plays 
WHERE comment IS NOT NULL 
    AND (comment LIKE '%http%' OR comment LIKE '%.com%' OR comment LIKE '%.org%')
UNION ALL
SELECT 
    'Comments with email-like patterns',
    COUNT(*)
FROM plays 
WHERE comment IS NOT NULL AND comment LIKE '%@%'
UNION ALL
SELECT 
    'Comments with line breaks',
    COUNT(*)
FROM plays 
WHERE comment IS NOT NULL AND (comment LIKE '%\n%' OR comment LIKE '%\r%')
UNION ALL
SELECT 
    'Very long comments (500+ chars)',
    COUNT(*)
FROM plays 
WHERE comment IS NOT NULL AND LENGTH(comment) >= 500;

-- 9. Artist/Album diversity in commented plays
SELECT 
    'Unique artists in commented plays' AS metric,
    COUNT(DISTINCT s.artist) AS count
FROM plays p
JOIN songs s ON p.song_id = s.song_id
WHERE p.comment IS NOT NULL AND TRIM(p.comment) != ''
UNION ALL
SELECT 
    'Unique albums in commented plays',
    COUNT(DISTINCT s.album)
FROM plays p
JOIN songs s ON p.song_id = s.song_id
WHERE p.comment IS NOT NULL AND TRIM(p.comment) != '' AND s.album IS NOT NULL;

-- 10. Processing time estimation query
-- This gives you a sample to estimate processing time per comment
SELECT 
    COUNT(*) as sample_count,
    AVG(LENGTH(comment)) as avg_length,
    SUM(LENGTH(comment)) as total_chars,
    CAST(SUM(LENGTH(comment)) / 4.5 AS INTEGER) as estimated_tokens
FROM (
    SELECT comment 
    FROM plays 
    WHERE comment IS NOT NULL AND LENGTH(TRIM(comment)) >= 20
    ORDER BY RANDOM() 
    LIMIT 1000
);