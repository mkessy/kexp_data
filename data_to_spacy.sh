prodigy data-to-spacy ./data/spacy_training_data_music_only/ \
    --ner kexp_ner_annotations_v2 \
    --eval-split 0.2 \
    -l ARTIST_NAME,SONG_TITLE,ALBUM_TITLE,VENUE_NAME,LOCATION,DATE_EXPR