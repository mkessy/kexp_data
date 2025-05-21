# Annotation Schema for KEXP DJ Comments

This document outlines the schema for SpanCat (Span Categorization) tasks for annotating KEXP DJ comments.

## SpanCat Labels (Span Categorization)

These labels categorize spans of text (which can range from short phrases to multiple sentences) based on the type of information they convey.

### `_SPAN` Suffix Labels (Sentence or Multi-Sentence Level):

1.  **`ARTIST_BIO_LINE_SPAN`**: A declarative sentence or multi-sentence passage providing biographical information about an artist or group. This includes their origin, history, key characteristics, significant achievements, or formation details.

    - _Example_: "The Beatles were an English rock band formed in Liverpool in 1960, comprising John Lennon, Paul McCartney, George Harrison and Ringo Starr."
    - _Typically contains_: SpanCat Tags (`ARTIST_NAME_TAG`, `GENRE_TAG`, `ROLE_TAG`).

2.  **`RELEASE_ANNOUNCEMENT_SPAN`**: A sentence or sentences specifically announcing an upcoming or new release (album, EP, single, etc.), often including its title, the artist's release count (e.g., "4th studio record"), and an explicit release date or timeframe.

    - _Example_: "Their new album, 'Cosmic Echoes', is their third studio effort and is slated for release on October 26th."
    - _Typically contains_: SpanCat Tags (`ALBUM_TITLE_TAG`, `ARTIST_NAME_TAG`, `GENRE_TAG`).

3.  **`SONG_ATTRIBUTION_SPAN`**: A sentence or sentences stating the origin or source of the _currently playing song_ or a song that is the immediate subject of discussion (e.g., pointing to an album, release year, or its status).

    - _Example_: "That was 'Starman' from David Bowie's 1972 classic album 'The Rise and Fall of Ziggy Stardust and the Spiders from Mars'."
    - _Example_: "Next up, a track from their forthcoming EP."
    - _Typically contains_: SpanCat Tags (`ALBUM_TITLE_TAG`, `SONG_TITLE_TAG`, `ARTIST_NAME_TAG`).

4.  **`ARTIST_STATEMENT_SPAN`** _(For later addition)_: A span capturing an artist/group's attributed statement, structured generally as: `[Artist/Group Name]` + `[Verb phrase indicating speech, e.g., "said", "stated", "revealed"]` + `[Optional time/context, e.g., "in an interview with X", "yesterday", "during their concert at Y"]` + `["Actual quote text"]`.

    - _Example_: "Björk stated in a recent interview with 'Music Weekly', \"Nature is my biggest inspiration for this new record.\""
    - _Typically contains_: SpanCat Tags (`ARTIST_NAME_TAG`, `VENUE_NAME_TAG` or `EVENT_NAME_TAG` if the context is an event/venue, and the actual quote text would be part of the span itself, not a separate `QUOTE_TEXT` NER).

5.  **`GROUP_COMPOSITION_SPAN`**: A sentence or sentences listing or describing the members of a group/band and potentially their roles.

    - _Example_: "The supergroup consists of Alice Alpha on vocals, Bob Beta on guitar, and Charlie Gamma on drums."
    - _Typically contains_: SpanCat Tags (`ARTIST_NAME_TAG` for group and members, `ROLE_TAG`).

6.  **`RECORD_LABEL_INFO_SPAN`**: A sentence or sentences mentioning a record label in relation to an artist or release.

    - _Example_: "Their early demos were picked up by IndieGiant Records, who released their first two albums."
    - _Typically contains_: SpanCat Tags (`RECORD_LABEL_NAME_TAG`, `ARTIST_NAME_TAG`, `ALBUM_TITLE_TAG`).

7.  **`EVENT_INFO_SPAN`**: A sentence or sentences providing details about a specific performance, festival, tour, or other event, including who, what, where, and when.
    - _Example_: "Catch them live at the Paramount Theatre next Friday, June 15th, as part of their 'Electric Dreams' North American tour."
    - _Typically contains_: SpanCat Tags (`ARTIST_NAME_TAG`, `EVENT_NAME_TAG`, `VENUE_NAME_TAG`).

### `_TAG` Suffix Labels (Short, N-Gram Level Phrases):

8.  **`ARTIST_NAME_TAG`**: The name of a musical artist, band, or group.
    - _Example_: "Fleetwood Mac", "Pink Floyd", "Kendrick Lamar"
9.  **`ALBUM_TITLE_TAG`**: The title of an album.
    - _Example_: "Rumours", "The Dark Side of the Moon"
10. **`SONG_TITLE_TAG`**: The title of a song.
    - _Example_: "Go Your Own Way", "Money", "Alright"
11. **`RECORD_LABEL_NAME_TAG`**: The name of a record label.
    - _Example_: "Sub Pop Records", "Warner Bros.", "Top Dawg Entertainment"
12. **`ARTIST_GROUP_LOC_ROLE_TAG`**: A short, descriptive phrase (typically not a full sentence, often adjectival or appositive) identifying an artist/group along with their location and/or role(s).
    - _Example_: "Seattle-based indie-folk band The Lumineers", "legendary producer Quincy Jones", "the Parisian electronic duo Daft Punk"
    - _Typically contains_: SpanCat Tags (`ARTIST_NAME_TAG`, `GENRE_TAG`, `ROLE_TAG`).
13. **`GENRE_TAG`**: A short span (typically 1-3 n-grams, maybe slightly more for subgenres) explicitly naming a musical genre or style.
    - _Example_: "rock", "post-punk", "RAP", "electronic", "symphonic metal", "acid jazz"
14. **`ROLE_TAG`**: A short span (typically 1-3 n-grams) explicitly naming an artist's role, profession, or instrument.
    - _Example_: "producer", "DJ", "harpist", "composer", "vocalist", "lead guitarist", "activist"
15. **`EVENT_NAME_TAG`**: A short span naming a specific event, festival, recurring show, or tour.
    - _Example_: "Coachella 2023", "KEXP's Audioasis", "The 'Endless Summer' Tour", "Bumbershoot"
16. **`VENUE_NAME_TAG`**: A short span naming a specific venue, club, arena, or place of performance/event.
    - _Example_: "Neumos", "The Fillmore", "Madison Square Garden", "The Tractor Tavern"
