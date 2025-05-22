# Annotation Schema for KEXP DJ Comments (Version 4.0)

This document outlines the schema for SpanCat (Span Categorization) tasks for annotating KEXP DJ comments. The goal is to capture detailed information about artists, music, events, their descriptions, and interrelations.

## SpanCat Labels (Span Categorization)

Labels are divided into `_SPAN` (typically sentence or multi-sentence level, capturing a complete thought or piece of information) and `_TAG` (typically short, n-gram level phrases, representing specific entities or descriptors).

### `_SPAN` Suffix Labels (Contextual Information Spans):

1.  **`ARTIST_BIO_SPAN`**: A declarative sentence or multi-sentence passage providing general biographical information about an artist or group, excluding specific origin phrases covered by `ARTIST_LOC_ORGIN_SPAN` or alias definitions covered by `ARTIST_ALIAS_SPAN`.

    - _Example_: "John Heathers aka BigFox is a DJ/Producer/multi-instrumentalist who grew up in Bamako. He works there with his momma and his big dog leo" (The part about being a DJ/Producer... and his personal life in Bamako).
    - _Typically contains_: `ARTIST_TAG`, `ROLE_TAG`, `LOC_TAG`, `DATE_TAG`, `GENRE_TAG`.

2.  **`ARTIST_LOC_ORGIN_SPAN`**: A phrase or sentence specifically stating an artist's geographical origin, where they are primarily based, or a descriptive phrase linking them to a location and often a genre/role.

    - _Example_: "Seattle post-punk band The Rangers"
    - _Example_: "from the Brooklyn DJ/Producer known as..."
    - _Example_: "Parcels are an Australian band originally from Byron Bay, now based in Berlin."
    - _Typically contains_: `ARTIST_TAG`, `LOC_TAG`, `ROLE_TAG`, `GENRE_TAG`.

3.  **`ARTIST_ALIAS_SPAN`**: A phrase or sentence that explicitly states an alias, pseudonym, stage name, or a "known as" name for an artist.

    - _Example_: "John Heathers aka BigFox"
    - _Example_: "Rita Jones performs as METRONOME"
    - _Example from samples_: "Beatrice Laus, known professionally as beabadoobee"
    - _Typically contains_: Multiple `ARTIST_TAG`s (the real name and the alias).

4.  **`NEW_RELEASE_SPAN`**: A sentence or sentences specifically announcing an upcoming or new musical release (album, EP, single, etc.), often including its title, the artist's release count, and an explicit release date or timeframe.

    - _Example_: "Their new album, 'Cosmic Echoes', is their third studio effort and is slated for release on October 26th."
    - _Example from samples_: "New single from Turnstile, released TODAY from their forthcoming album due out June 6th."
    - _Typically contains_: `ALBUM_TAG` (or `SONG_TAG` for singles), `ARTIST_TAG`, `DATE_TAG`.

5.  **`SONG_ATTRIBUTION_SPAN` (Please confirm if this should be kept/merged)**: A sentence or sentences stating the origin or source of the _currently playing song_ or a song that is the immediate subject of discussion (e.g., pointing to an album, release year, or its status).

    - _Example_: "That was 'Starman' from David Bowie's 1972 classic album 'The Rise and Fall of Ziggy Stardust and the Spiders from Mars'."
    - _Example from samples_: "Mountain Song was on their second album, Nothing's Shocking, released in 1988."
    - _Typically contains_: `SONG_TAG`, `ALBUM_TAG`, `ARTIST_TAG`, `DATE_TAG`.

6.  **`SOUND_DESCRIPTION_SPAN`**: A sentence or multi-sentence passage describing the sound, style, musical characteristics, or general lyrical themes of an album, song, or artist's musical output.

    - _Example_: "The debut album from this Portland-based, Nashville-born artist is an often-poignant set of well-crafted folk-pop with a lush sound combining guitars, mandolin, strings, horns, woodwinds, field recordings and more with his serene vocals and evocative lyrics exploring his family heritage, racism and the hardships of immigration."
    - _Typically contains_: `GENRE_TAG`, `INSTRUMENT_TAG`, `ARTIST_TAG`, `ALBUM_TAG`, `SONG_TAG`.

7.  **`THEME_INSPO_MENTION_SPAN`**: A sentence or phrase that explicitly mentions lyrical themes, conceptual inspirations, or the story/meaning behind a specific song or album. More focused than `SOUND_DESCRIPTION_SPAN`.

    - _Example_: "He mentioned the entire album is a concept piece about the migratory patterns of birds."
    - _Example from samples_: "The broader themes of the LOVED album will focus the connection between the members of the group, the audience, and the music itself – building suspense of a story yet told."
    - _Typically contains_: `ALBUM_TAG` or `SONG_TAG`, and descriptive text about themes/inspiration.

8.  **`ARTIST_QUOTE_SPAN`**: A span capturing an artist/group's attributed statement (a direct or reported quote). This includes the quoted text and often the context of the statement.

    - _Example_: "Björk stated in a recent interview, \"Nature is my biggest inspiration.\""
    - _Typically contains_: `ARTIST_TAG` (speaker), actual quoted text.

9.  **`GROUP_COMP_SPAN`**: A sentence or sentences listing or describing the members of a group/band, often including their roles or instruments.

    - _Example_: "The supergroup consists of Alice Alpha on vocals, Bob Beta on guitar, and Charlie Gamma on drums."
    - _Typically contains_: `ARTIST_TAG` (for group and individual members), `ROLE_TAG`, `INSTRUMENT_TAG`.

10. **`COLLAB_MENTION_SPAN`**: Specifically captures a collaboration between distinct artists/groups on a song, album, or performance.

    - _Example_: "This track is a collaboration between DJ Shadow and Run the Jewels."
    - _Example from samples_: "\"The Medium\" features Unknown Mortal Orchestra's Ruban Nielson on lead guitar..." (The phrase "features Unknown Mortal Orchestra's Ruban Nielson" including the artists involved).
    - _Typically contains_: Multiple `ARTIST_TAG`s, and potentially `SONG_TAG` or `ALBUM_TAG`.

11. **`INFLUENCE_MENTION_SPAN`**: A sentence or phrase stating that an artist, song, or album is influenced by another artist, work, or genre.

    - _Example_: "Her vocal style is heavily influenced by Billie Holiday."
    - _Example_: "You can hear a clear homage to Kraftwerk in their synth lines."
    - _Typically contains_: `ARTIST_TAG` (both the influenced and influencer if mentioned), `SONG_TAG`, `ALBUM_TAG`, `GENRE_TAG`.

12. **`RECORD_LABEL_SPAN`**: A sentence or sentences providing information about a record label in relation to an artist or release, beyond just mentioning the name (which would be `RECORD_LABEL_TAG`).

    - _Example_: "Their early demos were picked up by IndieGiant Records, who then funded their first two albums."
    - _Typically contains_: `RECORD_LABEL_TAG`, `ARTIST_TAG`, `ALBUM_TAG`.

13. **`SHOW_DATE_SPAN`**: A sentence or clause detailing one or more upcoming or past shows for an artist, including dates, venues, and locations.

    - _Example_: "The Beatles are playing in Seattle at Neumos on Jan 23 and in Portland at the Shithole on Feb 21."
    - _Example_: "You can catch them at the Paramount next Tuesday, or in LA the following Friday."
    - _Typically contains_: `ARTIST_TAG`, `VENUE_TAG`, `LOC_TAG`, `DATE_TAG`, `EVENT_TAG` (if part of a named tour/festival).

14. **`SEE_MORE_SPAN`**: A phrase, often including a URL or a call to action, directing listeners to find more information, listen to archives, or visit a website.
    - _Example_: "For more details on their tour, visit their website at artist.com."
    - _Example from samples_: "You can watch the music video here: [URL]"
    - _Typically contains_: URLs, calls to action like "listen here", "find out more".

### `_TAG` Suffix Labels (Specific Entity & Descriptor Tags):

15. **`ARTIST_TAG`**: The name of a musical artist, band, or group.

    - _Example_: "Fleetwood Mac", "Parcels", "beabadoobee", "John Heathers", "BigFox".

16. **`ALBUM_TAG`**: The title of an album or EP.

    - _Example_: "Rumours", "LOVED", "Kala", "MAHAL".

17. **`SONG_TAG`**: The title of a song.

    - _Example_: "Go Your Own Way", "Mountain Song", "Bird Flu".

18. **`RECORD_LABEL_TAG`**: The name of a record label.

    - _Example_: "Sub Pop Records", "Merge Records".

19. **`GENRE_TAG`**: A short span explicitly naming a musical genre or style.

    - _Example_: "rock", "post-punk", "J-Pop", "shoegaze", "indiepop".

20. **`ROLE_TAG`**: A short span explicitly naming an artist's role, profession, or primary instrument.

    - _Example_: "producer", "DJ", "vocalist", "guitarist", "multi-instrumentalist".

21. **`EVENT_TAG`**: A short span naming a specific event, festival, recurring show series, or tour (past, present, or upcoming).

    - _Example_: "Coachella 2023", "Lollapalooza", "KEXP's Audioasis", "The 'Endless Summer' Tour".

22. **`VENUE_TAG`**: A short span naming a specific venue, club, arena, or place of performance/event.

    - _Example_: "Neumos", "The Fillmore", "KEXP studio".

23. **`INSTRUMENT_TAG`**: A short span naming a musical instrument, a specific vocal style, or a distinct sound element.

    - _Example_: "guitars", "mandolin", "serene vocals", "drum machine", "synthesizer".

24. **`STUDIO_TAG`**: The name of a recording studio.

    - _Example_: "Abbey Road Studios", "Electric Lady Studios".

25. **`LOC_TAG`**: A short span naming a geographical location (city, state, country, region).

    - _Example_: "Seattle", "Brooklyn", "Kobe, Japan", "Byron Bay", "Australian".

26. **`DATE_TAG`**: A short span representing a specific date, year, decade, or relative time expression.
    - _Example_: "1960", "October 26th", "next month", "Jan 23", "2022", "the 90's", "TODAY".

### Semantic Relations & Richer Knowledge (Derived/Inferred)

The annotation of the above spans and tags will facilitate the extraction and inference of these semantic relationships to build a comprehensive knowledge graph:

- **`IS_ALIAS_OF`**: (Artist A `ARTIST_TAG`, Artist B `ARTIST_TAG`) - Derived from `ARTIST_ALIAS_SPAN`.
- **`HAS_MEMBER / MEMBER_OF`**: (Group `ARTIST_TAG`, Member `ARTIST_TAG`) - Derived from `GROUP_COMP_SPAN`.
- **`COLLABORATED_WITH`**: (Artist A `ARTIST_TAG`, Artist B `ARTIST_TAG`) - Derived from `COLLAB_MENTION_SPAN`.
- **`INFLUENCED_BY / INFLUENCES`**: (Artist/Work A, Artist/Work/Genre B) - Derived from `INFLUENCE_MENTION_SPAN`.
- **`HAS_THEME_OR_INSPIRATION`**: (Song/Album, Textual Description) - Derived from `THEME_INSPO_MENTION_SPAN`.
- **`ATTRIBUTED_QUOTE`**: (Artist `ARTIST_TAG`, Quoted Text) - Derived from `ARTIST_QUOTE_SPAN`.
- **`PERFORMED_AT_EVENT`**: (Artist `ARTIST_TAG`, Event `EVENT_TAG`) - Derived from `SHOW_DATE_SPAN`.
- **`EVENT_AT_VENUE_ON_DATE_IN_LOCATION`**: (Event `EVENT_TAG`, Venue `VENUE_TAG`, Date `DATE_TAG`, Location `LOC_TAG`) - Derived from `SHOW_DATE_SPAN`.
- **`ORIGINATES_FROM`**: (Artist `ARTIST_TAG`, Location `LOC_TAG`) - Derived from `ARTIST_LOC_ORGIN_SPAN`.

This version incorporates your latest list and clarifications. Please review it, especially the definition of `SHOW_DATE_SPAN` and the re-inclusion of `SONG_ATTRIBUTION_SPAN` for your consideration.
