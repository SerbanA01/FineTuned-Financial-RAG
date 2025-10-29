import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any

import ftfy
import nltk
from nltk.tokenize import sent_tokenize

# --- NLTK Punkt Tokenizer Check ---
# The 'punkt' tokenizer is essential for sentence segmentation. This block ensures it's
# downloaded and available before any processing begins, preventing runtime errors.
try:
    nltk.data.find('tokenizers/punkt')
except (nltk.downloader.DownloadError, LookupError):
    logging.info("NLTK 'punkt' model not found or outdated. Downloading/Updating...")
    nltk.download('punkt', quiet=True)

def normalize_text_generic(text: str) -> str:
    """
    Performs general text normalization and fixes common encoding issues.

    This function uses the `ftfy` library to resolve Unicode Mojibake and then
    replaces common non-standard punctuation (like smart quotes) with their
    standard ASCII equivalents.

    @param text: The input string to normalize.
    @return: The normalized string.
    """
    if not text: return ""
    # Use ftfy to fix text encoding issues, like Mojibake.
    text = ftfy.fix_text(text)
    # Manually replace specific non-standard Unicode characters with standard equivalents.
    replacements = {'\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
                    '\u2022': '* ', '\u2013': '-', '\u2014': '--', '\u00A0': ' ', '�': "'"}
    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)
    return text

def clean_speech_text(text: str) -> str:
    """
    Removes common conversational artifacts and boilerplate from transcript text.

    This function targets specific phrases often inserted by operators or transcription
    services (e.g., "[Operator Instructions]") that are not part of the core dialogue.

    @param text: The speech text to be cleaned.
    @return: The cleaned speech text.
    """
    if not text: return ""
    # Collapse multiple spaces or tabs into a single space.
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # A list of regex patterns for common non-speech artifacts to be removed.
    noise_patterns = [
        r'\[Operator Instructions\]', r'\[Technical Issues\]', r'\[Phonetic\]',
        r'\[Indecipherable\]', r'\[Speech Overlap\]', r'\[Foreign Speech\]',
        r'Sir, please go ahead\.', r'Please go ahead\.',
        r'Today\'s conference is being recorded\.',
        r'\(Operator Instructions\)'
    ]
    for pattern_str in noise_patterns:
        text = re.sub(pattern_str, '', text, flags=re.IGNORECASE)
    return text.strip()

def sentence_segment_and_filter(text: str, min_sentence_words: int = 2) -> str:
    """
    Segments text into sentences, filters out duplicates and very short sentences.

    Uses NLTK's sentence tokenizer and provides a regex-based fallback. It ensures
    that the final output is a clean, coherent block of text composed of valid,
    unique sentences.

    @param text: A block of text to be segmented.
    @param min_sentence_words: The minimum number of words a sentence must have to be included.
    @return: A single string containing the filtered and joined sentences.
    """
    if not text: return ""
    # Normalize newlines to spaces for consistent sentence tokenization.
    text_single_line = re.sub(r'\s*\n\s*', ' ', text).strip()
    if not text_single_line: return ""

    # Attempt to use the more robust NLTK tokenizer first.
    try:
        sents_original = sent_tokenize(text_single_line)
    # If NLTK fails (e.g., on unusual input), fall back to a simpler regex-based split.
    except Exception:
        sents_original = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_single_line) if s.strip()]

    valid_sentences = []
    seen_sentences_norm = set()
    # Iterate through sentences to filter and deduplicate.
    for s in sents_original:
        s_stripped = s.strip()
        if not s_stripped: continue
        # Normalize for case-insensitive duplicate checking.
        norm_s = re.sub(r'\s+', ' ', s_stripped).lower()
        # A sentence is valid if it meets the minimum word count and has not been seen before.
        if len(s_stripped.split()) >= min_sentence_words and norm_s not in seen_sentences_norm:
            valid_sentences.append(s_stripped)
            seen_sentences_norm.add(norm_s)
    return ' '.join(valid_sentences)

def is_potential_speaker_line_with_titles(line: str, participant_list_full: list = None) -> str | None:
    """
    Uses heuristics to determine if a line is a speaker identifier (e.g., "John Doe -- CEO").

    This function is crucial for parsing the transcript structure. It checks against a
    pre-extracted list of participants and uses several regex patterns to identify
    lines that introduce a new speaker.

    @param line: The line of text to check.
    @param participant_list_full: A list of known participant strings to match against.
    @return: The matched speaker string if found, otherwise None.
    """
    line_stripped = line.strip()
    # Short-circuit for empty or overly long lines which are unlikely to be speaker names.
    if not line_stripped or len(line_stripped) > 150:
        return None

    # High-confidence check: See if the line exactly matches a known participant from the header.
    if participant_list_full:
        # Sort by length to match longer names first (e.g., "John A. Smith" before "John Smith").
        sorted_participants_full = sorted(participant_list_full, key=len, reverse=True)
        for p_full_string in sorted_participants_full:
            if re.fullmatch(re.escape(p_full_string) + r":?", line_stripped, re.IGNORECASE):
                return p_full_string.rstrip(':')
            if line_stripped.lower().startswith(p_full_string.lower() + ":"):
                 return p_full_string

    # Heuristic 1: Match "Name -- Title" format.
    match = re.fullmatch(r"([\w\s\.'-]+?)\s*--\s*([\w\s,&'\.-]+(?:Relations|Officer|Analyst|Operator|Moderator|Host|Chairman|Chairwoman)?[\w\s,\.'-]*)$", line_stripped)
    if match:
        return line_stripped

    # Heuristic 2: Match common generic roles like "Operator".
    match = re.fullmatch(r"(Operator|Moderator|Host)", line_stripped, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Heuristic 3: Match "Proper Name:" format, then try to enrich it with a title if available.
    name_colon_match = re.fullmatch(r"([A-Z][a-z'.]+(?:\s[A-Z][a-z'.]+){0,3}):", line_stripped)
    if name_colon_match:
        name_only = name_colon_match.group(1).strip()
        if participant_list_full:
            for p_full in participant_list_full:
                if p_full.lower().startswith(name_only.lower()):
                    # Prefer the full name with title if a match is found.
                    if "--" in p_full or any(role.lower() in p_full.lower() for role in ["analyst", "officer", "director"]):
                        return p_full
        return name_only # Return the simple name if no enriched version is found.

    # Heuristic 4: Lower-confidence check for short, capitalized names without punctuation.
    words = line_stripped.split()
    if 1 <= len(words) <= 4 and all(word[0].isupper() for word in words) and not re.search(r"[.!?,;]$", line_stripped):
        # Exclude common conversational phrases that might be capitalized.
        if line_stripped.lower() not in ["thank you", "good morning", "good afternoon", "good evening", "questions and answers", "prepared remarks"]:
            name_only = line_stripped
            if participant_list_full:
                for p_full in participant_list_full:
                    if p_full.lower().startswith(name_only.lower()):
                         if "--" in p_full or any(role.lower() in p_full.lower() for role in ["analyst", "officer", "director"]):
                            return p_full
            return name_only
    return None

def enrich_metadata_from_content(first_lines: list, current_metadata: dict) -> dict:
    """
    Scans the beginning of a transcript to find missing metadata like year, quarter, or company name.

    This acts as a fallback for when the initial metadata is incomplete, improving
    the quality and searchability of the final processed data.

    @param first_lines: The first few lines of the transcript content.
    @param current_metadata: The existing metadata dictionary.
    @return: The metadata dictionary, potentially with new fields added.
    """
    text_to_search = " ".join(first_lines)

    # Attempt to find year and quarter if they are missing.
    if current_metadata.get("year", "YYYY") == "YYYY" or current_metadata.get("quarter", "QX") == "QX":
        year_quarter_match = re.search(r"(\d{4})\s+(First|Second|Third|Fourth)\s+Quarter", text_to_search, re.IGNORECASE)
        if year_quarter_match:
            if current_metadata.get("year", "YYYY") == "YYYY":
                current_metadata["year"] = year_quarter_match.group(1)
            
            if current_metadata.get("quarter", "QX") == "QX":
                quarter_str_from_text = year_quarter_match.group(2).lower()
                q_map = {"first": "Q1", "second": "Q2", "third": "Q3", "fourth": "Q4"}
                current_metadata["quarter"] = q_map.get(quarter_str_from_text, "QX")

    # Attempt to extract the company name from common introductory phrases.
    if not current_metadata.get("company_name") or current_metadata.get("company_name", "").lower() == "company":
        company_patterns = [
            r"welcome to the ([\w\s\.'&-]+?)(?:\s+\d{4}\s+(?:First|Second|Third|Fourth)\s+Quarter|\s+earnings conference call)",
            r"Operator:\s*Good day, and welcome to the ([\w\s\.'&-]+?)\s+(?:Q\d\s\d{4}\s)?(?:Earnings )?Conference Call",
             r"^([\w\s\.'&-]+?)\s+\([A-Z0-9:\.]+?\)\s+Q\d\s+\d{4}\s+Earnings\s+Call",
            r"(?:conference call for|earnings call of|results of)\s+([\w\s\.'&-]+?)(?:,|Company|Inc\.?|LLC\.?|$)"
        ]
        for pattern in company_patterns:
            company_match = re.search(pattern, text_to_search, re.IGNORECASE)
            if company_match:
                potential_name = company_match.group(1).strip()
                # Ensure the extracted name is meaningful.
                if potential_name and potential_name.lower() not in ["company", "the company"]:
                    current_metadata["company_name"] = potential_name
                    break 
    return current_metadata

def process_transcript_to_json_speaker_turns(
    transcript_content: str,
    output_filename: str,
    base_doc_metadata: dict
):
    """
    Main orchestration function to process a raw transcript into a structured JSON file.

    This function implements a state machine that reads the transcript line by line,
    identifies speakers and sections (e.g., Prepared Remarks, Q&A), cleans the dialogue,
    and aggregates it into a list of "speaker turns".

    @param transcript_content: The raw string content of the earnings call transcript.
    @param output_filename: The path where the final JSON output will be saved.
    @param base_doc_metadata: A dictionary of initial metadata (ticker, date, etc.).
    """
    processing_target_id = base_doc_metadata.get('conceptual_filename', base_doc_metadata.get('ticker', 'UNKNOWN'))
    logging.info(f"Processing transcript for {processing_target_id} -> {output_filename}")

    if not transcript_content or not transcript_content.strip():
        logging.warning(f"Empty transcript content for {processing_target_id}. Skipping.")
        return

    # 1. Enrich initial metadata using the transcript content itself.
    doc_metadata = base_doc_metadata.copy()
    doc_metadata = enrich_metadata_from_content(transcript_content.splitlines()[:30], doc_metadata)

    # 2. Perform initial normalization on the entire document.
    full_content = normalize_text_generic(transcript_content)
    
    # 3. Extract the "Participants" section to get a high-confidence list of speakers.
    participants_full_strings = []
    participants_section_match = re.search(
        # This regex looks for a participants header and captures the block of text that follows.
        r"(?:Call Participants|Participants|Conference Call Participants|Executives|Analysts|Speakers):\s*\n(.*?)(?:\n\n|\n[A-ZÀ-ÖØ-Þ][\w\s,&'\.-]+:|Prepared Remarks:|Questions and Answers:)",
        full_content, re.IGNORECASE | re.DOTALL
    )
    if participants_section_match:
        participants_section = participants_section_match.group(1)
        raw_lines_or_comma_separated = []
        # Handle both line-separated and comma-separated participant lists.
        for line in participants_section.splitlines():
            if ',' in line and '--' not in line:
                 raw_lines_or_comma_separated.extend(p.strip() for p in line.split(','))
            else:
                 raw_lines_or_comma_separated.append(line.strip())

        for rpl in raw_lines_or_comma_separated:
            if not rpl: continue
            is_short_capitalized_line = (len(rpl.split()) <= 5 and len(rpl) > 2 and not rpl.lower().startswith(("more ", "all ")))
            # A line is considered a participant if it contains "--" (Name -- Title) or is a short, capitalized line.
            if "--" in rpl or is_short_capitalized_line:
                cleaned_rpl = re.sub(r"^[*\-\d\.]+\s*", "", rpl).strip().rstrip(':')
                if cleaned_rpl:
                    participants_full_strings.append(cleaned_rpl)

        # Deduplicate the final list.
        participants_full_strings = list(set(p for p in participants_full_strings if p and len(p.split()) < 10))
        if participants_full_strings:
            doc_metadata["extracted_participants"] = participants_full_strings
            logging.debug(f"Extracted full participant strings for {processing_target_id}: {participants_full_strings}")
        else:
            logging.debug(f"No participant strings extracted for {processing_target_id} from section.")

    # 4. Remove large boilerplate sections (e.g., legal disclaimers, forward-looking statements).
    temp_text = full_content
    boilerplate_markers = [
        # Each tuple contains a start and end regex to identify a block of text to be removed.
        ("Please note the discussion today will contain forward-looking statements", "except as required by law."),
        ("During today's call, management will also discuss certain non-GAAP financial measures", "news release issued earlier today."),
        ("As a reminder, this conference call is being recorded.", "ir.[\\w\\.-]+com[\\.\\w\\/]*"),
        (r"\[Operator Instructions\]", r"\[Operator Instructions\]"),
        (r"Thank you, operator\.", r"Thank you, operator\."),
        (r"Your first question(?: in queue)? comes from the line of .*?\.\s*Your line is open\.", r"Your first question(?: in queue)? comes from the line of .*?\.\s*Your line is open\."),
        (r"Our next question comes from the line of .*?\.\s*Your line is now open\.", r"Our next question comes from the line of .*?\.\s*Your line is now open\."),
        (r"And that concludes the question-and-answer session\.", r"Have a great day\."),
        (r"Duration: \d+ minutes", r"All earnings call transcripts|Copyright © \d{4}|Thomson Reuters .*? transcript"),
        (r"(?:Call Participants|Participants|Conference Call Participants|Executives|Analysts|Speakers):.*?(?=\n\n|\n[A-ZÀ-ÖØ-Þ][\w\s,&'\.-]+:|Prepared Remarks:|Questions and Answers:|$)", r"(?:Call Participants|Participants|Conference Call Participants|Executives|Analysts|Speakers):.*?(?=\n\n|\n[A-ZÀ-ÖØ-Þ][\w\s,&'\.-]+:|Prepared Remarks:|Questions and Answers:|$)")
    ]

    for start_marker_regex, end_marker_regex in boilerplate_markers:
        try:
            start_re = re.compile(start_marker_regex, re.IGNORECASE | re.DOTALL)
            end_re = re.compile(end_marker_regex, re.IGNORECASE | re.DOTALL)
            
            # If start and end are the same, it's a simple substitution.
            if start_marker_regex == end_marker_regex:
                temp_text = start_re.sub('', temp_text)
                continue

            # For blocks, find all start markers and remove the text until the corresponding end marker.
            new_text_parts = []
            last_end = 0
            for start_match in start_re.finditer(temp_text):
                start_idx = start_match.start()
                new_text_parts.append(temp_text[last_end:start_idx])
                
                end_match = end_re.search(temp_text, pos=start_match.end())
                if end_match:
                    last_end = end_match.end()
                else: # Fallback if an end marker isn't found.
                    if start_marker_regex.startswith("Duration:") or "(?:Call Participants|Participants|Conference Call Participants|Executives|Analysts|Speakers):" in start_marker_regex :
                        logging.debug(f"Boilerplate section '{start_marker_regex}' started but no clear end found. Removing till end of document from match point.")
                        last_end = len(temp_text)
                    else:
                        line_end_m = re.search(r"(\n|$)", temp_text, pos=start_match.end())
                        last_end = line_end_m.start() if line_end_m else start_match.end()
                        logging.debug(f"Boilerplate section '{start_marker_regex}' started but no clear end found. Removing only the start line/paragraph.")
            new_text_parts.append(temp_text[last_end:])
            temp_text = "".join(new_text_parts)
        except Exception as e: logging.debug(f"Boilerplate removal error for '{start_marker_regex}': {e}")
    
    lines = temp_text.splitlines()

    # 5. Main state machine loop to process the transcript line-by-line.
    speaker_turns = []
    current_speaker_full = "Unknown"
    current_section = "Introduction/Presentation"
    speech_accumulator = [] # Accumulates lines of dialogue for the current speaker.
    turn_id_counter = 0

    def flush_speaker_turn():
        """Processes and saves the accumulated speech for the current speaker."""
        nonlocal speech_accumulator, current_speaker_full, current_section, speaker_turns, turn_id_counter
        if speech_accumulator:
            full_speech_block = "\n".join(speech_accumulator).strip()
            # Apply final cleaning and sentence segmentation to the accumulated speech.
            cleaned_block = clean_speech_text(full_speech_block)
            final_sentences = sentence_segment_and_filter(cleaned_block, min_sentence_words=2)
            if final_sentences:
                turn_id_counter += 1
                # Deconstruct the full speaker string into name and title.
                simple_name = current_speaker_full.split(' -- ')[0].strip()
                title = current_speaker_full.split(' -- ')[1].strip() if ' -- ' in current_speaker_full else "N/A"
                
                if title == "N/A" and simple_name.lower() in ["operator", "moderator", "host"]:
                    title = simple_name

                # Append the structured turn to the results list.
                speaker_turns.append({
                    "turn_id": f"{doc_metadata.get('ticker','UNK')}_{doc_metadata.get('year','YYYY')}_{doc_metadata.get('quarter','QX')}_turn{turn_id_counter:03d}",
                    "speaker_full": current_speaker_full,
                    "speaker_simple_name": simple_name,
                    "speaker_title_affiliation": title,
                    "section": current_section,
                    "text": final_sentences
                })
            speech_accumulator = [] # Reset for the next speaker.

    for i, line_text in enumerate(lines):
        line_stripped = line_text.strip()
        if not line_stripped: continue

        # Check for section headers.
        if line_stripped.lower().startswith("prepared remarks") or line_stripped.lower().startswith("presentation"):
            flush_speaker_turn()
            current_section = "Prepared Remarks"
            current_speaker_full = "Unknown"
            continue
        elif line_stripped.lower().startswith("questions and answers") or line_stripped.lower().startswith("q&a"):
            flush_speaker_turn()
            current_section = "Q&A"
            current_speaker_full = "Unknown"
            continue

        # Check if the line introduces a new speaker.
        potential_new_speaker_full = is_potential_speaker_line_with_titles(line_stripped, participants_full_strings)
        
        if potential_new_speaker_full:
            # This logic adds a check to avoid misidentifying a line as a speaker change.
            is_true_speaker_line = True
            if i + 1 < len(lines):
                next_line_stripped = lines[i+1].strip()
                if not next_line_stripped or is_potential_speaker_line_with_titles(next_line_stripped, participants_full_strings) or \
                   next_line_stripped.lower().startswith(("prepared remarks", "questions and answers", "q&a")):
                    pass
                elif len(next_line_stripped.split()) < 3 and next_line_stripped[0].islower():
                    pass # Less likely to be a new speaker if the next line is a short, lowercase fragment.

            if is_true_speaker_line:
                flush_speaker_turn() # A new speaker is found, so process the previous one's speech.
                current_speaker_full = potential_new_speaker_full
                # Handle cases where the speaker line itself contains some speech.
                if re.match(r"^(Okay|Great|Alright|All right|Thank you|Thanks|Yes|Well|So|And now|Next question).{0,20}$", line_stripped, re.IGNORECASE):
                    pass
                elif line_stripped.endswith(":") and line_stripped.lower() == potential_new_speaker_full.lower() + ":":
                    pass # Line is only the speaker name, no speech.
                else:
                    # Extract speech that's on the same line as the speaker's name.
                    speech_after_speaker = line_stripped[len(potential_new_speaker_full):].lstrip(": ")
                    if speech_after_speaker:
                         speech_accumulator.append(speech_after_speaker)
                continue
        
        # If it's not a speaker or section change, it's part of the ongoing dialogue.
        speech_accumulator.append(line_text)

    # After the loop, flush the last speaker's accumulated dialogue.
    flush_speaker_turn()

    if not speaker_turns:
        logging.warning(f"No speaker turns generated for {processing_target_id} after processing.")

    # 6. Assemble the final JSON object and write it to a file.
    output_data = {
        "document_metadata": doc_metadata,
        "speaker_turns": speaker_turns
    }

    try:
        output_path_obj = Path(output_filename)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        if speaker_turns:
             logging.info(f"Successfully processed {processing_target_id} into JSON with {len(speaker_turns)} speaker turns -> {output_filename}")
        else:
             logging.info(f"Processed {processing_target_id}, but no speaker turns extracted. Output JSON saved: {output_filename}")
    except Exception as e:
        logging.error(f"Could not write JSON output file {output_filename}: {e}")