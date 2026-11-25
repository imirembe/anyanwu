"""Pure rules for the first bounded caption/audio alignment experiment."""
import math
import unicodedata


def alignment_text(text):
    """ASR alignment view only; keep original training-label text separately."""
    text=unicodedata.normalize("NFC",text).casefold().replace("’","'").replace("‘","'")
    output=[]
    for i,char in enumerate(text):
        category=unicodedata.category(char)
        if category[0] in ("L","N","M") or char.isspace():
            output.append(char)
        elif char in ("'","-") and 0<i<len(text)-1 and text[i-1].isalpha() and text[i+1].isalpha():
            output.append(char)
        else:
            output.append(" ")
    return " ".join("".join(output).split())


def eligible_caption(entry):
    return (entry.get("display_bounded_candidate") is True
            and entry.get("image_transcription_confidence")=="high"
            and not entry.get("uncertainties"))


def caption_token_ranges(ids, texts, decode):
    """Find exact decoded text boundaries in a jointly encoded caption sequence."""
    normalise=lambda value:' '.join(value.split())
    if normalise(decode(ids))!=' '.join(texts):
        raise ValueError('Joint tokenizer round trip changed the published labels')
    ranges=[]
    start=0
    for index in range(len(texts)):
        expected=' '.join(texts[:index+1])
        ends=[end for end in range(start+1,len(ids)+1) if normalise(decode(ids[:end]))==expected]
        if not ends:
            raise ValueError('No exact token boundary between source captions')
        end=len(ids) if index==len(texts)-1 else min(ends)
        ranges.append((start,end))
        start=end
    return ranges


def evaluate_path(spans,expected_ids,frame_count,frame_seconds,window_start):
    """Validate the forced path and flag uncertain timing, never rejudge subtitles."""
    if not spans or [s[0] for s in spans]!=expected_ids:
        raise ValueError("Forced path did not cover every target token in order.")
    previous=0
    for _,start,end,score in spans:
        if not previous<=start<end<=frame_count or not math.isfinite(score) or not 0<=score<=1:
            raise ValueError("Forced path has invalid bounds or scores.")
        previous=end
    probabilities=[s[3] for s in spans]
    mean=sum(probabilities)/len(probabilities)
    warnings=[]
    if mean<0.10:
        warnings.append("low_acoustic_alignment_score")
    if spans[0][1]<=1 or spans[-1][2]>=frame_count-1:
        warnings.append("alignment_touches_search_window_edge")
    start=window_start+spans[0][1]*frame_seconds
    end=window_start+spans[-1][2]*frame_seconds
    if not 0.20<=end-start<=12:
        warnings.append("implausible_aligned_duration")
    return {"relative_audio_start_seconds":round(start,6),"relative_audio_end_seconds":round(end,6),
            "mean_token_probability":round(mean,6),"minimum_token_probability":round(min(probabilities),6),
            "token_count":len(spans),"timing_warnings":warnings,
            "interpretation":"Machine timing confidence only; not a verdict on the user-reviewed subtitle meaning or speech match."}
