"""Conservative raw reference extraction for offline request experiments.

Rules are developed from training text. References are tentative text spans,
never resolved contacts, clock values, locations or instructions to act.
"""
import hashlib
import re
import unicodedata

from .resources import LabError

FIELDS = ("contact_reference", "time_reference", "date_reference", "location_reference")
SCOPE = {
    "alarm": ("time_reference", "date_reference"),
    "calendar_update": ("time_reference", "date_reference"),
    "make_call": ("contact_reference",),
    "text": ("contact_reference",),
    "time": ("time_reference", "date_reference", "location_reference"),
    "weather": ("time_reference", "date_reference", "location_reference"),
}
BOUNDARY = r"[\w\u0300-\u036f]"
IG_NUMBER = r"(?:iri(?:\s+na\s+(?:otu|abụọ|abuo))?|otu|abụọ|abuo|atọ|ato|anọ|ano|ise|isii|asaa|asatọ|asato|itoolu)"
DAYPART = r"(?:ụtụtụ|ututu|ehihie|mgbede|abalị|abali)"
IG_CLOCK = re.compile(r"(?<!"+BOUNDARY+r")(?P<value>(?:ọkara\s+gafere\s+)?elekere\s+"+IG_NUMBER+r"(?:\s+(?:nke\s+)?"+DAYPART+r")?)(?!"+BOUNDARY+r")", re.I)
NUMERIC_CLOCK = re.compile(r"(?<![\w:])(?P<value>(?:(?:0?[1-9]|1[0-2])(?::[0-5][0-9])?\s*(?:am|pm)|(?:[01]?[0-9]|2[0-3]):[0-5][0-9]))(?![\w:])", re.I)
EN_CLOCK = re.compile(r"\b(?:at|for)\s+(?P<value>(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)(?:\s+(?:thirty|fifteen|forty[- ]five))?(?:\s+in\s+the\s+(?:morning|afternoon|evening))?)\b", re.I)
NAMED_CLOCK = re.compile(r"\b(?P<value>noon|midnight|mid\s+day)\b", re.I)
DATE = re.compile(r"(?<!"+BOUNDARY+r")(?P<value>(?:(?:next|this)\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|today|tomorrow|tonight|taa|ta|echi|Tọsdee|Tọzdee|Tuzdee|Tuuzdee|Wenezdee)(?!"+BOUNDARY+r")", re.I)
EN_CONTACT = re.compile(r"\b(?:call(?:\s+to)?|phone(?!\s+call\b)|text(?:\s+to)?)\s+(?P<value>[^\n.,!?]+)", re.I)
IG_CALL = re.compile(r"(?<!"+BOUNDARY+r")(?:ịkpọ|ịkpo|kpọọ?|kpoo?)\s+(?P<value>[^\n.,!?]+)", re.I)
IG_TEXT = re.compile(r"(?<!"+BOUNDARY+r")(?:zịgara|zigara|deegara|degara|deere|zịpụrụ)\s+(?:m\s+)?(?P<value>.+?)\s+ozi(?!"+BOUNDARY+r")", re.I)
CONTACT_STOP = re.compile(r"\s+(?:and|to\s+(?:tell|ask)|tell|ask|ka|ugbua|ugbu\s+a|n['’]ekwent[ịi]|ozi|n['’]elekere|at)\b", re.I)
EN_LOCATION = re.compile(r"\b(?:in|for)\s+(?P<value>[^\n!?]+)", re.I)
IG_LOCATION = re.compile(r"(?=(?<!"+BOUNDARY+r")(?:na\s+|n['’](?:obodo\s+)?)(?P<value>[^\n!?]+))", re.I)
LOCATION_STOP = re.compile(r"\s+(?:right\s+now|now|today|tomorrow|tonight|n['’]ubochi|n['’]ụbọchị|ugbu\s+a|ugbua|echi|taa|ta)\b", re.I)


def transcript(text):
    if not isinstance(text, str) or len(text)>2048 or "\x00" in text:
        raise LabError("Use a transcript of at most 2,048 characters without NULs.")
    return unicodedata.normalize("NFC", text)


def span(text, field, start, end):
    while start<end and (text[start].isspace() or text[start] in ",.;!?"): start+=1
    while end>start and (text[end-1].isspace() or text[end-1] in ",.;!?"): end-=1
    if start==end: return None
    return {"field":field, "value":text[start:end],
            "start_byte":len(text[:start].encode()), "limit_byte":len(text[:end].encode())}


def extract_details(text, label):
    text=transcript(text)
    allowed=SCOPE.get(label, ())
    found=[]
    def add(field, start, end):
        item=span(text,field,start,end)
        if item and item not in found: found.append(item)
    if "contact_reference" in allowed:
        patterns=(IG_TEXT,EN_CONTACT) if label=="text" else (IG_CALL,EN_CONTACT)
        for pattern in patterns:
            match=pattern.search(text)
            if not match: continue
            start,end=match.span("value")
            stop=CONTACT_STOP.search(text,start,end)
            if stop: end=stop.start()
            add("contact_reference",start,end)
            break
    if "time_reference" in allowed:
        for pattern in (IG_CLOCK,NUMERIC_CLOCK,EN_CLOCK,NAMED_CLOCK):
            for match in pattern.finditer(text):
                start,end=match.span("value")
                a,b=len(text[:start].encode()),len(text[:end].encode())
                if not any(s["field"]=="time_reference" and a<s["limit_byte"] and s["start_byte"]<b for s in found):
                    add("time_reference",start,end)
    if "date_reference" in allowed:
        for match in DATE.finditer(text): add("date_reference",*match.span("value"))
    if "location_reference" in allowed:
        # Last Igbo preposition handles a date/daypart followed by a place.
        # Do not interpret dates as place names or resolve any place reference.
        candidates=list(IG_LOCATION.finditer(text))+list(EN_LOCATION.finditer(text))
        for match in sorted(candidates,key=lambda m:m.start(),reverse=True):
            start,end=match.span("value")
            stop=LOCATION_STOP.search(text,start,end)
            if stop: end=stop.start()
            value=text[start:end].strip(" .,;!?")
            if not value or DATE.search(value) or re.match(r"^(?:"+DAYPART+r"|ụbọchị|ubochi|ya|ọ|m|a)(?:\s|$)",value,re.I): continue
            add("location_reference",start,end)
            break
    return sorted(found,key=lambda s:(s["start_byte"],s["limit_byte"],s["field"]))


def offset_reference(text):
    return {"unit":"UTF-8 bytes", "normalization":"NFC",
            "transcript_sha256":hashlib.sha256(transcript(text).encode()).hexdigest()}


def detail_metrics(rows, predictions, gold):
    """Exact-span scores for the explicitly reviewed four-field scope only."""
    totals={f:{"tp":0,"fp":0,"fn":0} for f in FIELDS}
    exact=with_gold=exact_with_gold=0
    for row, predicted in zip(rows,predictions):
        expected=gold[row["id"]]["spans"]
        def key(s):return s["field"],s["start_byte"],s["limit_byte"],s["value"]
        actual,wanted={key(s) for s in predicted},{key(s) for s in expected}
        exact+=actual==wanted
        if wanted: with_gold+=1;exact_with_gold+=actual==wanted
        for f in FIELDS:
            a={s for s in actual if s[0]==f};b={s for s in wanted if s[0]==f}
            totals[f]["tp"]+=len(a&b);totals[f]["fp"]+=len(a-b);totals[f]["fn"]+=len(b-a)
    def rates(counts):
        tp,fp,fn=(counts[k] for k in ("tp","fp","fn"))
        return {**counts,"precision":tp/(tp+fp) if tp+fp else None,
                "recall":tp/(tp+fn) if tp+fn else None,"f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
    return {"examples":len(rows),"exact_records":exact,"records_with_gold_spans":with_gold,
            "exact_records_with_gold_spans":exact_with_gold,
            "micro":rates({k:sum(v[k] for v in totals.values()) for k in ("tp","fp","fn")}),
            "per_field":{f:rates(v) for f,v in totals.items()},
            "reference":"Codex AI-reviewed raw spans, four-field scope only; not native-speaker or resolved-entity accuracy."}
