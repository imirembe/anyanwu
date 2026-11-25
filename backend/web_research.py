"""Bounded, in-memory web results from actual GPT-Live Responses events.

Only the server sideband collects research. Never crawl URLs or retain audio,
input transcripts, raw events, search queries, credentials or model instructions.
"""
from __future__ import annotations

from collections import OrderedDict
import urllib.parse

MAX_GROUPS = 8
MAX_CARDS = 5
MAX_TEXT = 6000
MAX_CITATIONS = 12


def source_url(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 2048 or any(c.isspace() or ord(c) < 32 for c in value):
        return None
    try:
        parsed = urllib.parse.urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            return None
        # Access .port to reject invalid ports; URLs are links, never server fetches.
        parsed.port
    except ValueError:
        return None
    return value


class WebResearch:
    def __init__(self):
        self.groups = OrderedDict()
        self.cards = OrderedDict()
        self.seen = OrderedDict()

    def consume(self, envelope: object) -> bool:
        if not isinstance(envelope, dict) or envelope.get("type") != "response.event":
            return False
        event = envelope.get("event")
        if not isinstance(event, dict):
            return False
        kind = event.get("type")
        if not isinstance(kind, str):
            return False
        event_id = envelope.get("event_id")
        if isinstance(event_id, str) and len(event_id) <= 256:
            if event_id in self.seen:
                return False
            self.seen[event_id] = None
            if len(self.seen) > 256:
                self.seen.popitem(last=False)
        group_id = envelope.get("delegation_id")
        if not isinstance(group_id, str) or not 0 < len(group_id) <= 256:
            return False
        relevant = kind.startswith("response.web_search_call.") or kind in {
            "response.output_item.done", "response.completed", "response.failed", "response.incomplete",
        }
        if not relevant:
            return False
        if group_id not in self.groups:
            self.groups[group_id] = {"searched": False, "status": "idle"}
            if len(self.groups) > MAX_GROUPS:
                expired, _ = self.groups.popitem(last=False)
                self.cards = OrderedDict((k, v) for k, v in self.cards.items() if v["group"] != expired)
        group = self.groups[group_id]
        if kind in {"response.web_search_call.in_progress", "response.web_search_call.searching"}:
            group.update(searched=True, status="searching")
        elif kind == "response.web_search_call.completed":
            group.update(searched=True, status="answering")
        elif kind == "response.output_item.done":
            item = event.get("item")
            if not isinstance(item, dict):
                return False
            if item.get("type") == "web_search_call":
                group.update(searched=True, status="answering" if item.get("status") == "completed" else "error")
            elif item.get("type") == "message" and item.get("role") == "assistant":
                item_id = item.get("id")
                content = item.get("content")
                if not isinstance(item_id, str) or len(item_id) > 256 or not isinstance(content, list):
                    return False
                for index, part in enumerate(content[:8]):
                    card = self._card(group_id, item_id, index, part, group["searched"])
                    if card:
                        self.cards[card["id"]] = card
                        while len(self.cards) > MAX_CARDS:
                            self.cards.popitem(last=False)
        elif group["searched"]:
            group["status"] = "complete" if kind == "response.completed" else "error"
        return True

    @staticmethod
    def _card(group: str, item_id: str, index: int, part: object, searched: bool) -> dict | None:
        if not isinstance(part, dict) or part.get("type") != "output_text" or not isinstance(part.get("text"), str):
            return None
        original = part["text"]
        text = original[:MAX_TEXT]
        if not text:
            return None
        annotations = part.get("annotations", [])
        citations = []
        seen = set()
        if isinstance(annotations, list):
            for annotation in annotations[:128]:
                if not isinstance(annotation, dict) or annotation.get("type") != "url_citation":
                    continue
                url = source_url(annotation.get("url"))
                if not url:
                    continue
                start, end = annotation.get("start_index"), annotation.get("end_index")
                if (not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int)
                        or isinstance(end, bool) or not 0 <= start <= end <= len(original) or start >= len(text)):
                    # Without a usable offset, keep the source as an inline endnote.
                    start = end = len(text)
                else:
                    end = min(end, len(text))
                key = (url, start, end)
                if key in seen:
                    continue
                seen.add(key)
                title = annotation.get("title")
                title = title[:200] if isinstance(title, str) and title.strip() else urllib.parse.urlsplit(url).hostname
                citations.append({"url": url, "title": title, "start": start, "end": end})
                if len(citations) == MAX_CITATIONS:
                    break
        if not searched and not citations:
            return None  # Ordinary delegated conversation is not a web result.
        return {"id": f"{group}:{item_id}:{index}", "group": group, "text": text,
                "citations": citations, "truncated": len(original) > len(text),
                "uncited": not citations}

    def snapshot(self) -> dict:
        latest = next(reversed(self.groups.values()), None) if self.groups else None
        status = latest["status"] if latest else "idle"
        return {"status": status, "cards": [dict(card, citations=[dict(c) for c in card["citations"]])
                                             for card in self.cards.values()]}

    def finish(self) -> None:
        for group in self.groups.values():
            if group["status"] in {"searching", "answering"}:
                group["status"] = "error"
