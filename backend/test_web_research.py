import json
import unittest

from web_research import MAX_CARDS, MAX_CITATIONS, MAX_GROUPS, MAX_TEXT, WebResearch, source_url


def envelope(kind, group="lookup_1", **data):
    return {"type": "response.event", "delegation_id": group, "event": {"type": kind, **data}}


def message(text="Ị́ nọ ebe a? [1]", url="https://example.org/igbo", item_id="message_1", **annotation):
    return {"type": "message", "role": "assistant", "id": item_id, "content": [
        {"type": "output_text", "text": text, "annotations": [
            {"type": "url_citation", "url": url, "title": "Igbo source", "start_index": 12, "end_index": 15, **annotation}
        ]}
    ]}


class WebResearchTests(unittest.TestCase):
    def test_real_search_progress_citations_and_empty_lifecycle_snapshot(self):
        research = WebResearch()
        self.assertEqual(research.snapshot(), {"status": "idle", "cards": []})
        research.consume(envelope("response.web_search_call.searching"))
        self.assertEqual(research.snapshot()["status"], "searching")
        research.consume(envelope("response.web_search_call.completed"))
        research.consume(envelope("response.output_item.done", item=message()))
        # GPT-Live forwards response.completed with an empty output array.
        research.consume(envelope("response.completed", response={"output": []}))
        snapshot = research.snapshot()
        self.assertEqual(snapshot["status"], "complete")
        self.assertEqual(snapshot["cards"][0]["text"], "Ị́ nọ ebe a? [1]")
        self.assertEqual(snapshot["cards"][0]["citations"][0], {
            "url": "https://example.org/igbo", "title": "Igbo source", "start": 12, "end": 15})

    def test_replay_and_repeated_item_do_not_duplicate_cards(self):
        research = WebResearch()
        event = {**envelope("response.output_item.done", item=message()), "event_id": "event_1"}
        research.consume(event)
        self.assertFalse(research.consume(event))
        research.consume(envelope("response.output_item.done", item=message()))
        self.assertEqual(len(research.snapshot()["cards"]), 1)

    def test_unrelated_events_private_input_and_tool_arguments_are_not_retained(self):
        research = WebResearch()
        events = [{"type": "session.input_transcript.delta", "delta": "PRIVATE_INPUT"},
                  envelope("response.output_text.delta", delta="PRIVATE_INPUT"),
                  envelope("response.output_item.done", item={"type": "function_call", "arguments": "PRIVATE_INPUT"}),
                  envelope("response.output_item.done", item={"type": "web_search_call", "status": "completed",
                                                             "action": {"query": "PRIVATE_INPUT"}})]
        for event in events: research.consume(event)
        self.assertNotIn("PRIVATE_INPUT", json.dumps(research.snapshot()))
        self.assertEqual(research.snapshot()["cards"], [])

    def test_unsearched_conversation_not_presented_as_research(self):
        research = WebResearch()
        item = message()
        item["content"][0]["annotations"] = []
        research.consume(envelope("response.output_item.done", item=item))
        self.assertEqual(research.snapshot()["cards"], [])
        research.consume(envelope("response.web_search_call.searching"))
        research.consume(envelope("response.output_item.done", item=item))
        self.assertTrue(research.snapshot()["cards"][0]["uncited"])

    def test_source_urls_and_invalid_offsets_are_safe(self):
        for value in ("javascript:alert(1)", "data:text/html,hello", "file:///etc/passwd", "https://user:secret@example.org",
                      "https://example.org:bad", "https://example.org/\nhello", "https://[", None):
            self.assertIsNone(source_url(value))
        research = WebResearch()
        research.consume(envelope("response.output_item.done", item=message(start_index=-1, end_index=999)))
        citation = research.snapshot()["cards"][0]["citations"][0]
        self.assertEqual(citation["start"], len("Ị́ nọ ebe a? [1]"))
        self.assertEqual(citation["start"], citation["end"])

    def test_bounds_and_snapshot_cannot_mutate_internal_state(self):
        research = WebResearch()
        text = "ị" * (MAX_TEXT + 100)
        item = message(text=text)
        item["content"][0]["annotations"] *= 200
        for i in range(20):
            item["id"] = f"message_{i}"
            research.consume(envelope("response.output_item.done", group=f"lookup_{i}", item=item))
        snapshot = research.snapshot()
        self.assertLessEqual(len(research.groups), MAX_GROUPS)
        self.assertEqual(len(snapshot["cards"]), MAX_CARDS)
        self.assertTrue(all(len(card["text"]) == MAX_TEXT and card["truncated"] for card in snapshot["cards"]))
        self.assertTrue(all(len(card["citations"]) <= MAX_CITATIONS for card in snapshot["cards"]))
        snapshot["cards"][0]["citations"][0]["url"] = "changed"
        self.assertNotEqual(research.snapshot()["cards"][0]["citations"][0]["url"], "changed")
        for i in range(500): research.consume({**envelope("response.completed"), "event_id": f"event_{i}"})
        self.assertLessEqual(len(research.seen), 256)

    def test_malformed_events_and_unconfirmed_search_end(self):
        research = WebResearch()
        for value in (None, [], {}, {"type": "response.event", "event": []}, envelope("response.output_item.done", item=[])):
            self.assertFalse(research.consume(value))
        research.consume(envelope("response.web_search_call.searching"))
        research.finish()
        self.assertEqual(research.snapshot()["status"], "error")
        self.assertEqual(research.snapshot()["cards"], [])


if __name__ == "__main__": unittest.main()
