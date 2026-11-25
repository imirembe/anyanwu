"""No-key, no-paid-network checks for the isolated voice trial."""
import http.client
import io
import json
import threading
import time
import unittest
import urllib.error
from unittest.mock import patch

from openai_voice_test import Call, LocalServer, OpenAITransport, Trial, TrialError, make_handler, session_payload


REQUEST_ID = "a" * 32


class Socket:
    def __init__(self):
        self.sent = []
        self.events = [json.dumps({"type": "session.started"}),
                       json.dumps({"type": "session.usage.updated", "usage": {"seconds": 12}})]
        self.wake = threading.Event()

    def __enter__(self): return self
    def __exit__(self, *args): return False

    def send(self, value):
        self.sent.append(json.loads(value))
        if self.sent[-1]["type"] == "session.close":
            self.events.append(json.dumps({"type": "session.closed", "usage": {"seconds": 15}, "reason": "close_requested"}))

    def recv(self, timeout):
        if self.events:
            return self.events.pop(0)
        self.wake.wait(0.005)
        raise TimeoutError()


class Transport:
    def __init__(self):
        self.socket = Socket()
        self.creations = 0
        self.session_id = None

    def create(self, sdp, voice="marin", direction=""):
        self.creations += 1
        self.voice, self.direction = voice, direction
        return {"session_id": "live_opaque-prefix", "sdp": "v=0\r\nanswer"}

    def attach(self, session_id):
        self.session_id = session_id
        return self.socket


def eventually(predicate):
    until = time.monotonic() + 2
    while time.monotonic() < until:
        if predicate(): return
        time.sleep(0.005)
    raise AssertionError("Condition did not become true")


class TrialTests(unittest.TestCase):
    def test_policy_and_credential_separation(self):
        payload = session_payload("v=0")
        config = payload["session"]
        self.assertFalse(config["store"])
        self.assertIn("Igbo, English and mixed", config["instructions"])
        self.assertIn("Always speak and write your replies", config["instructions"])
        self.assertEqual(config["delegation"]["responses"]["tools"], [{"type": "web_search"}])
        self.assertEqual(config["delegation"]["responses"]["tool_choice"], "auto")
        self.assertIn("Igbo", config["delegation"]["responses"]["instructions"])
        self.assertIn("no dedicated live rail", config["delegation"]["responses"]["instructions"])
        self.assertNotIn("format", config["audio"])
        self.assertNotIn("key", json.dumps(payload))

    def test_one_call_only_and_close_final_usage(self):
        transport = Transport()
        trial = Trial(transport)
        result = trial.create(REQUEST_ID, "v=0")
        self.assertEqual(set(result), {"sdp", "test_seconds", "comparison_instruction"})
        with self.assertRaises(TrialError) as error:
            trial.create("b" * 32, "v=0")
        self.assertEqual(error.exception.status, 409)
        eventually(lambda: trial.call.seconds == 12)
        trial.command(REQUEST_ID, "stop")
        eventually(lambda: trial.call.confirmed)
        self.assertEqual(trial.command(REQUEST_ID, "status")["seconds"], 15)  # cumulative, not 27
        self.assertEqual(transport.session_id, "live_opaque-prefix")
        self.assertEqual(transport.creations, 1)
        self.assertEqual([e["type"] for e in transport.socket.sent], ["session.close"])
        trial.shutdown()

    def test_stop_while_create_pending_is_delivered_later(self):
        entered, release = threading.Event(), threading.Event()
        transport = Transport()
        original = transport.create
        def create(*args):
            entered.set(); release.wait(2); return original(*args)
        transport.create = create
        trial = Trial(transport)
        thread = threading.Thread(target=lambda: trial.create(REQUEST_ID, "v=0"))
        thread.start(); self.assertTrue(entered.wait(1))
        trial.command(REQUEST_ID, "stop")
        release.set(); thread.join(2)
        eventually(lambda: trial.call.confirmed)
        self.assertEqual(trial.call.state, "closed")

    def test_abandoned_browser_closed_by_server_without_final_browser_event(self):
        now = [100.0]
        trial = Trial(Transport(), clock=lambda: now[0])
        trial.create(REQUEST_ID, "v=0")
        eventually(lambda: trial.call.state == "active")
        now[0] += 30
        eventually(lambda: trial.call.confirmed)
        self.assertEqual(trial.call.reason, "close_requested")

    def test_server_deadline_even_with_browser_heartbeat(self):
        now = [100.0]
        trial = Trial(Transport(), clock=lambda: now[0])
        trial.create(REQUEST_ID, "v=0")
        eventually(lambda: trial.call.state == "active")
        now[0] += 121
        trial.command(REQUEST_ID, "status")
        eventually(lambda: trial.call.confirmed)

    def test_no_started_replay_needed_to_close_existing_sideband(self):
        transport = Transport()
        transport.socket.events.clear()
        trial = Trial(transport)
        trial.create(REQUEST_ID, "v=0")
        trial.command(REQUEST_ID, "stop")
        eventually(lambda: trial.call.confirmed)

    def test_unconfirmed_control_failure_blocks_new_charges(self):
        transport = Transport()
        def attach(_): raise RuntimeError("must never appear in a response or log: SECRET")
        transport.attach = attach
        trial = Trial(transport)
        trial.create(REQUEST_ID, "v=0")
        eventually(lambda: trial.call.state == "unconfirmed")
        self.assertNotIn("SECRET", json.dumps(trial.command(REQUEST_ID, "status")))
        with self.assertRaises(TrialError): trial.create("b" * 32, "v=0")
        self.assertEqual(transport.creations, 1)

    def test_final_close_timeout_is_unconfirmed(self):
        transport = Transport()
        transport.socket.send = lambda value: None
        now = [100.0]
        trial = Trial(transport, clock=lambda: now[0])
        trial.create(REQUEST_ID, "v=0")
        eventually(lambda: trial.call.state == "active")
        trial.command(REQUEST_ID, "stop")
        eventually(lambda: trial.call.state == "closing")
        time.sleep(0.02)
        now[0] += 16
        eventually(lambda: trial.call.state == "unconfirmed")
        self.assertFalse(trial.call.confirmed)

    def test_no_key_means_no_call_and_no_attempt(self):
        trial = Trial(None)
        with self.assertRaises(TrialError): trial.create(REQUEST_ID, "v=0")
        self.assertEqual(trial.attempts, 0)

    def test_browser_close_waits_for_last_sideband_citations(self):
        from test_web_research import message, envelope
        transport = Transport()
        trial = Trial(transport)
        trial.create(REQUEST_ID, "v=0")
        eventually(lambda: trial.call.state == "active")
        def delayed_events():
            time.sleep(0.03)
            transport.socket.events.extend([json.dumps(envelope("response.output_item.done", item=message())),
                                           json.dumps({"type": "session.closed", "usage": {"seconds": 15}})])
        thread = threading.Thread(target=delayed_events)
        thread.start()
        result = trial.command(REQUEST_ID, "closed", {"type": "session.closed", "usage": {"seconds": 15}})
        thread.join(1)
        self.assertTrue(result["confirmed"])
        self.assertEqual(result["research"]["cards"][0]["citations"][0]["url"], "https://example.org/igbo")

    def test_research_is_per_call_and_normalized_in_status(self):
        from test_web_research import message, envelope
        trial = Trial(Transport())
        trial.create(REQUEST_ID, "v=0")
        trial.call.research.consume(envelope("response.output_item.done", item=message()))
        result = trial.command(REQUEST_ID, "status")
        self.assertEqual(result["research"]["cards"][0]["text"], "Ị́ nọ ebe a? [1]")
        self.assertNotIn("PRIVATE_KEY_FIXTURE", json.dumps(result))
        trial.command(REQUEST_ID, "stop")
        eventually(lambda: trial.call.confirmed)
        trial.transport = Transport()
        trial.create("b" * 32, "v=0")
        self.assertEqual(trial.command("b" * 32, "status")["research"]["cards"], [])
        trial.command("b" * 32, "stop")
        eventually(lambda: trial.call.confirmed)

    def test_ambiguous_create_is_not_retried(self):
        transport = Transport()
        def create(*_): raise TrialError(502, "Unconfirmed", True)
        transport.create = create
        trial = Trial(transport)
        with self.assertRaises(TrialError): trial.create(REQUEST_ID, "v=0")
        self.assertEqual(trial.call.state, "unconfirmed")
        with self.assertRaises(TrialError): trial.create("b" * 32, "v=0")
        self.assertEqual(trial.attempts, 1)

    def test_upstream_key_only_in_authorization_and_no_redirect(self):
        transport = OpenAITransport("PRIVATE_KEY_FIXTURE")
        response = io.BytesIO(json.dumps({"session": {"id": "live_test"}, "transport": {"type": "webrtc", "sdp": "v=0"}}).encode())
        with patch("urllib.request.build_opener") as build:
            build.return_value.open.return_value.__enter__.return_value = response
            result = transport.create("v=0")
            request = build.return_value.open.call_args.args[0]
            self.assertEqual(request.headers["Authorization"], "Bearer PRIVATE_KEY_FIXTURE")
            self.assertNotIn("PRIVATE_KEY_FIXTURE", request.data.decode())
            self.assertNotIn("PRIVATE_KEY_FIXTURE", json.dumps(result))
            self.assertIsNone(build.call_args.args[0].redirect_request(None, None, 302, "", {}, "https://example.com"))

    def test_provider_exception_does_not_expose_body_or_key(self):
        transport = OpenAITransport("PRIVATE_KEY_FIXTURE")
        with patch("urllib.request.build_opener") as build:
            build.return_value.open.side_effect = urllib.error.HTTPError("https://api.openai.com", 401, "secret", {}, io.BytesIO(b"private"))
            with self.assertRaises(TrialError) as error: transport.create("v=0")
        self.assertEqual(error.exception.status, 401)
        self.assertFalse(error.exception.ambiguous)
        self.assertNotIn("private", str(error.exception).lower())


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.trial = Trial(None)
        self.server = LocalServer(("127.0.0.1", 0), make_handler(self.trial, "http://127.0.0.1:0", "test_nonce"))
        self.port = self.server.server_port
        self.origin = f"http://127.0.0.1:{self.port}"
        self.server.RequestHandlerClass = make_handler(self.trial, self.origin, "test_nonce")
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(1)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        connection.request(method, path, body, headers or {})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def test_only_allowlisted_assets_and_config_available(self):
        status, headers, body = self.request("GET", "/api/config")
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["configured"])
        self.assertTrue(json.loads(body)["web_search"])
        self.assertEqual(json.loads(body)["default_voice"], "owerri-marin")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        for path in ("/../server.py", "/.env", "/backend/.env", "/fonts/../../AGENTS.md", "/api/config?extra=1"):
            self.assertEqual(self.request("GET", path)[0], 404)
        self.assertEqual(self.request("GET", "/")[0], 200)
        self.assertEqual(self.request("GET", "/controller.mjs")[0], 200)
        self.assertEqual(self.request("GET", "/research.mjs")[0], 200)

    def test_dns_rebinding_and_cross_origin_protection(self):
        self.assertEqual(self.request("GET", "/api/config", headers={"Host": "attacker.example"})[0], 403)
        self.assertEqual(self.request("GET", "/api/config", headers={"Sec-Fetch-Site": "cross-site"})[0], 403)
        body = json.dumps({"request_id": REQUEST_ID, "sdp": "v=0"})
        for headers in ({}, {"Origin": "http://attacker.example", "X-Trial-Token": "test_nonce"}, {"Origin": self.origin}):
            self.assertEqual(self.request("POST", "/api/session", body, headers)[0], 403)

    def test_request_validation_happens_before_provider_call(self):
        headers = {"Origin": self.origin, "X-Trial-Token": "test_nonce", "Content-Type": "application/json"}
        for payload in ([], {"request_id": "bad", "sdp": "v=0"}, {"request_id": REQUEST_ID, "sdp": "no"}):
            self.assertEqual(self.request("POST", "/api/session", json.dumps(payload), headers)[0], 400)
        self.assertEqual(self.request("POST", "/api/session", "x" * 70000, headers)[0], 413)
        self.assertEqual(self.request("POST", "/api/session", json.dumps({"request_id": REQUEST_ID, "sdp": "v=0"}), headers)[0], 503)
        self.assertEqual(self.trial.attempts, 0)
        self.assertEqual(self.request("POST", "/api/voice-design", json.dumps({"create": True, "prompt": "untrusted"}), headers)[0], 400)
        self.assertEqual(self.request("POST", "/api/voice-design", json.dumps({"create": True}), headers)[0], 503)


if __name__ == "__main__": unittest.main()
