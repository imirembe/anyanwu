"""Bound uploads and concurrent work before an ASGI application reads a body."""
from __future__ import annotations

import asyncio
import json
import secrets


class HTTPBoundary:
    def __init__(self, app, *, token: str, max_body: int, concurrency: int):
        self.app = app
        self.token = token.encode()
        self.max_body = max_body
        self.concurrency = concurrency
        self.active = 0

    @staticmethod
    async def reject(send, status: int, message: str):
        body = json.dumps({"error": message}).encode()
        headers = [(b"content-type", b"application/json"), (b"cache-control", b"no-store")]
        if status == 503:
            headers.append((b"retry-after", b"2"))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        if self.token and not secrets.compare_digest(headers.get(b"authorization", b""), b"Bearer " + self.token):
            await self.reject(send, 401, "A valid access token is required.")
            return
        # Health probes remain responsive while a model is working.
        if scope["method"] in {"GET", "HEAD"}:
            await self.app(scope, receive, send)
            return
        if self.active >= self.concurrency:
            await self.reject(send, 503, "The service is busy. Please retry shortly.")
            return
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            await self.reject(send, 400, "Invalid request size.")
            return
        if declared < 0 or declared > self.max_body:
            await self.reject(send, 413, "The request is too large.")
            return
        # One event loop per process; no await between checking and admission.
        self.active += 1
        try:
            async def collect():
                body = bytearray()
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return None
                    body.extend(message.get("body", b""))
                    if len(body) > self.max_body:
                        raise OverflowError
                    if not message.get("more_body", False):
                        return bytes(body)

            try:
                body = await asyncio.wait_for(collect(), timeout=15)
            except OverflowError:
                await self.reject(send, 413, "The request is too large.")
                return
            except TimeoutError:
                await self.reject(send, 408, "The upload timed out.")
                return
            if body is None:
                return
            if declared and declared != len(body):
                await self.reject(send, 400, "The request size does not match its body.")
                return
            delivered = False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            async def no_store(message):
                if message["type"] == "http.response.start":
                    message["headers"] = list(message.get("headers", [])) + [(b"cache-control", b"no-store")]
                await send(message)

            await self.app(scope, replay, no_store)
        finally:
            self.active -= 1


async def model_call(function, *args):
    """Keep admission occupied until a running native model actually finishes.

    Cancelling an asyncio waiter cannot stop a CUDA/thread operation. Releasing
    its slot early could start another operation on the same model instance.
    """
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            cancelled = True
            if task.done():
                # Retrieve any native failure without abandoning the operation.
                task.result()
                break
    if cancelled:
        raise asyncio.CancelledError
    return result
