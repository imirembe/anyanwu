import test from 'node:test';
import assert from 'node:assert/strict';
import {Captions, VoiceTrial} from './controller.mjs';

const tick = () => new Promise(resolve => setImmediate(resolve));
const delta = (text, extra = {}) => ({type: 'session.input_transcript.delta', delta: text, start_ms: 1, end_ms: 2, ...extra});

test('real deltas preserve Igbo accents, spacing and repeated words; event IDs deduplicate', () => {
  const captions = new Captions();
  assert.equal(captions.append({type: 'listening', delta: 'fake'}), false);
  captions.append(delta('Ị̀ nọ ', {event_id: '1'}));
  captions.append(delta('nọ ebe a?', {event_id: '2', start_ms: 3, end_ms: 4}));
  captions.append(delta('nọ ebe a?', {event_id: '2'}));
  assert.equal(captions.rows[0].text, 'Ị̀ nọ nọ ebe a?');
  assert.equal(captions.rows[0].start, 1);
  assert.equal(captions.rows[0].end, 4);
  captions.append({type: 'session.output_transcript.delta', delta: '<script>alert(1)</script>', start_ms: 5, end_ms: 6});
  assert.equal(captions.rows[1].role, 'assistant');
  assert.equal(captions.rows[1].text, '<script>alert(1)</script>'); // Rendered with textContent, never HTML.
});

test('caption history and event IDs stay bounded without storage', () => {
  const captions = new Captions();
  for (let i = 0; i < 1000; i++) captions.append(delta('ị'.repeat(1000), {event_id: `${i}`, start_ms: i * 3000, end_ms: i * 3000 + 1}));
  assert.ok(captions.rows.length <= 30);
  assert.ok(captions.rows.reduce((n, r) => n + r.text.length, 0) <= 24000);
  assert.ok(captions.seen.size <= 512);
  captions.clear(); assert.deepEqual(captions.rows, []); assert.equal(captions.seen.size, 0);
});

function fixture({media, api} = {}) {
  const tracks = [{stopped: 0, stop() { this.stopped++; }}];
  const stream = {getTracks: () => tracks};
  const channel = {readyState: 'open', sent: [], close() { this.closed = true; }, send(s) { this.sent.push(JSON.parse(s)); }};
  const peer = {iceGatheringState: 'complete', localDescription: {sdp: 'v=0'},
    addTrack() {}, createDataChannel(name) { assert.equal(name, 'oai-events'); return channel; },
    async createOffer() { return {type: 'offer', sdp: 'v=0'}; }, async setLocalDescription() {}, async setRemoteDescription() {},
    close() { this.closed = true; }};
  const output = {srcObject: null, pause() {}, async play() {}};
  const requests = [], states = [], events = [], problems = [];
  const engine = new VoiceTrial({output, media: media || (async () => stream), peer: () => peer, id: () => 'a'.repeat(32),
    api: async (path, body) => { requests.push({path, body}); return api ? api(path, body) : {sdp: 'v=0', test_seconds: 120}; },
    state: s => states.push(s), event: e => events.push(e), problem: m => problems.push(m)});
  return {engine, tracks, stream, channel, peer, requests, states, events, problems};
}

test('cancel while permission is pending releases a late microphone without creating a call', async () => {
  let resolve;
  const f = fixture({media: () => new Promise(r => {resolve = r;})});
  const pending = f.engine.start();
  f.engine.stop();
  resolve(f.stream); await pending;
  assert.equal(f.tracks[0].stopped, 1);
  assert.equal(f.requests.length, 0);
  assert.equal(f.engine.current, null);
});

test('stop during paid session creation keeps processing the answer and confirms close', async () => {
  let answer;
  const f = fixture({api: (path) => path === '/api/session' ? new Promise(r => {answer = r;}) : {state: 'closing'}});
  const pending = f.engine.start(); await tick();
  assert.ok(answer);
  f.engine.stop();
  assert.ok(f.tracks[0].stopped);
  answer({sdp: 'v=0', test_seconds: 120}); await pending;
  f.channel.onmessage({data: JSON.stringify({type: 'session.started'})});
  assert.equal(f.channel.sent.at(-1).type, 'session.close');
  assert.notEqual(f.peer.closed, true); // Receiver stays open until session.closed.
  f.channel.onmessage({data: JSON.stringify({type: 'session.closed', usage: {seconds: 15}, reason: 'close_requested'})});
  await tick();
  assert.ok(f.peer.closed); assert.ok(f.channel.closed);
  assert.equal(f.engine.current, null);
  assert.equal(f.states.at(-1), 'closed');
  assert.ok(f.requests.some(r => r.path === '/api/closed'));
});

test('page exit stops hardware and releases transports without pretending to confirm usage', async () => {
  const f = fixture({api: async path => path === '/api/status' ? {state: 'active'} : {sdp: 'v=0', test_seconds: 120}});
  await f.engine.start();
  f.engine.stop('', true); await tick();
  assert.ok(f.tracks[0].stopped); assert.ok(f.peer.closed); assert.ok(f.channel.closed);
  assert.equal(f.states.at(-1), 'stopped');
  assert.ok(f.requests.some(r => r.path === '/api/stop'));
});

test('an old permission result cannot overwrite a new session', async () => {
  let resolve;
  let calls = 0;
  const f = fixture({media: () => ++calls === 1 ? new Promise(r => {resolve = r;}) : Promise.resolve({getTracks: () => []})});
  const first = f.engine.start(); f.engine.stop();
  await f.engine.start(); const current = f.engine.current;
  resolve(f.stream); await first;
  assert.equal(f.engine.current, current);
  assert.ok(f.tracks[0].stopped);
  f.engine.stop('', true);
});

test('selected voice reaches the server and comparison starts only on a running channel', async () => {
  const f = fixture({api: async path => path === '/api/session'
    ? {sdp: 'v=0', test_seconds: 120, comparison_instruction: 'Read the same prepared Igbo passage.'} : {state: 'active'}});
  assert.equal(f.engine.readComparison(), false);
  await f.engine.start({voice: 'owerri-marin'});
  assert.equal(f.requests[0].body.voice, 'owerri-marin');
  assert.equal(f.engine.readComparison(), false);
  f.channel.onmessage({data: JSON.stringify({type: 'session.started'})});
  assert.equal(f.engine.readComparison(), true);
  assert.deepEqual(f.channel.sent.at(-1), {type: 'session.instructions.append', delegation_id: null,
    event_id: 'comparison_' + 'a'.repeat(32), content: 'Read the same prepared Igbo passage.'});
  assert.ok(!f.events.some(e => e.type === 'session.output_transcript.delta'));
  f.engine.stop('', true);
  assert.equal(f.engine.readComparison(), false);
});
