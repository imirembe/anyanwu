// Browser WebRTC only. No MediaRecorder, storage, audio SDK or transcript upload.
export class Captions {
  constructor() { this.clear(); }
  clear() { this.rows = []; this.seen = new Set(); this.serial = 0; }
  append(event) {
    const role = event.type === 'session.input_transcript.delta' ? 'user'
      : event.type === 'session.output_transcript.delta' ? 'assistant' : null;
    if (!role || typeof event.delta !== 'string' || !event.delta || event.delta.length > 8000) return false;
    if (event.event_id && this.seen.has(event.event_id)) return false;
    if (event.event_id) {
      this.seen.add(event.event_id);
      if (this.seen.size > 512) this.seen.delete(this.seen.values().next().value);
    }
    const start = Number.isFinite(event.start_ms) ? event.start_ms : null;
    const end = Number.isFinite(event.end_ms) ? event.end_ms : null;
    const last = this.rows.at(-1);
    // Timing is only a display grouping heuristic; it does not mark a completed turn.
    if (last && last.role === role && last.text.length + event.delta.length <= 8000
        && start !== null && last.end !== null && start >= last.start && start - last.end < 1800) {
      last.text += event.delta; last.end = end;
    } else {
      this.rows.push({id: ++this.serial, role, text: event.delta, start, end});
    }
    while (this.rows.length > 30 || this.rows.reduce((n, row) => n + row.text.length, 0) > 24000) this.rows.shift();
    return true;
  }
}

export function waitForICE(peer, timeout = 10000) {
  if (peer.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise((resolve, reject) => {
    const done = error => {
      clearTimeout(timer);
      peer.removeEventListener('icegatheringstatechange', check);
      peer.removeEventListener('connectionstatechange', closed);
      error ? reject(error) : resolve();
    };
    const check = () => { if (peer.iceGatheringState === 'complete') done(); };
    const closed = () => { if (peer.connectionState === 'closed') done(new Error('Connection closed.')); };
    const timer = setTimeout(() => done(new Error('Network setup timed out. No session was requested.')), timeout);
    peer.addEventListener('icegatheringstatechange', check);
    peer.addEventListener('connectionstatechange', closed);
    check();
  });
}

export class VoiceTrial {
  constructor({api, output, state, event, problem, media = () => navigator.mediaDevices.getUserMedia({audio: true}),
               peer = () => new RTCPeerConnection(), id = () => crypto.randomUUID()}) {
    Object.assign(this, {api, output, state, event, problem, media, peer, id});
    this.current = null;
  }
  async start({voice} = {}) {
    if (this.current) return;
    const c = {id: this.id(), stream: null, pc: null, dc: null, posted: false, started: false,
      stopping: false, finished: false, closeSent: false, comparison: null,
      poll: null, polling: false, deadline: null, finalTimer: null};
    this.current = c;
    this.state('microphone');
    try {
      const stream = await this.media();
      if (c.finished || this.current !== c) { stream.getTracks().forEach(track => track.stop()); return; }
      c.stream = stream;
      c.pc = this.peer();
      stream.getTracks().forEach(track => c.pc.addTrack(track, stream));
      c.pc.ontrack = e => {
        if (c.finished || c.stopping || this.current !== c) return;
        this.output.srcObject = e.streams[0] || new MediaStream([e.track]);
        this.output.play().catch(() => this.state('speaker'));
      };
      c.pc.onconnectionstatechange = () => {
        if (c.pc?.connectionState === 'failed') this.stop('The audio connection failed.');
      };
      c.dc = c.pc.createDataChannel('oai-events');
      c.dc.onmessage = e => {
        if (c.finished || this.current !== c || typeof e.data !== 'string' || e.data.length > 65536) return;
        try { this.receive(c, JSON.parse(e.data)); } catch { /* Ignore malformed events. */ }
      };
      c.dc.onclose = () => { if (!c.finished) this.stop('The voice connection closed.'); };
      c.dc.onerror = () => this.stop('The voice connection failed.');
      this.state('connecting');
      await c.pc.setLocalDescription(await c.pc.createOffer());
      await waitForICE(c.pc);
      if (c.finished || this.current !== c) return;
      c.posted = true;
      // Do not automatically retry creation, including on ambiguous network errors.
      const result = await this.api('/api/session', {request_id: c.id, sdp: c.pc.localDescription.sdp, ...(voice ? {voice} : {})});
      if (c.finished) return;
      c.comparison = result.comparison_instruction;
      await c.pc.setRemoteDescription({type: 'answer', sdp: result.sdp});
      c.deadline = setTimeout(() => this.stop(), result.test_seconds * 1000);
      this.poll(c);
      if (c.stopping) this.sendClose(c);
    } catch (error) {
      if (c.finished || this.current !== c) return;
      this.problem(error?.message || 'The voice session could not start.');
      if (c.posted) this.stop();
      else this.cleanup(c, false);
    }
  }
  receive(c, e) {
    if (e.type === 'session.started') {
      c.started = true;
      this.event(e);
      c.stopping ? this.sendClose(c) : this.state('active');
    } else if (e.type === 'session.closed') {
      // Keep this listener registered before sending session.close; no fabricated confirmation.
      c.stream?.getTracks().forEach(track => track.stop());
      this.output.pause();
      this.event(e);
      // Collect the last research/citations too. Lifecycle snapshots omit output
      // arrays, and the final individual message may have used the sideband.
      this.api('/api/closed', {request_id: c.id, event: {type: e.type, usage: e.usage, reason: e.reason}}, true)
        .then(status => {
          if (!c.finished && this.current === c) this.event({type: 'local.status', ...status});
        }).catch(() => {}).finally(() => this.cleanup(c, true));
    } else if (e.type === 'error') {
      this.stop('The voice service reported an error.');
    } else {
      this.event(e);
    }
  }
  sendClose(c) {
    if (c.started && c.dc?.readyState === 'open' && !c.closeSent) {
      c.dc.send(JSON.stringify({type: 'session.close', event_id: 'browser_close'}));
      c.closeSent = true;
    }
  }
  readComparison() {
    const c = this.current;
    if (!c || !c.started || c.stopping || c.finished || c.dc?.readyState !== 'open'
        || typeof c.comparison !== 'string' || c.comparison.length > 2000) return false;
    c.dc.send(JSON.stringify({type: 'session.instructions.append', delegation_id: null,
      event_id: `comparison_${this.id()}`, content: c.comparison}));
    return true;
  }
  stop(message = '', immediate = false) {
    const c = this.current;
    if (!c || c.finished) return;
    if (message) this.problem(message);
    c.stopping = true;
    // Stop microphone hardware immediately; retain the data channel for final usage.
    c.stream?.getTracks().forEach(track => track.stop());
    this.output.pause();
    this.state('closing');
    if (!c.posted) { this.cleanup(c, false); return; }
    this.sendClose(c);
    this.api('/api/stop', {request_id: c.id}, true).catch(() => {});
    if (immediate) { this.cleanup(c, false); return; }
    if (!c.finalTimer) c.finalTimer = setTimeout(() => {
      this.problem('Final usage is unconfirmed. Check API usage before another trial.');
      this.cleanup(c, false);
    }, 20000);
    if (!c.poll) this.poll(c);
  }
  async poll(c) {
    if (c.finished || this.current !== c || c.polling) return;
    c.polling = true;
    try {
      const status = await this.api('/api/status', {request_id: c.id});
      if (c.finished || this.current !== c) return;
      this.event({type: 'local.status', ...status});
      if (status.confirmed) { this.cleanup(c, true); return; }
      if (status.state === 'rejected') { this.cleanup(c, false); return; }
      if (status.state === 'unconfirmed') {
        this.stop(status.reason || 'Session finalization is unconfirmed.');
      } else if (status.state === 'closing' && !c.stopping) this.stop();
    } catch { /* A missing page heartbeat also triggers the server watchdog. */ }
    finally { c.polling = false; }
    if (!c.finished) c.poll = setTimeout(() => { c.poll = null; this.poll(c); }, 1500);
  }
  cleanup(c, confirmed) {
    if (c.finished) return;
    c.finished = true;
    clearTimeout(c.deadline); clearTimeout(c.poll); clearTimeout(c.finalTimer);
    c.stream?.getTracks().forEach(track => track.stop());
    if (c.dc) { c.dc.onclose = null; c.dc.onerror = null; c.dc.onmessage = null; c.dc.close(); }
    if (c.pc) { c.pc.ontrack = null; c.pc.onconnectionstatechange = null; c.pc.close(); }
    this.output.pause(); this.output.srcObject = null;
    if (this.current === c) { this.current = null; this.state(confirmed ? 'closed' : 'stopped'); }
  }
}
