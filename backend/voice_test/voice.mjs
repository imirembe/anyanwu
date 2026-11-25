import {Captions, VoiceTrial} from './controller.mjs';
import {renderResearch} from './research.mjs';

const $ = id => document.getElementById(id);
const captions = new Captions();
let config = null;
let seconds = 0;
let finalized = false;
let started = false;
let researchSignature = '';
let designing = false;
const ready = () => Boolean(config?.configured && config?.web_search === true && config?.voices?.length && !designing);
function voiceControls(active = Boolean(trial.current)) {
  $('voice').disabled = active || !ready();
  $('design').disabled = active || !ready() || config.voice_design_state !== 'ready';
  $('start').disabled = active || !ready();
}
function loadVoices(previous) {
  $('voice').replaceChildren(...(config.voices || []).map(voice => {
    const option = document.createElement('option'); option.value = voice.id; option.textContent = voice.label; return option;
  }));
  const ids = config.voices?.map(v => v.id) || [];
  $('voice').value = ids.includes(previous) ? previous : config.default_voice;
  $('comparison-text').textContent = config.comparison_text || '';
  $('design-status').textContent = config.voice_design_state === 'saved' ? 'Saved experimental voice available in the menu.'
    : ['unconfirmed', 'rejected'].includes(config.voice_design_state) ? 'Creation was already attempted. Check OpenAI Voices before another attempt.' : '';
}
const labels = {
  microphone: 'Allow microphone access to begin.', connecting: 'Connecting to the voice model…',
  active: 'Listening · speak naturally', closing: 'Microphone off · confirming the session has ended…',
  closed: 'Session ended · final voice usage confirmed', stopped: 'Microphone off',
  speaker: 'Tap Enable speaker to hear the reply.',
};

function renderCaptions() {
  $('empty').hidden = captions.rows.length > 0;
  const container = document.querySelector('.conversation');
  const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100;
  $('messages').replaceChildren(...captions.rows.map(row => {
    const item = document.createElement('li'); item.className = `message ${row.role}`;
    const label = document.createElement('span'); label.className = 'sender';
    label.textContent = row.role === 'user' ? 'You' : '☀ Anyanwụ';
    const body = document.createElement('p'); body.className = 'bubble';
    if (row.role === 'assistant') body.lang = 'ig';
    body.textContent = row.text;
    item.append(label, body); return item;
  }));
  if (nearBottom) container.scrollTop = container.scrollHeight;
}

function usage() {
  $('usage').textContent = started
    ? `Voice usage: ${Math.ceil(seconds)} s · ${finalized ? 'final voice total' : 'latest observed, not final'} · research and search charges are separate`
    : 'No session started.';
}

const api = async (path, data, keepalive = false) => {
  const abort = new AbortController();
  const timeout = setTimeout(() => abort.abort(), path === '/api/voice-design' ? 60000 : path === '/api/session' ? 45000 : 5000);
  try {
    const response = await fetch(path, {method: 'POST', credentials: 'same-origin', cache: 'no-store', keepalive,
      signal: abort.signal, headers: {'Content-Type': 'application/json', 'X-Trial-Token': config.nonce}, body: JSON.stringify(data)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'The local server could not complete the request.');
    return result;
  } finally { clearTimeout(timeout); }
};
const trial = new VoiceTrial({api, output: $('speaker'),
  state: next => {
    const active = Boolean(trial.current);
    $('status').textContent = labels[next] || next;
    voiceControls(active);
    $('stop').disabled = !active || next === 'closing';
    $('play').hidden = next !== 'speaker';
    $('compare').disabled = !active || !['active', 'speaker'].includes(next);
  },
  problem: message => { $('error').textContent = message; $('error').hidden = false; },
  event: e => {
    if (captions.append(e)) renderCaptions();
    if (e.type === 'local.status' && e.research) {
      const signature = JSON.stringify(e.research);
      if (signature !== researchSignature) {
        researchSignature = signature;
        const scroll = document.querySelector('.conversation');
        const nearBottom = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 100;
        renderResearch(document, $('research-results'), e.research);
        const status = {searching: 'Searching the web…', answering: 'Preparing the research answer…',
          complete: 'Web research completed', error: 'The web lookup did not finish', idle: ''}[e.research.status] || '';
        $('research-status').textContent = status;
        $('research').hidden = !status && !e.research.cards?.length;
        if (nearBottom) scroll.scrollTop = scroll.scrollHeight;
      }
    }
    const observed = e.type === 'local.status' ? e.seconds : e.usage?.seconds;
    if (typeof observed === 'number' && Number.isFinite(observed)) seconds = Math.max(seconds, observed);
    if (e.type === 'session.closed' || (e.type === 'local.status' && e.confirmed)) finalized = true;
    if (e.type === 'session.started' || e.type === 'local.status' || e.type === 'session.usage.updated') started = true;
    usage();
  },
});

$('start').addEventListener('click', () => {
  $('error').hidden = true; captions.clear(); renderCaptions();
  researchSignature = ''; $('research').hidden = true; $('research-results').replaceChildren(); $('research-status').textContent = '';
  seconds = 0; finalized = false; started = false; usage();
  trial.start({voice: $('voice').value});
});
$('stop').addEventListener('click', () => trial.stop());
$('compare').addEventListener('click', () => {
  if (trial.readComparison()) $('status').textContent = 'Comparison reading requested · listen to the actual speech';
});
$('design').addEventListener('click', async () => {
  if (designing || trial.current || !ready()) return;
  designing = true; voiceControls();
  $('design-status').textContent = 'Creating the experimental voice…';
  try {
    const result = await api('/api/voice-design', {create: true});
    Object.assign(config, result); loadVoices('owerri-designed');
    $('design-status').textContent = result.metadata_saved ? 'Saved. Choose the designed voice for your next session.'
      : 'Created and available for this run. Local metadata could not be saved; the voice remains in your OpenAI project.';
  } catch (error) {
    config.voice_design_state = 'unconfirmed';
    $('design-status').textContent = error.message || 'Creation was not confirmed. Check the OpenAI Voices page before retrying.';
  } finally { designing = false; voiceControls(); }
});
$('play').addEventListener('click', async () => {
  try { await $('speaker').play(); $('play').hidden = true; $('status').textContent = 'Listening · speak naturally'; }
  catch { $('status').textContent = 'The browser blocked the speaker. Check its audio permissions.'; }
});
$('clear').addEventListener('click', () => { captions.clear(); renderCaptions(); });
window.addEventListener('pagehide', () => {
  trial.stop('', true); captions.clear(); renderCaptions();
  researchSignature = ''; $('research-results').replaceChildren(); $('research-status').textContent = ''; $('research').hidden = true;
});
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') trial.stop();
});

try {
  const response = await fetch('/api/config', {cache: 'no-store'});
  if (!response.ok) throw new Error('Open the exact local URL printed in Terminal.');
  config = await response.json();
  if (!navigator.mediaDevices?.getUserMedia || !window.RTCPeerConnection) throw new Error('Use a browser with microphone and WebRTC support on this Mac.');
  loadVoices(); voiceControls();
  $('status').textContent = !config.configured ? 'Preview only · enter your API key through the Terminal command to test voice.'
    : !config.web_search || !config.voices ? 'Restart the voice-test Terminal command to load voice comparison.' : 'Ready · Owerri voice comparison and web search available';
} catch (error) { $('status').textContent = error.message; }
