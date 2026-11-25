const $ = selector => document.querySelector(selector);
const nonce = $('meta[name="review-nonce"]').content;
const reviewed = new Set();
let total = 0;
function progress() { $('#progress').textContent = `${reviewed.size} / ${total} saved`; }
function hasRating(review) {
  return ['intelligibility', 'cadence', 'pronunciation', 'transcript_match'].some(key => review[key] !== null)
    || review.notes.trim().length > 0;
}
async function load() {
  const response = await fetch('/api/pilot', {cache: 'no-store'});
  if (!response.ok) throw new Error('The local sample could not be loaded. Restart the listening server.');
  const {manifest, reviews} = await response.json();
  total = manifest.clips.length;
  const seconds = manifest.clips.reduce((sum, clip) => sum + clip.duration_seconds, 0);
  const bytes = manifest.clips.reduce((sum, clip) => sum + clip.bytes, 0);
  $('#summary').textContent = `${total} clips · ${manifest.speaker_count} speaker · ${Math.round(seconds)} seconds · ${(bytes/1048576).toFixed(2)} MiB`;
  for (const [index, clip] of manifest.clips.entries()) {
    const node = $('#clip-template').content.cloneNode(true);
    const article = node.querySelector('article');
    node.querySelector('.clip-number').textContent = `Recording ${String(index+1).padStart(2,'0')}`;
    node.querySelector('.clip-meta').textContent = `${clip.speaker_label} · ${clip.duration_seconds.toFixed(1)}s`;
    node.querySelector('.transcript').textContent = clip.transcript;
    const audio = node.querySelector('audio');
    audio.src = `/audio/${clip.id}`;
    audio.setAttribute('aria-label', `Play recording ${index+1}`);
    audio.addEventListener('play', () => {
      document.querySelectorAll('audio').forEach(other => { if (other !== audio) other.pause(); });
    });
    const link = node.querySelector('.source-link');
    link.href = `https://huggingface.co/datasets/google/WaxalNLP/viewer/ibo_tts/train?row=${clip.source_row_index}`;
    const form = node.querySelector('form');
    const status = node.querySelector('.save-state');
    const saved = reviews[clip.id];
    if (saved) {
      for (const name of ['intelligibility','cadence','pronunciation','transcript_match','notes']) {
        form.elements.namedItem(name).value = saved[name] ?? '';
      }
      if (hasRating(saved)) reviewed.add(clip.id);
      status.textContent = 'Saved locally';
    }
    form.addEventListener('input', () => { status.textContent = 'Unsaved changes'; });
    form.addEventListener('submit', async event => {
      event.preventDefault();
      const button = form.querySelector('button');
      const values = new FormData(form);
      const review = {clip_id: clip.id, audio_sha256: clip.sha256, notes: values.get('notes'),
        transcript_match: values.get('transcript_match') || null};
      for (const key of ['intelligibility','cadence','pronunciation']) {
        review[key] = values.get(key) ? Number(values.get(key)) : null;
      }
      button.disabled = true;
      status.textContent = 'Saving…';
      try {
        const response = await fetch('/api/review', {method:'POST',
          headers:{'Content-Type':'application/json','X-Review-Nonce':nonce}, body:JSON.stringify(review)});
        if (!response.ok) throw new Error('Could not save. Check the server and try again.');
        const result = await response.json();
        status.textContent = 'Saved locally';
        if (hasRating(result.saved)) reviewed.add(clip.id); else reviewed.delete(clip.id);
        progress();
      } catch (error) { status.textContent = error.message; }
      finally { button.disabled = false; }
    });
    $('#clips').append(article);
  }
  progress();
}
load().catch(error => { $('#error').hidden = false; $('#error').textContent = error.message; $('#summary').textContent = 'Sample unavailable'; });
