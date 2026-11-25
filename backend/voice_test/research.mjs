// Render only bounded, real research output. Source data never becomes HTML.
const MAX_TEXT = 6000;
export function sourceURL(value) {
  if (typeof value !== 'string' || value.length > 2048 || /[\s\x00-\x1f]/u.test(value)) return null;
  try {
    const url = new URL(value);
    return ['http:', 'https:'].includes(url.protocol) && url.hostname && !url.username && !url.password ? url.href : null;
  } catch { return null; }
}

export function citationSegments(card) {
  const chars = Array.from(typeof card?.text === 'string' ? card.text : '').slice(0, MAX_TEXT);
  const annotations = (Array.isArray(card?.citations) ? card.citations : []).slice(0, 12).flatMap(citation => {
    if (!citation || typeof citation !== 'object') return [];
    const url = sourceURL(citation.url);
    if (!url) return [];
    let {start, end} = citation;
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start || end > chars.length) start = end = chars.length;
    const title = typeof citation.title === 'string' && citation.title.trim() ? citation.title.slice(0, 200) : new URL(url).hostname;
    return [{url, title, start, end}];
  }).sort((a, b) => a.start - b.start || a.end - b.end);
  const segments = [];
  let cursor = 0;
  for (const citation of annotations) {
    if (citation.start > cursor) segments.push({text: chars.slice(cursor, citation.start).join('')});
    const label = citation.start >= cursor ? chars.slice(citation.start, citation.end).join('') : '';
    // Replace opaque citation markers; preserve ordinary cited text and accents.
    segments.push({url: citation.url, title: citation.title,
      text: label && !/[\uE200-\uE2FF]/u.test(label) ? label : ` [${citation.title}]`});
    cursor = Math.max(cursor, citation.end);
  }
  if (cursor < chars.length) segments.push({text: chars.slice(cursor).join('')});
  return segments;
}

export function renderResearch(document, container, snapshot) {
  const cards = Array.isArray(snapshot?.cards) ? snapshot.cards.slice(-5) : [];
  container.replaceChildren(...cards.map(card => {
    const article = document.createElement('article'); article.className = 'research-card';
    const title = document.createElement('h3'); title.textContent = '☀ Web research';
    const body = document.createElement('p'); body.className = 'research-answer'; body.lang = 'ig';
    for (const segment of citationSegments(card)) {
      if (segment.url) {
        const link = document.createElement('a');
        link.href = segment.url; link.textContent = segment.text; link.title = segment.title;
        link.target = '_blank'; link.rel = 'noopener noreferrer'; link.referrerPolicy = 'no-referrer';
        body.append(link);
      } else body.append(document.createTextNode(segment.text));
    }
    article.append(title, body);
    if (card.uncited || !citationSegments(card).some(s => s.url) || card.truncated) {
      const note = document.createElement('p'); note.className = 'small';
      note.textContent = card.truncated ? 'Long research answer shortened for this trial.' : 'No source citations were returned for this answer.';
      article.append(note);
    }
    return article;
  }));
}
