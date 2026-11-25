import test from 'node:test';
import assert from 'node:assert/strict';
import {sourceURL, citationSegments, renderResearch} from './research.mjs';

test('citations preserve codepoint offsets, dotted letters, tone marks and emoji', () => {
  const text = '☀ Ị́ nọ ebe a? [1]';
  const start = Array.from(text).indexOf('[');
  const segments = citationSegments({text, citations: [{url: 'https://example.org/igbo', title: 'Source', start, end: start + 3}]});
  assert.equal(segments.map(s => s.text).join(''), text);
  assert.equal(segments.at(-1).url, 'https://example.org/igbo');
  const astral = citationSegments({text: '🌞 Ndewo [1]', citations: [{url: 'https://example.org', start: 8, end: 11}]});
  assert.equal(astral.map(s => s.text).join(''), '🌞 Ndewo [1]');
  assert.equal(astral.at(-1).text, '[1]');
});

test('unsafe links are rejected and malformed offsets become source endnotes', () => {
  for (const url of ['javascript:alert(1)', 'data:text/html,hi', 'file:///etc/passwd', 'https://user:password@example.org', 'https://example.org/\nhi']) assert.equal(sourceURL(url), null);
  const segments = citationSegments({text: 'Ndewo.', citations: [{url: 'https://example.org', title: 'Source', start: -1, end: 999}]});
  assert.equal(segments[0].text, 'Ndewo.');
  assert.equal(segments[1].text, ' [Source]');
});

test('overlapping annotations retain both clickable sources without duplicating text', () => {
  const segments = citationSegments({text: 'Ndewo.', citations: [
    {url: 'https://a.example', title: 'A', start: 0, end: 6}, {url: 'https://b.example', title: 'B', start: 0, end: 6}]});
  assert.deepEqual(segments.map(s => s.text), ['Ndewo.', ' [B]']);
  assert.equal(segments.filter(s => s.url).length, 2);
});

class Node {
  constructor(tag) { this.tag = tag; this.children = []; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
}
const document = {createElement: tag => new Node(tag), createTextNode: text => ({tag: '#text', textContent: text})};

test('research renders untrusted titles/text as text with safe clickable inline links', () => {
  const container = new Node('div');
  renderResearch(document, container, {cards: [{text: '<img src=x onerror=alert(1)> Ndewo [1]',
    citations: [{url: 'https://example.org', title: '<script>bad</script>', start: 33, end: 36}]}]});
  const article = container.children[0];
  const body = article.children[1];
  assert.equal(body.lang, 'ig');
  assert.ok(body.children[0].textContent.startsWith('<img'));
  const link = body.children.find(c => c.tag === 'a');
  assert.equal(link.title, '<script>bad</script>');
  assert.equal(link.rel, 'noopener noreferrer');
  assert.equal(link.referrerPolicy, 'no-referrer');
  assert.equal(link.target, '_blank');
  assert.ok(!JSON.stringify(article).includes('innerHTML'));
});

test('UI bounds cards and text and labels missing citations', () => {
  const container = new Node('div');
  renderResearch(document, container, {cards: Array.from({length: 20}, () => ({text: 'ị'.repeat(10000), citations: [], uncited: true}))});
  assert.equal(container.children.length, 5);
  assert.equal(container.children[0].children[1].children[0].textContent.length, 6000);
  assert.equal(container.children[0].children[2].textContent, 'No source citations were returned for this answer.');
});
