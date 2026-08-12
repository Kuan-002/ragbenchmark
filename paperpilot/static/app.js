let sessionId = null;
let mode = 'agentic';
let turnCounter = 0;
let currentPaper = null;

const methodLabels = {
  agentic: 'Agentic-RAG',
  rrf_ce: 'Traditional BM25+CE',
};

function setMode(nextMode) {
  mode = nextMode;
  document.querySelectorAll('.method-btn').forEach(btn => btn.classList.remove('active'));
  const active = document.getElementById(`method-${nextMode}`);
  if (active) active.classList.add('active');
  document.getElementById('method-title').textContent = methodLabels[nextMode] || nextMode;
}

const dropZone = document.getElementById('drop-zone');
dropZone.addEventListener('dragover', event => {
  event.preventDefault();
  dropZone.classList.add('dragging');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));
dropZone.addEventListener('drop', event => {
  event.preventDefault();
  dropZone.classList.remove('dragging');
  const file = event.dataTransfer.files[0];
  if (file) uploadPDF(file);
});

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) uploadPDF(file);
}

async function uploadPDF(file) {
  showStatus(`Parsing ${file.name}...`, 'loading');
  const form = new FormData();
  form.append('file', file);
  try {
    const res = await fetch('/api/upload/pdf', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Upload failed');
    initSession(data);
  } catch (error) {
    showStatus(error.message, 'error');
  }
}

async function loadDemoList() {
  try {
    const res = await fetch('/api/demo/papers');
    const data = await res.json();
    if (!Array.isArray(data.papers) || data.papers.length === 0) return;

    const list = document.getElementById('demo-list');
    list.innerHTML = data.papers.map(p => `
      <button class="demo-card" onclick="loadDemo('${escAttr(p.key)}', this)">
        <strong>${escHtml(p.display_name || p.key)}</strong>
        <span>${escHtml(p.title || '')}</span>
        <small>${Number(p.num_chunks || 0)} chunks</small>
      </button>
    `).join('');
    document.getElementById('demo-section').classList.remove('hidden');
  } catch {}
}

async function loadDemo(key, button) {
  button.disabled = true;
  showStatus('Loading demo paper...', 'loading');
  const form = new FormData();
  form.append('paper_key', key);
  try {
    const res = await fetch('/api/demo/load', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Load failed');
    initSession(data);
  } catch (error) {
    showStatus(error.message, 'error');
    button.disabled = false;
  }
}

function initSession(data) {
  sessionId = data.session_id;
  currentPaper = data;

  document.getElementById('paper-title').textContent = data.title || 'Untitled paper';
  document.getElementById('paper-abstract').textContent = data.abstract || 'No abstract available.';
  document.getElementById('chunk-count').textContent = data.num_chunks || 0;
  document.getElementById('section-count').textContent = (data.sections || []).length;
  document.getElementById('topbar-status').textContent = 'Paper loaded';

  const sectionList = document.getElementById('section-list');
  sectionList.innerHTML = (data.sections || [])
    .map(section => `<li>${escHtml(section || 'Untitled section')}</li>`)
    .join('');

  document.getElementById('upload-screen').classList.add('hidden');
  document.getElementById('main-screen').classList.remove('hidden');
  document.getElementById('qa-history').innerHTML = `
    <div class="empty-state">
      <p>Ask a question to compare how the selected method retrieves evidence and answers.</p>
    </div>`;
  setMode(mode);
  document.getElementById('question-input').focus();
}

function resetApp() {
  sessionId = null;
  currentPaper = null;
  document.getElementById('main-screen').classList.add('hidden');
  document.getElementById('upload-screen').classList.remove('hidden');
  document.getElementById('topbar-status').textContent = 'No paper loaded';
  document.getElementById('upload-status').classList.add('hidden');
  document.getElementById('file-input').value = '';
}

async function submitQuestion() {
  const input = document.getElementById('question-input');
  const question = input.value.trim();
  if (!question || !sessionId) return;

  input.value = '';
  const history = document.getElementById('qa-history');
  const empty = history.querySelector('.empty-state');
  if (empty) empty.remove();

  const turnId = ++turnCounter;
  const item = document.createElement('article');
  item.className = 'qa-turn';
  item.innerHTML = `
    <div class="question-row">
      <span class="method-pill">${escHtml(methodLabels[mode])}</span>
      <p>${escHtml(question)}</p>
    </div>
    <div class="answer-card" id="answer-${turnId}">
      <div class="skeleton wide"></div>
      <div class="skeleton"></div>
      <div class="skeleton short"></div>
    </div>`;
  history.appendChild(item);
  history.scrollTop = history.scrollHeight;
  setComposerDisabled(true);

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        question,
        mode,
        top_k: 5,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Query failed');
    renderAnswer(document.getElementById(`answer-${turnId}`), data, turnId);
  } catch (error) {
    document.getElementById(`answer-${turnId}`).innerHTML = `
      <div class="error-box">${escHtml(error.message)}</div>`;
  } finally {
    setComposerDisabled(false);
    history.scrollTop = history.scrollHeight;
  }
}

function askSuggestion(question) {
  const input = document.getElementById('question-input');
  input.value = question;
  submitQuestion();
}

function renderAnswer(card, data, turnId) {
  const sources = Array.isArray(data.sources) ? data.sources : [];
  const trace = Array.isArray(data.agent_trace) ? data.agent_trace : [];
  const method = data.method_label || methodLabels[data.mode] || data.mode || 'Method';

  card.innerHTML = `
    <div class="answer-meta">
      <span>${escHtml(method)}</span>
      ${data.paper_title ? `<span>${escHtml(data.paper_title)}</span>` : ''}
    </div>
    <div class="answer-text">${renderMarkdown(data.answer || '')}</div>
    <div class="panel-actions">
      ${trace.length ? `<button class="fold-btn" data-panel-button="trace-${turnId}" onclick="togglePanel('trace-${turnId}', ${turnId})">Tool calls</button>` : ''}
      <button class="fold-btn" data-panel-button="sources-${turnId}" onclick="togglePanel('sources-${turnId}', ${turnId})">Evidence (${sources.length})</button>
    </div>
    ${trace.length ? renderTrace(trace, turnId) : ''}
    ${renderSources(sources, turnId)}
  `;

  if (typeof renderMath === 'function') renderMath(card);
}

function renderTrace(trace, turnId) {
  const rows = trace.map(round => {
    const calls = (round.calls || []).map(call => `
      <div class="trace-call">
        <div class="trace-head">
          <strong>${escHtml(call.tool_name || 'tool')}</strong>
          <span>${call.evidence_count == null ? '' : `${call.evidence_count} evidence`}</span>
        </div>
        ${call.reason ? `<p class="trace-reason">${escHtml(call.reason)}</p>` : ''}
        <p>${escHtml(call.observation || '')}</p>
        ${call.added_chunk_ids?.length ? `<small>Added: ${escHtml(call.added_chunk_ids.join(', '))}</small>` : ''}
      </div>
    `).join('');
    return `
      <div class="trace-round">
        <div class="trace-round-title">Round ${Number(round.round || 0)}</div>
        ${calls || '<p class="trace-empty">No tool call.</p>'}
      </div>`;
  }).join('');

  return `<div id="trace-${turnId}" class="fold-panel trace-panel" data-turn-panel="${turnId}">${rows}</div>`;
}

function renderSources(sources, turnId) {
  const rows = sources.map((source, index) => `
    <div class="source-row">
      <div class="source-header">
        <span class="source-rank">${index + 1}</span>
        <strong>${escHtml(source.paper_title || 'Current paper')}</strong>
        <span>${escHtml(source.section || 'Source')}</span>
        <small>${escHtml(source.chunk_id || '')}</small>
      </div>
      <p>${escHtml(source.text || '')}</p>
    </div>
  `).join('');

  return `<div id="sources-${turnId}" class="fold-panel sources-panel" data-turn-panel="${turnId}">${rows || '<p class="trace-empty">No sources returned.</p>'}</div>`;
}

function togglePanel(id, turnId) {
  const panel = document.getElementById(id);
  if (!panel) return;
  const willOpen = !panel.classList.contains('open');

  document.querySelectorAll(`[data-turn-panel="${turnId}"]`).forEach(item => {
    item.classList.remove('open');
  });
  document.querySelectorAll(`[data-panel-button$="-${turnId}"]`).forEach(button => {
    button.classList.remove('active');
    button.setAttribute('aria-expanded', 'false');
  });

  if (willOpen) {
    panel.classList.add('open');
    const button = document.querySelector(`[data-panel-button="${id}"]`);
    if (button) {
      button.classList.add('active');
      button.setAttribute('aria-expanded', 'true');
    }
  }
}

function handleKey(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    submitQuestion();
  }
}

function showStatus(message, type) {
  const status = document.getElementById('upload-status');
  status.textContent = message;
  status.className = `status ${type}`;
  status.classList.remove('hidden');
}

function setComposerDisabled(disabled) {
  document.getElementById('question-input').disabled = disabled;
  document.getElementById('send-btn').disabled = disabled;
}

function renderMarkdown(value) {
  return escHtml(value)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\[(\d+(?:,\s*\d+)*)\]/g, '<sup>[$1]</sup>')
    .replace(/\n/g, '<br>');
}

function escHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function escAttr(value) {
  return escHtml(value).replace(/`/g, '&#96;');
}

document.addEventListener('DOMContentLoaded', () => {
  loadDemoList();
  setMode('agentic');
});
