/* PaperPilot frontend */

// ── i18n ──────────────────────────────────────────────────────────────────────
const LANGS = {
  en: {
    'nav.upload':   'Single Paper',
    'nav.library':  'Paper Library',
    'mode.label':   'Retrieval',
    'mode.ce':      'CE (Precise)',
    'mode.rrf':     'RRF (Fast)',

    'upload.title':           'Paper Q&A Assistant',
    'upload.subtitle':        'Upload an NLP paper and ask questions in natural language. Answers are source-cited.',
    'upload.tabPdf':          'Upload PDF',
    'upload.tabArxiv':        'arXiv ID',
    'upload.dropHint':        'Click or drag a PDF file here',
    'upload.dropSub':         'Supports NLP papers in arXiv format',
    'upload.arxivPlaceholder':'e.g. 2310.06825 or https://arxiv.org/abs/2310.06825',
    'upload.arxivBtn':        'Import',
    'upload.parsing':         'Parsing {name}...',
    'upload.downloading':     'Downloading arXiv:{id}...',
    'upload.loadingDemo':     'Loading demo paper...',
    'upload.error':           'Error: ',

    'demo.label':    'Or try a demo paper',
    'demo.loading':  'Loading...',

    'paper.loaded':   'Paper loaded',
    'paper.sections': 'Sections',
    'paper.change':   'Change paper',

    'qa.placeholder': 'Ask a question, e.g.: What architecture does the model use?',
    'qa.hint':        'Optimized for NLP papers. Enter to send, Shift+Enter for a new line',
    'qa.sources':     'View {n} source(s)',
    'qa.expand':      'Show full text',
    'qa.helpful':     'Was this answer helpful?',
    'qa.yes':         'Yes',
    'qa.no':          'No',
    'qa.modeCE':      'CE Precise',
    'qa.modeRRF':     'RRF Fast',

    'type.factual':     'Factual',
    'type.why_how':     'Explanatory',
    'type.numerical':   'Numerical',
    'type.methodology': 'Methodology',
    'type.comparison':  'Comparison',

    'hint.why_how':   'This type of question may require synthesising multiple paragraphs. Treat the answer as a reference.',
    'hint.numerical': 'Numerical answers may be in tables; retrieval may be incomplete.',

    'feedback.title':        'What went wrong?',
    'feedback.desc':         'Your feedback helps us improve retrieval quality.',
    'feedback.irrelevant':   'Irrelevant paragraphs',
    'feedback.irrelevantDesc':'Retrieved content is unrelated to the question.',
    'feedback.incomplete':   'Relevant but incomplete',
    'feedback.incompleteDesc':'Found related paragraphs but missing key information.',
    'feedback.skip':         'Skip',
  },
};

let _lang = 'en';

function t(key, vars = {}) {
  let s = (LANGS[_lang] || LANGS.en)[key] || key;
  for (const [k, v] of Object.entries(vars)) s = s.replace(`{${k}}`, v);
  return s;
}

function applyLang() {
  document.documentElement.lang = 'en';
  document.querySelectorAll('[data-i18n]').forEach(el => {
    el.textContent = t(el.dataset.i18n);
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  });
  document.querySelectorAll('a[href="/library"], a[href^="/library?"]').forEach(el => {
    el.textContent = 'Paper Library';
    el.setAttribute('href', '/library?ui=en-only-2');
  });
  document.querySelectorAll('a[href="/"], a[href^="/?"]').forEach(el => {
    el.textContent = 'Single Paper';
    el.setAttribute('href', '/?ui=en-only-2');
  });
}

function toggleLang() {
  _lang = 'en';
  applyLang();
}

// ── State ─────────────────────────────────────────────────────────────────────
let sessionId    = null;
let mode         = 'ce';
let pendingQ     = null;
let _turnCounter = 0;

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
  event.currentTarget.classList.add('active');
  document.getElementById(`tab-${tab}`).classList.remove('hidden');
}

// ── Mode toggle ───────────────────────────────────────────────────────────────
function setMode(m) {
  mode = m;
  document.getElementById('btn-ce').classList.toggle('active',  m === 'ce');
  document.getElementById('btn-rrf').classList.toggle('active', m === 'rrf');
}

// ── Drag-and-drop ─────────────────────────────────────────────────────────────
const dropZone = document.getElementById('drop-zone');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.style.borderColor = '#2563eb'; });
dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = ''; });
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.style.borderColor = '';
  const file = e.dataTransfer.files[0];
  if (file && file.type === 'application/pdf') uploadPDF(file);
});

// ── File select ───────────────────────────────────────────────────────────────
function handleFileSelect(evt) {
  const file = evt.target.files[0];
  if (file) uploadPDF(file);
}

// ── Upload PDF ────────────────────────────────────────────────────────────────
async function uploadPDF(file) {
  showStatus(t('upload.parsing', { name: file.name }), 'loading');
  const form = new FormData();
  form.append('file', file);
  try {
    const res  = await fetch('/api/upload/pdf', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Upload failed');
    initSession(data);
  } catch (e) {
    showStatus(t('upload.error') + e.message, 'error');
  }
}

// ── arXiv import ──────────────────────────────────────────────────────────────
async function handleArxiv() {
  const id = document.getElementById('arxiv-input').value.trim();
  if (!id) return;
  showStatus(t('upload.downloading', { id }), 'loading');
  const form = new FormData();
  form.append('arxiv_id', id);
  try {
    const res  = await fetch('/api/upload/arxiv', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Import failed');
    initSession(data);
  } catch (e) {
    showStatus(t('upload.error') + e.message, 'error');
  }
}

// ── Demo papers ───────────────────────────────────────────────────────────────
async function loadDemoList() {
  try {
    const res  = await fetch('/api/demo/papers');
    const data = await res.json();
    if (!data.papers || data.papers.length === 0) return;

    const section = document.getElementById('demo-section');
    const list    = document.getElementById('demo-list');
    list.innerHTML = data.papers.map(p => `
      <button class="demo-card" onclick="loadDemo('${escHtml(p.key)}', this)">
        <span class="demo-card-name">${escHtml(p.display_name)}</span>
        <span class="demo-card-meta">${p.num_chunks} chunks</span>
      </button>`).join('');
    section.classList.remove('hidden');
  } catch {}
}

async function loadDemo(key, btn) {
  const prev = btn.querySelector('.demo-card-name').textContent;
  btn.disabled = true;
  btn.querySelector('.demo-card-name').textContent = t('demo.loading');
  showStatus(t('upload.loadingDemo'), 'loading');
  const form = new FormData();
  form.append('paper_key', key);
  try {
    const res  = await fetch('/api/demo/load', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Load failed');
    initSession(data);
  } catch (e) {
    showStatus(t('upload.error') + e.message, 'error');
    btn.disabled = false;
    btn.querySelector('.demo-card-name').textContent = prev;
  }
}

// ── Init session ──────────────────────────────────────────────────────────────
function initSession(data) {
  sessionId = data.session_id;
  document.getElementById('paper-title').textContent    = data.title    || '(No title)';
  document.getElementById('paper-abstract').textContent = data.abstract || '(No abstract)';

  const list = document.getElementById('section-list');
  list.innerHTML = '';
  (data.sections || []).forEach(s => {
    const li = document.createElement('li');
    li.textContent = s;
    list.appendChild(li);
  });

  document.getElementById('upload-screen').classList.add('hidden');
  document.getElementById('main-screen').classList.remove('hidden');
  document.getElementById('mode-toggle').classList.remove('hidden');
  document.getElementById('qa-history').innerHTML = '';
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetApp() {
  sessionId = null;
  document.getElementById('main-screen').classList.add('hidden');
  document.getElementById('upload-screen').classList.remove('hidden');
  document.getElementById('mode-toggle').classList.add('hidden');
  document.getElementById('upload-status').classList.add('hidden');
  document.getElementById('file-input').value = '';
  document.getElementById('arxiv-input').value = '';
}

// ── Submit question ───────────────────────────────────────────────────────────
async function submitQuestion() {
  const input = document.getElementById('question-input');
  const q     = input.value.trim();
  if (!q || !sessionId) return;
  input.value = '';
  pendingQ = q;

  const history = document.getElementById('qa-history');
  const item    = document.createElement('div');
  item.className = 'qa-item';
  item.innerHTML = `<div class="question-bubble">${escHtml(q)}</div>`;
  history.appendChild(item);

  const ansCard = document.createElement('div');
  ansCard.className = 'answer-card';
  ansCard.innerHTML = `
    <div class="skeleton skeleton-line" style="width:80%"></div>
    <div class="skeleton skeleton-line"></div>
    <div class="skeleton skeleton-line"></div>`;
  item.appendChild(ansCard);
  history.scrollTop = history.scrollHeight;

  const turnId = ++_turnCounter;
  ansCard.dataset.turnId = turnId;

  try {
    const res  = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, question: q, mode, top_k: 5 }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Query failed');
    renderAnswer(ansCard, data, q, turnId);
  } catch (e) {
    ansCard.innerHTML = `<p style="color:#dc2626;font-size:14px;">${t('upload.error')}${escHtml(e.message)}</p>`;
  }
  history.scrollTop = history.scrollHeight;
}

// ── Render answer card ────────────────────────────────────────────────────────
function renderAnswer(card, data, question, turnId) {
  const typeBadgeClass = `badge-${data.query_type}`;
  const modeLabel      = data.mode === 'ce' ? t('qa.modeCE') : t('qa.modeRRF');

  let hintHtml = '';
  if (data.hint) {
    const hintText = t(data.hint) !== data.hint ? t(data.hint) : data.hint;
    hintHtml = `<div class="hint-banner">${escHtml(hintText)}</div>`;
  }

  let sourcesHtml = data.sources.map((s, i) => {
    const srcId = `src-text-t${turnId}-${i}`;
    return `
    <div class="source-item">
      <div class="source-header">
        <div class="source-rank">${s.rank}</div>
        <span class="source-section">${escHtml(s.section)}</span>
      </div>
      <div class="source-text" id="${srcId}">${escHtml(s.text)}</div>
      <button class="source-expand" onclick="expandSource(this, '${srcId}')">${t('qa.expand')}</button>
    </div>`;
  }).join('');

  card.innerHTML = `
    <div class="answer-meta">
      <span class="query-type-badge ${typeBadgeClass}">${t('type.' + data.query_type) || data.query_type}</span>
      <span class="mode-chip">${modeLabel}</span>
    </div>
    ${hintHtml}
    <div class="answer-text">${renderMarkdown(data.answer)}</div>
    <button class="sources-toggle" onclick="toggleSources(this)">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="6 9 12 15 18 9"/>
      </svg>
      ${t('qa.sources', { n: data.sources.length })}
    </button>
    <div class="sources-list">${sourcesHtml}</div>
    <div class="feedback-row">
      <span class="feedback-label">${t('qa.helpful')}</span>
      <button class="feedback-btn" data-question="${escHtml(question)}" data-helpful="true"  onclick="handleFeedback(this)">${t('qa.yes')}</button>
      <button class="feedback-btn" data-question="${escHtml(question)}" data-helpful="false" onclick="handleFeedback(this)">${t('qa.no')}</button>
    </div>`;

  if (typeof renderMath === 'function') renderMath(card);
}

// ── Sources toggle ────────────────────────────────────────────────────────────
function toggleSources(btn) {
  btn.classList.toggle('open');
  btn.nextElementSibling.classList.toggle('open');
}

function expandSource(btn, id) {
  document.getElementById(id).classList.add('expanded');
  btn.style.display = 'none';
}

// ── Feedback ──────────────────────────────────────────────────────────────────
let _feedbackBtn      = null;
let _feedbackHelpful  = null;
let _feedbackQuestion = null;

function handleFeedback(btn) {
  const helpful  = btn.dataset.helpful === 'true';
  const question = btn.dataset.question;

  const row = btn.closest('.feedback-row');
  row.querySelectorAll('.feedback-btn').forEach(b => b.classList.remove('chosen-yes','chosen-no'));
  btn.classList.add(helpful ? 'chosen-yes' : 'chosen-no');

  _feedbackBtn      = btn;
  _feedbackHelpful  = helpful;
  _feedbackQuestion = question;

  if (!helpful) {
    document.getElementById('feedback-modal').classList.remove('hidden');
    document.getElementById('modal-backdrop').classList.remove('hidden');
  } else {
    postFeedback(helpful, null);
  }
}

async function submitFailure(failureType) {
  closeModal();
  await postFeedback(_feedbackHelpful, failureType);
}

function closeModal() {
  document.getElementById('feedback-modal').classList.add('hidden');
  document.getElementById('modal-backdrop').classList.add('hidden');
  if (_feedbackHelpful === false && !_feedbackBtn?.dataset.sent) {
    postFeedback(false, null);
  }
}

async function postFeedback(helpful, failureType) {
  if (!sessionId) return;
  try {
    await fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id:   sessionId,
        question:     _feedbackQuestion,
        helpful,
        failure_type: failureType,
      }),
    });
    if (_feedbackBtn) _feedbackBtn.dataset.sent = '1';
  } catch {}
}

// ── Enter key ─────────────────────────────────────────────────────────────────
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    submitQuestion();
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function showStatus(msg, type) {
  const el = document.getElementById('upload-status');
  el.textContent = msg;
  el.className   = `upload-status ${type}`;
  el.classList.remove('hidden');
}

function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function renderMarkdown(s) {
  return escHtml(s)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.+?)`/g, '<code style="background:#f3f4f6;padding:1px 4px;border-radius:3px;font-size:13px">$1</code>')
    .replace(/\[(\d+)\]/g, '<sup style="color:#2563eb;font-weight:600">[$1]</sup>')
    .replace(/\n/g, '<br>');
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  applyLang();
  loadDemoList();
});
