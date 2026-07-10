const form = document.getElementById('chat-form');
const questionInput = document.getElementById('question');
const answerEl = document.getElementById('answer');
const sourcesEl = document.getElementById('sources');
const statusPill = document.getElementById('status-pill');
const documentMeta = document.getElementById('document-meta');
const loadingIndicator = document.getElementById('loading-indicator');
const askBtn = document.getElementById('ask-btn');
const clearBtn = document.getElementById('clear-btn');

function setBusy(isBusy) {
  askBtn.disabled = isBusy;
  clearBtn.disabled = isBusy;
  loadingIndicator.textContent = isBusy ? 'Thinking…' : '';
}

function renderSources(sources) {
  if (!sources.length) {
    sourcesEl.className = 'sources-empty';
    sourcesEl.textContent = 'No sources returned.';
    return;
  }

  sourcesEl.className = 'sources-list';
  sourcesEl.innerHTML = sources
    .map(
      (source) => `
        <article class="source-card">
          <strong>Page ${source.page_number}${source.score !== null && source.score !== undefined ? ` · score ${Number(source.score).toFixed(3)}` : ''}</strong>
          <div class="source-snippet">${escapeHtml(source.snippet)}</div>
        </article>
      `,
    )
    .join('');
}

function renderAnswer(text, isError = false) {
  answerEl.className = isError ? 'answer-text error' : 'answer-text';
  answerEl.textContent = text;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

async function loadDocumentMeta() {
  try {
    const response = await fetch('/document');
    if (!response.ok) {
      statusPill.textContent = 'Source PDF not ready';
      statusPill.style.background = 'rgba(255, 141, 141, 0.12)';
      statusPill.style.color = '#ffb0b0';
      documentMeta.textContent = 'Check your Azure Search configuration.';
      return;
    }

    const data = await response.json();
    statusPill.textContent = 'Source PDF loaded';
    statusPill.style.background = 'rgba(57, 198, 168, 0.14)';
    statusPill.style.color = '#8df0dc';
    documentMeta.textContent = data.title || 'Indexed PDF';
  } catch (error) {
    statusPill.textContent = 'Source PDF not ready';
    documentMeta.textContent = 'Unable to load document metadata.';
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) {
    renderAnswer('Type a question first.', true);
    return;
  }

  setBusy(true);
  renderAnswer('');
  sourcesEl.className = 'sources-empty';
  sourcesEl.textContent = 'Looking up relevant chunks…';

  try {
    const response = await fetch('/ask', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question, top_k: 5 }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Request failed');
    }

    renderAnswer(data.answer || 'No answer returned.');
    renderSources(data.sources || []);
  } catch (error) {
    renderAnswer(error.message || 'Something went wrong.', true);
    sourcesEl.className = 'sources-empty';
    sourcesEl.textContent = 'No sources available.';
  } finally {
    setBusy(false);
  }
});

clearBtn.addEventListener('click', () => {
  questionInput.value = '';
  renderAnswer('Your answer will appear here.');
  sourcesEl.className = 'sources-empty';
  sourcesEl.textContent = 'No sources yet.';
});

loadDocumentMeta();
