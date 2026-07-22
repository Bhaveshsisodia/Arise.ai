const messages = document.getElementById('messages');
const input = document.getElementById('question');
const button = document.getElementById('send');
const STREAM_RENDER_INTERVAL_MS = 28;
const STREAM_RENDER_CHARS_PER_TICK = 24;
const conversationHistory = [];
const SESSION_STORAGE_KEY = 'arise_chat_session_id';

function getSessionId() {
    const existing = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (existing) {
        return existing;
    }
    const created = (window.crypto && window.crypto.randomUUID)
        ? window.crypto.randomUUID()
        : `session-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    window.sessionStorage.setItem(SESSION_STORAGE_KEY, created);
    return created;
}

function escapeHtml(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function renderMarkdown(text) {
    const escaped = escapeHtml(text);
    const lines = escaped.split(/\r?\n/);
    let html = '';
    let inList = false;
    let listType = null;

    const closeList = () => {
        if (inList) {
            html += listType === 'ol' ? '</ol>' : '</ul>';
            inList = false;
            listType = null;
        }
    };

    const renderInline = (line) => {
        return line
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code>$1</code>');
    };

    const isTableRow = (line) => /^\s*\|.*\|\s*$/.test(line);
    const isSeparatorRow = (line) => /^\s*\|[\s:-]+(\|[\s:-]+)+\|\s*$/.test(line);

    for (let i = 0; i < lines.length; i += 1) {
        const line = lines[i];
        const headingMatch = line.match(/^(#{1,3})\s+(.*)$/);
        const bulletMatch = line.match(/^\s*[-*]\s+(.*)$/);
        const orderedMatch = line.match(/^\s*(\d+)\.\s+(.*)$/);

        if (headingMatch) {
            closeList();
            const level = Math.min(3, headingMatch[1].length);
            html += `<h${level}>${renderInline(headingMatch[2])}</h${level}>`;
            continue;
        }

        if (isTableRow(line) && i + 1 < lines.length && isSeparatorRow(lines[i + 1])) {
            closeList();
            html += '<table class="markdown-table"><thead><tr>';
            const headers = line.trim().slice(1, -1).split('|').map((cell) => cell.trim());
            headers.forEach((cell) => {
                html += `<th>${renderInline(cell)}</th>`;
            });
            html += '</tr></thead><tbody>';
            i += 1;
            while (i + 1 < lines.length && isTableRow(lines[i + 1])) {
                const row = lines[i + 1].trim().slice(1, -1).split('|').map((cell) => cell.trim());
                html += '<tr>';
                row.forEach((cell) => {
                    html += `<td>${renderInline(cell)}</td>`;
                });
                html += '</tr>';
                i += 1;
            }
            html += '</tbody></table>';
            continue;
        }

        if (orderedMatch) {
            if (!inList || listType !== 'ol') {
                closeList();
                inList = true;
                listType = 'ol';
                html += '<ol>';
            }
            html += `<li>${renderInline(orderedMatch[2])}</li>`;
            continue;
        }

        if (bulletMatch) {
            if (!inList || listType !== 'ul') {
                closeList();
                inList = true;
                listType = 'ul';
                html += '<ul>';
            }
            html += `<li>${renderInline(bulletMatch[1])}</li>`;
            continue;
        }

        if (line.trim() === '') {
            closeList();
            html += '<p></p>';
            continue;
        }

        closeList();
        html += `<p>${renderInline(line)}</p>`;
    }

    closeList();
    return html || '<p></p>';
}

function formatApiError(errorPayload, fallbackMessage = 'An unexpected error occurred.') {
    if (!errorPayload) {
        return fallbackMessage;
    }

    if (typeof errorPayload === 'string') {
        return errorPayload;
    }

    const errorObject = errorPayload.error && typeof errorPayload.error === 'object'
        ? errorPayload.error
        : errorPayload;

    const message = typeof errorObject.message === 'string' && errorObject.message.trim()
        ? errorObject.message.trim()
        : fallbackMessage;

    const code = typeof errorObject.code === 'string' && errorObject.code.trim()
        ? errorObject.code.trim()
        : '';

    return code ? `${message} (${code})` : message;
}

function addMessage(role, text) {
    const el = document.createElement('div');
    el.className = 'message ' + role;

    const label = document.createElement('strong');
    label.textContent = role === 'user' ? 'You' : role === 'bot' ? 'Assistant' : 'Error';
    el.appendChild(label);

    const content = document.createElement('div');
    content.className = 'message-text';
    content.innerHTML = renderMarkdown(text);
    el.appendChild(content);

    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
}

function createStreamRenderer(content) {
    const state = {
        renderedText: '',
        pendingText: '',
        intervalId: null,
        flushAllAtEnd: false,
    };
    const cursor = document.createElement('span');
    cursor.className = 'stream-cursor';

    const paint = () => {
        content.innerHTML = renderMarkdown(state.renderedText);
        if (!state.flushAllAtEnd) {
            content.appendChild(cursor);
        }
        messages.scrollTop = messages.scrollHeight;
    };

    const render = () => {
        if (!state.pendingText) {
            if (state.flushAllAtEnd && state.intervalId !== null) {
                clearInterval(state.intervalId);
                state.intervalId = null;
            }
            return;
        }

        const chunkSize = state.flushAllAtEnd
            ? state.pendingText.length
            : Math.min(STREAM_RENDER_CHARS_PER_TICK, state.pendingText.length);

        state.renderedText += state.pendingText.slice(0, chunkSize);
        state.pendingText = state.pendingText.slice(chunkSize);
        paint();

        if (!state.pendingText && state.flushAllAtEnd && state.intervalId !== null) {
            clearInterval(state.intervalId);
            state.intervalId = null;
        }
    };

    const ensureRunning = () => {
        if (state.intervalId !== null) {
            return;
        }
        state.intervalId = window.setInterval(render, STREAM_RENDER_INTERVAL_MS);
    };

    return {
        push(text) {
            if (!text) {
                return;
            }
            state.pendingText += text;
            ensureRunning();
        },
        finish() {
            state.flushAllAtEnd = true;
            if (state.pendingText) {
                ensureRunning();
                render();
            } else {
                paint();
            }
            if (state.intervalId !== null) {
                clearInterval(state.intervalId);
                state.intervalId = null;
            }
        },
        getText() {
            return state.renderedText + state.pendingText;
        },
    };
}

function addSources(sources) {
    const container = document.createElement('div');
    container.className = 'sources';
    container.innerHTML = '<strong>Sources</strong>';
    sources.forEach(src => {
        const item = document.createElement('div');
        item.className = 'source-item';
        const sourceLabel = src.source_id ? `Source [${src.source_id}]` : 'Source';
        const title = src.document_title || 'Source Document';
        const relevance = src.relevance !== undefined && src.relevance !== null ? ` • Relevance: ${src.relevance}` : '';
        item.innerHTML = `
            <div class="meta">${sourceLabel} • ${title}${relevance}</div>
            <div class="submeta">Section: ${src.section} • Pages: ${src.pages}</div>
            <div class="snippet">${src.snippet}</div>
        `;
        container.appendChild(item);
    });
    messages.appendChild(container);
    messages.scrollTop = messages.scrollHeight;
}

async function sendQuestion() {
    const question = input.value.trim();
    if (!question) return;
    const sessionId = getSessionId();
    addMessage('user', question);
    conversationHistory.push({ role: 'user', content: question });
    input.value = '';
    button.disabled = true;

    const botMessage = document.createElement('div');
    botMessage.className = 'message bot';

    const label = document.createElement('strong');
    label.textContent = 'Assistant';
    botMessage.appendChild(label);

    const content = document.createElement('div');
    content.className = 'message-text';
    botMessage.appendChild(content);

    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.textContent = 'Typing...';
    botMessage.appendChild(typingIndicator);

    messages.appendChild(botMessage);
    messages.scrollTop = messages.scrollHeight;

    try {
        const response = await fetch('/ask/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, session_id: sessionId }),
        });

        if (!response.ok) {
            let errorMessage = 'Request failed.';
            try {
                const errorPayload = await response.json();
                errorMessage = formatApiError(errorPayload, errorMessage);
            } catch (_) {
                errorMessage = await response.text();
            }
            content.innerHTML = renderMarkdown(errorMessage);
            botMessage.className = 'message error';
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let done = false;
        let buffer = '';
        const streamRenderer = createStreamRenderer(content);

        while (!done) {
            const { value, done: doneReading } = await reader.read();
            done = doneReading;
            if (value) {
                buffer += decoder.decode(value, { stream: true });
                let newlineIndex;
                while ((newlineIndex = buffer.indexOf('\n')) >= 0) {
                    const line = buffer.slice(0, newlineIndex).trim();
                    buffer = buffer.slice(newlineIndex + 1);
                    if (!line) continue;
                    try {
                        const event = JSON.parse(line);
                        if (event.type === 'delta') {
                            streamRenderer.push(event.text);
                        } else if (event.type === 'sources') {
                            streamRenderer.finish();
                            if (Array.isArray(event.sources) && event.sources.length > 0) {
                                addSourcesToMessage(botMessage, event.sources);
                            }
                        } else if (event.type === 'error') {
                            const errorMessage = formatApiError(event.error, 'Streaming request failed.');
                            streamRenderer.push(`\n[ERROR] ${errorMessage}`);
                            streamRenderer.finish();
                            botMessage.className = 'message error';
                        }
                    } catch (err) {
                        console.error('Failed to parse stream event', err, line);
                    }
                }
            }
        }
        streamRenderer.finish();
        const finalAnswer = streamRenderer.getText().trim();
        if (finalAnswer) {
            conversationHistory.push({ role: 'assistant', content: finalAnswer });
        }
    } catch (err) {
        content.textContent = `Error: ${err.message}`;
        botMessage.className = 'message error';
    } finally {
        if (typingIndicator && typingIndicator.parentElement) {
            typingIndicator.remove();
        }
        button.disabled = false;
    }
}

function addSourcesToMessage(botMessage, sources) {
    const container = document.createElement('div');
    container.className = 'sources';
    const heading = document.createElement('div');
    heading.className = 'sources-heading';
    heading.textContent = 'Sources';
    container.appendChild(heading);

    sources.forEach(src => {
        const item = document.createElement('div');
        item.className = 'source-item';

        const topRow = document.createElement('div');
        topRow.className = 'source-top-row';
        topRow.textContent = src.source_id ? `Source [${src.source_id}] · ${src.document_title}` : src.document_title || 'Source Document';
        item.appendChild(topRow);

        const meta = document.createElement('div');
        meta.className = 'source-meta';
        const relevance = src.relevance !== undefined && src.relevance !== null ? ` · Relevance: ${src.relevance}` : '';
        meta.textContent = `Section: ${src.section} · Pages: ${src.pages}${relevance}`;
        item.appendChild(meta);

        const snippet = document.createElement('div');
        snippet.className = 'source-snippet';
        snippet.textContent = src.snippet;
        item.appendChild(snippet);

        container.appendChild(item);
    });

    botMessage.appendChild(container);
    messages.scrollTop = messages.scrollHeight;
}

button.addEventListener('click', sendQuestion);
input.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') sendQuestion();
});
