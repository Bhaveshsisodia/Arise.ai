const messages = document.getElementById('messages');
const input = document.getElementById('question');
const button = document.getElementById('send');

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
    addMessage('user', question);
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
            body: JSON.stringify({ question }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            content.textContent = errorText;
            botMessage.className = 'message error';
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let done = false;
        let buffer = '';
        botMessage.textBuffer = '';

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
                            botMessage.textBuffer += event.text;
                            content.innerHTML = renderMarkdown(botMessage.textBuffer);
                            messages.scrollTop = messages.scrollHeight;
                        } else if (event.type === 'sources') {
                            if (Array.isArray(event.sources) && event.sources.length > 0) {
                                addSourcesToMessage(botMessage, event.sources);
                            }
                        } else if (event.type === 'error') {
                            botMessage.textBuffer += `\n[ERROR] ${event.error}`;
                            content.innerHTML = renderMarkdown(botMessage.textBuffer);
                            botMessage.className = 'message error';
                        }
                    } catch (err) {
                        console.error('Failed to parse stream event', err, line);
                    }
                }
            }
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
