// js/deck.js

document.addEventListener('DOMContentLoaded', function () {
    // --- DOM Elements ---
    const deckForm = document.getElementById('deckForm');
    const deckResultContainer = document.getElementById('deckResultContainer');
    const deckResult = document.getElementById('deckResult');
    const deckLoadingIndicator = document.getElementById('deckLoadingIndicator');
    const apiUrlInput = document.getElementById('apiUrl');

    // --- Deck Generation Form Submission ---
    deckForm.addEventListener('submit', function (e) {
        e.preventDefault();
        deckResultContainer.classList.add('d-none');
        deckLoadingIndicator.classList.remove('d-none');

        // Gather payload from form
        const apiUrl = apiUrlInput.value.replace(/\/$/, '');
        const payload = {
            strategy: document.getElementById('strategy').value,
            commander_name: document.getElementById('commander').value,
            bracket: parseInt(document.getElementById('bracket').value),
            max_price: document.getElementById('maxPrice').value || null,
            land_quality: document.getElementById('landQuality').value,
            generationMethod: document.querySelector('input[name="generationMethod"]:checked').value
        };

        // POST to API
        fetch(`${apiUrl}/api/generate-deck`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            deckLoadingIndicator.classList.add('d-none');
            deckResultContainer.classList.remove('d-none');
            renderDeckResult(data, payload);
        })
        .catch(err => {
            deckLoadingIndicator.classList.add('d-none');
            deckResultContainer.classList.remove('d-none');
            deckResult.innerHTML = `<div class="alert alert-danger">Error: ${err}</div>`;
        });
    });

    // --- Render the Deck API Response, Handle Scryfall Suggestions ---
    function renderDeckResult(data, originalPayload) {
        let html = '';

        // Scryfall Suggestion Block
        if (data.success === false && data.suggestions && data.suggestions.length > 0) {
            html += `<div class="alert alert-warning"><strong>Card not found:</strong> <b>${data.card_requested}</b><br>Choose a suggested replacement below:</div>`;
            html += `<ul class="list-group mb-3">`;
            data.suggestions.forEach(s => {
                html += `<li class="list-group-item list-group-item-action suggestion-item" style="cursor:pointer" data-name="${s.name}">${s.name} <span class="badge bg-secondary">${s.set}</span></li>`;
            });
            html += `</ul>`;
            html += `<div class="mb-3"><button class="btn btn-secondary" id="retryOriginalBtn">Retry Original</button></div>`;
        }

        // Validation Status, Errors, and Warnings
        if (data.validation_status) {
            html += `<p><strong>Validation Status:</strong> <span class="badge bg-${data.validation_status === 'valid' ? 'success' : (data.validation_status === 'partial' ? 'warning' : 'danger')}">${data.validation_status}</span></p>`;
        }
        if (data.errors && data.errors.length > 0) {
            html += `<div class="alert alert-danger"><strong>Errors:</strong><ul>${data.errors.map(e => `<li>${e.message || JSON.stringify(e)}</li>`).join('')}</ul></div>`;
        }
        if (data.warnings && data.warnings.length > 0) {
            html += `<div class="alert alert-warning"><strong>Warnings:</strong><ul>${data.warnings.map(w => `<li>${w.message || JSON.stringify(w)}</li>`).join('')}</ul></div>`;
        }

        // Deck JSON and Pretty Text
        if (data.deck_json) {
            html += `<h5>Deck JSON</h5><pre>${JSON.stringify(data.deck_json, null, 2)}</pre>`;
        }
        if (data.deck_text) {
            html += `<h5>Deck Text</h5><pre>${data.deck_text}</pre>`;
        }

        // Debug Information
        if (data.debug_info) {
            html += `<details><summary>Debug Info</summary><pre>${JSON.stringify(data.debug_info, null, 2)}</pre></details>`;
        }

        deckResult.innerHTML = html;

        // --- Interactive: Scryfall Fix Suggestions ---
        document.querySelectorAll('.suggestion-item').forEach(item => {
            item.addEventListener('click', function () {
                const newName = this.getAttribute('data-name');
                if (data.error && data.error.toLowerCase().includes('commander')) {
                    document.getElementById('commander').value = newName;
                } else if (data.error && data.error.toLowerCase().includes('card')) {
                    // For main deck cards, you would extend your API to accept a card_overrides field.
                    alert('Please fix the card in your deck generation source, or implement a card override feature.');
                }
                deckForm.dispatchEvent(new Event('submit'));
            });
        });

        // Retry Original Button
        const retryBtn = document.getElementById('retryOriginalBtn');
        if (retryBtn) {
            retryBtn.addEventListener('click', function () {
                deckForm.dispatchEvent(new Event('submit'));
            });
        }
    }
});
