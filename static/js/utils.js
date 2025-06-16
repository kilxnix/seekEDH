// Utility functions for the MTG AI Endpoint Tester

// Get API URL from the input field
function getApiUrl() {
    return document.getElementById('apiUrl').value.trim();
}

// Helper function to show/hide loading indicators and result containers
function toggleLoading(section, isLoading) {
    document.getElementById(`${section}LoadingIndicator`).classList.toggle('d-none', !isLoading);
    document.getElementById(`${section}ResultContainer`).classList.toggle('d-none', isLoading);
}

// Helper function to display JSON results
function displayResult(section, data) {
    const pre = document.getElementById(`${section}Result`);
    pre.textContent = JSON.stringify(data, null, 2);

    // Image rendering support
    const containerId = `${section}Images`;
    let container = document.getElementById(containerId);
    if (!container) {
        container = document.createElement('div');
        container.id = containerId;
        container.className = 'image-grid mt-3';
        pre.parentNode.appendChild(container);
    }
    container.innerHTML = '';

    let cards = [];
    if (Array.isArray(data.cards)) {
        cards = data.cards;
    } else if (Array.isArray(data.visually_similar_cards)) {
        cards = data.visually_similar_cards.map(c => c.card || c);
    } else if (Array.isArray(data.matching_cards)) {
        cards = data.matching_cards.map(c => c.card || c);
    } else if (Array.isArray(data.synergistic_cards)) {
        cards = data.synergistic_cards.map(c => c.card || c);
    }

    if (cards.length > 0 && typeof renderMTGCards === 'function') {
        renderMTGCards(cards, `#${containerId}`, { template: 'grid' });
    }

    toggleLoading(section, false);
}
// Make API request with error handling
async function makeApiRequest(url, method = 'GET', data = null) {
    const options = {
        method: method,
        headers: {}
    };
    
    if (data) {
        options.headers['Content-Type'] = 'application/json';
        options.body = JSON.stringify(data);
    }
    
    try {
        const response = await fetch(url, options);
        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}