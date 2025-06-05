// enhanced_rag.js - Complete JavaScript for Enhanced RAG Test Interface

// Utility function to get API URL
function getApiUrl() {
    return document.getElementById('apiUrl').value.trim();
}

// Utility function to show/hide loading spinners
function showLoading(spinnerId) {
    document.getElementById(spinnerId).style.display = 'inline-block';
}

function hideLoading(spinnerId) {
    document.getElementById(spinnerId).style.display = 'none';
}

// Utility function to get selected colors
function getSelectedColors(containerId) {
    const colors = [];
    const container = document.getElementById(containerId);
    const checkboxes = container.querySelectorAll('input[type="checkbox"]:checked');
    
    checkboxes.forEach(checkbox => {
        colors.push(checkbox.value);
    });
    
    return colors;
}

// Handle colorless selection exclusivity
function handleColorlessSelection(colorContainerId) {
    const container = document.getElementById(colorContainerId);
    const colorlessCheckbox = container.querySelector('input[value="C"]');
    const otherCheckboxes = container.querySelectorAll('input[type="checkbox"]:not([value="C"])');
    
    if (colorlessCheckbox) {
        colorlessCheckbox.addEventListener('change', function() {
            if (this.checked) {
                // If colorless is selected, uncheck all other colors
                otherCheckboxes.forEach(cb => cb.checked = false);
            }
        });
        
        // If any other color is selected, uncheck colorless
        otherCheckboxes.forEach(cb => {
            cb.addEventListener('change', function() {
                if (this.checked) {
                    colorlessCheckbox.checked = false;
                }
            });
        });
    }
}

// Fill query examples
function fillQuery(textareaId, text) {
    document.getElementById(textareaId).value = text;
}

// Enhanced Search functionality
async function performEnhancedSearch() {
    const query = document.getElementById('searchQuery').value;
    if (!query.trim()) {
        alert('Please enter a search query');
        return;
    }
    
    showLoading('searchSpinner');
    
    const colors = getSelectedColors('searchColors');
    const filters = {};
    
    if (colors.length > 0) {
        filters.colors = colors;
    }
    
    const cardType = document.getElementById('searchType').value;
    if (cardType) {
        filters.type = cardType;
    }
    
    const maxPrice = document.getElementById('searchMaxPrice').value;
    if (maxPrice) {
        filters.max_price = parseFloat(maxPrice);
    }
    
    const requestBody = {
        query: query,
        filters: filters,
        top_k: parseInt(document.getElementById('searchLimit').value) || 20
    };
    
    try {
        const response = await fetch(`${getApiUrl()}/api/rag/enhanced-search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();
        displaySearchResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('searchResults').innerHTML = 
            `<div class="alert alert-danger">Error: ${error.message}</div>`;
    } finally {
        hideLoading('searchSpinner');
    }
}

// Display search results
function displaySearchResults(data) {
    const resultsDiv = document.getElementById('searchResults');
    
    if (!data.success) {
        resultsDiv.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
        return;
    }
    
    if (!data.cards || data.cards.length === 0) {
        resultsDiv.innerHTML = '<div class="alert alert-info">No cards found matching your criteria.</div>';
        return;
    }
    
    let html = `
        <div class="mb-3">
            <strong>Query:</strong> ${data.query}<br>
            <strong>Results:</strong> ${data.returned || data.cards.length} of ${data.total_found || data.cards.length} found<br>
    `;
    
    if (data.search_metadata) {
        html += `
            <strong>Match Types:</strong> ${data.search_metadata.match_types ? data.search_metadata.match_types.join(', ') : 'N/A'}<br>
            <strong>Avg Relevance:</strong> ${data.search_metadata.avg_relevance ? (data.search_metadata.avg_relevance * 100).toFixed(1) + '%' : 'N/A'}
        `;
    }
    
    html += '</div><hr>';
    
    data.cards.forEach((card, index) => {
        const colorIdentity = Array.isArray(card.color_identity) 
            ? (card.color_identity.length > 0 ? card.color_identity.join('') : 'Colorless')
            : (card.color_identity || 'Colorless');
        
        const price = card.prices_usd ? `$${parseFloat(card.prices_usd).toFixed(2)}` : 'N/A';
        
        html += `
            <div class="card card-result">
                <div class="card-body">
                    <h6 class="card-title">${card.name}</h6>
                    <div class="mb-2">
                        <span class="mechanic-tag">${card.type_line || 'Unknown Type'}</span>
                        <span class="mechanic-tag">${colorIdentity}</span>
                        <span class="mechanic-tag">${price}</span>
                    </div>
                    ${card.oracle_text ? `<p class="card-text small">${card.oracle_text}</p>` : ''}
                </div>
            </div>
        `;
    });
    
    resultsDiv.innerHTML = html;
}

// Combo Finder functionality
async function findCardCombos() {
    const comboCardsText = document.getElementById('comboCards').value;
    if (!comboCardsText.trim()) {
        alert('Please enter at least two combo cards');
        return;
    }
    
    showLoading('comboSpinner');
    
    const comboCards = comboCardsText.split(',').map(card => card.trim()).filter(card => card);
    if (comboCards.length < 2) {
        alert('Please enter at least two cards for combo analysis');
        hideLoading('comboSpinner');
        return;
    }
    
    const includeEnablers = document.getElementById('includeEnablers').checked;
    
    const requestBody = {
        cards: comboCards,
        max_combo_size: 3,
        include_enablers: includeEnablers
    };
    
    try {
        const response = await fetch(`${getApiUrl()}/api/rag/combo-finder`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();
        displayComboResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('comboResults').innerHTML = 
            `<div class="alert alert-danger">Error: ${error.message}</div>`;
    } finally {
        hideLoading('comboSpinner');
    }
}

// Display combo results
function displayComboResults(data) {
    const resultsDiv = document.getElementById('comboResults');
    
    if (!data.success) {
        resultsDiv.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
        return;
    }
    
    let html = `
        <div class="mb-3">
            <strong>Input Cards:</strong> ${data.input_cards ? data.input_cards.join(', ') : 'N/A'}<br>
        </div>
        <hr>
    `;
    
    // Display combo analysis for each card
    if (data.combo_analysis) {
        html += '<h6>Card Analysis</h6>';
        Object.entries(data.combo_analysis).forEach(([cardName, analysis]) => {
            if (analysis.success) {
                html += `
                    <div class="card mb-2">
                        <div class="card-body">
                            <h6>${cardName}</h6>
                            <div class="mb-2">
                                ${analysis.mechanics ? analysis.mechanics.map(m => `<span class="mechanic-tag">${m}</span>`).join(' ') : ''}
                            </div>
                            <div class="small">
                                <strong>Synergy Potential:</strong> ${analysis.synergy_potential || 'N/A'}<br>
                                <strong>Interactions:</strong> ${analysis.interactions ? analysis.interactions.length : 0}
                            </div>
                        </div>
                    </div>
                `;
            }
        });
    }
    
    // Display potential combos
    if (data.potential_combos && data.potential_combos.length > 0) {
        html += '<h6>Potential Combos</h6>';
        data.potential_combos.forEach(combo => {
            html += `
                <div class="alert alert-info">
                    <strong>${combo.type}:</strong> ${combo.description}
                </div>
            `;
        });
    }
    
    // Display synergistic cards
    if (data.synergistic_cards && data.synergistic_cards.length > 0) {
        html += '<h6>Synergistic Cards</h6>';
        data.synergistic_cards.slice(0, 10).forEach(result => {
            const card = result.card;
            const colorIdentity = Array.isArray(card.color_identity) 
                ? (card.color_identity.length > 0 ? card.color_identity.join('') : 'Colorless')
                : (card.color_identity || 'Colorless');
            
            html += `
                <div class="card card-result mb-2">
                    <div class="card-body">
                        <h6 class="card-title">${card.name}</h6>
                        <div class="mb-2">
                            <span class="synergy-score">Synergy: ${result.synergy_score ? (result.synergy_score * 100).toFixed(1) + '%' : 'N/A'}</span>
                            <span class="mechanic-tag">${colorIdentity}</span>
                        </div>
                        <p class="small text-muted">${result.synergy_reason || 'No reason provided'}</p>
                    </div>
                </div>
            `;
        });
    }
    
    // Display enabler cards
    if (data.enabler_cards && data.enabler_cards.length > 0) {
        html += '<h6>Enabler Cards</h6>';
        data.enabler_cards.forEach(enabler => {
            html += `
                <div class="alert alert-secondary">
                    <strong>${enabler.name}</strong> (${enabler.type}): ${enabler.reason}
                </div>
            `;
        });
    }
    
    resultsDiv.innerHTML = html;
}

// Card Synergy functionality
async function findCardSynergies() {
    const seedCardsText = document.getElementById('seedCards').value;
    if (!seedCardsText.trim()) {
        alert('Please enter at least one seed card');
        return;
    }
    
    showLoading('synergySpinner');
    
    const seedCards = seedCardsText.split(',').map(card => card.trim()).filter(card => card);
    const topK = parseInt(document.getElementById('synergyLimit').value) || 15;
    
    const requestBody = {
        seed_cards: seedCards,
        top_k: topK
    };
    
    try {
        const response = await fetch(`${getApiUrl()}/api/rag/card-synergies`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();
        displaySynergyResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('synergyResults').innerHTML = 
            `<div class="alert alert-danger">Error: ${error.message}</div>`;
    } finally {
        hideLoading('synergySpinner');
    }
}

// Display synergy results
function displaySynergyResults(data) {
    const resultsDiv = document.getElementById('synergyResults');
    
    if (!data.success) {
        resultsDiv.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
        return;
    }
    
    if (!data.synergistic_cards || data.synergistic_cards.length === 0) {
        resultsDiv.innerHTML = '<div class="alert alert-info">No synergistic cards found.</div>';
        return;
    }
    
    let html = `
        <div class="mb-3">
            <strong>Seed Cards:</strong> ${data.seed_cards ? data.seed_cards.join(', ') : 'N/A'}<br>
    `;
    
    if (data.seed_mechanics) {
        html += `<strong>Detected Mechanics:</strong> ${data.seed_mechanics.join(', ')}<br>`;
    }
    if (data.seed_themes) {
        html += `<strong>Detected Themes:</strong> ${data.seed_themes.join(', ')}<br>`;
    }
    if (data.total_candidates_analyzed) {
        html += `<strong>Candidates Analyzed:</strong> ${data.total_candidates_analyzed}`;
    }
    
    html += '</div><hr>';
    
    data.synergistic_cards.forEach(result => {
        const card = result.card;
        const colorIdentity = Array.isArray(card.color_identity) 
            ? (card.color_identity.length > 0 ? card.color_identity.join('') : 'Colorless')
            : (card.color_identity || 'Colorless');
        
        const synergyScore = result.synergy_score ? (result.synergy_score * 100).toFixed(1) : 'N/A';
        
        html += `
            <div class="card card-result">
                <div class="card-body">
                    <h6 class="card-title">${card.name}</h6>
                    <div class="mb-2">
                        <span class="synergy-score">Synergy: ${synergyScore}%</span>
                        <span class="mechanic-tag">${result.synergy_type || 'General'}</span>
                        <span class="mechanic-tag">${colorIdentity}</span>
                    </div>
                    <p class="small text-muted">${result.synergy_reason || 'No reason provided'}</p>
                    ${card.oracle_text ? `<p class="card-text small">${card.oracle_text}</p>` : ''}
                </div>
            </div>
        `;
    });
    
    resultsDiv.innerHTML = html;
}

// Mechanics Analysis functionality
async function analyzeCardMechanics() {
    const cardName = document.getElementById('mechanicsCard').value;
    if (!cardName.trim()) {
        alert('Please enter a card name');
        return;
    }
    
    showLoading('mechanicsSpinner');
    
    try {
        const response = await fetch(`${getApiUrl()}/api/rag/mechanics-analysis?card=${encodeURIComponent(cardName)}`);
        const data = await response.json();
        displayMechanicsResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('mechanicsResults').innerHTML = 
            `<div class="alert alert-danger">Error: ${error.message}</div>`;
    } finally {
        hideLoading('mechanicsSpinner');
    }
}

// Display mechanics analysis results
function displayMechanicsResults(data) {
    const resultsDiv = document.getElementById('mechanicsResults');
    
    if (!data.success) {
        resultsDiv.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
        return;
    }
    
    const cardName = data.card || 'Unknown Card';
    const oracleText = data.oracle_text || 'No oracle text available';
    const typeLine = data.type_line || 'Unknown type';
    const mechanics = data.mechanics || [];
    const synergyPotential = data.synergy_potential || 'Unknown';
    
    let html = `
        <div class="card mb-3">
            <div class="card-body">
                <h5 class="card-title">${cardName}</h5>
                <div class="mb-2">
                    <span class="mechanic-tag">${typeLine}</span>
                    <span class="mechanic-tag">Synergy Potential: ${synergyPotential}</span>
                </div>
                <p class="card-text">${oracleText}</p>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <h6>Mechanics</h6>
                <div class="mb-3">
                    ${mechanics.map(m => `<span class="mechanic-tag">${m}</span>`).join(' ')}
                </div>
    `;
    
    if (data.themes) {
        html += `
                <h6>Themes</h6>
                <div class="mb-3">
                    ${data.themes.map(t => `<span class="mechanic-tag">${t}</span>`).join(' ')}
                </div>
        `;
    }
    
    html += `
            </div>
            
            <div class="col-md-6">
                <h6>Interactions</h6>
    `;
    
    if (data.interactions && data.interactions.length > 0) {
        data.interactions.forEach(interaction => {
            html += `
                <div class="mb-2">
                    <strong>${interaction.type}:</strong> ${interaction.keywords ? interaction.keywords.join(', ') : 'None'}
                </div>
            `;
        });
    } else {
        html += '<p class="text-muted">No specific interactions detected</p>';
    }
    
    html += `
            </div>
        </div>
    `;
    
    // Add related cards
    if (data.related_cards_by_mechanic) {
        html += '<hr><h6>Related Cards by Mechanic</h6>';
        Object.entries(data.related_cards_by_mechanic).forEach(([mechanic, cards]) => {
            if (cards.length > 0) {
                html += `<div class="mb-2"><strong>${mechanic}:</strong> ${cards.map(c => c.name).join(', ')}</div>`;
            }
        });
    }
    
    resultsDiv.innerHTML = html;
}

// Universal Search functionality (Enhanced)
async function performUniversalSearch() {
    const query = document.getElementById('universalQuery').value;
    if (!query.trim()) {
        alert('Please enter a search query');
        return;
    }
    
    showLoading('universalSpinner');
    
    const context = {};
    const commander = document.getElementById('universalCommander').value;
    const budget = document.getElementById('universalBudget').value;
    
    if (commander) context.commander = commander;
    if (budget) context.budget = parseFloat(budget);
    
    const requestBody = {
        query: query,
        context: context
    };
    
    try {
        const response = await fetch(`${getApiUrl()}/api/rag/universal-search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();
        displayEnhancedUniversalResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('universalResults').innerHTML = 
            `<div class="alert alert-danger">Error: ${error.message}</div>`;
    } finally {
        hideLoading('universalSpinner');
    }
}

// Enhanced display for universal search results
function displayEnhancedUniversalResults(data) {
    const resultsDiv = document.getElementById('universalResults');
    
    if (!data.success) {
        resultsDiv.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
        return;
    }
    
    let html = `
        <div class="mb-3">
            <strong>Original Query:</strong> ${data.original_query}<br>
            <strong>Query Type:</strong> <span class="badge bg-primary">${data.query_type}</span><br>
    `;
    
    // Show detected cards if any
    if (data.detected_cards && data.detected_cards.length > 0) {
        html += `<strong>Detected Cards:</strong> ${data.detected_cards.join(', ')}<br>`;
    }
    
    // Show color identity if available
    if (data.color_identity && data.color_identity.length > 0) {
        const colorText = data.color_identity.join('');
        html += `<strong>Color Identity:</strong> <span class="badge bg-secondary">${colorText}</span><br>`;
    }
    
    // Show explanation
    if (data.explanation) {
        html += `<div class="alert alert-info mt-2">${data.explanation}</div>`;
    }
    
    html += '</div><hr>';
    
    // Handle card not found case
    if (data.query_type === 'card_not_found') {
        html += '<div class="alert alert-warning"><strong>Cards Not Found</strong></div>';
        
        if (data.searched_for && data.searched_for.length > 0) {
            html += `<p>Searched for: <strong>${data.searched_for.join(', ')}</strong></p>`;
        }
        
        if (data.suggestions && data.suggestions.length > 0) {
            html += '<h6>Did you mean:</h6>';
            data.suggestions.forEach(suggestion => {
                html += `
                    <div class="card card-result mb-2">
                        <div class="card-body">
                            <h6 class="card-title">${suggestion.name}</h6>
                            <div class="mb-2">
                                <span class="mechanic-tag">${suggestion.type_line || 'Unknown Type'}</span>
                            </div>
                        </div>
                    </div>
                `;
            });
        }
    }
    // Handle normal results
    else if (data.cards && data.cards.length > 0) {
        data.cards.forEach(card => {
            const colorIdentity = Array.isArray(card.color_identity) 
                ? (card.color_identity.length > 0 ? card.color_identity.join('') : 'Colorless')
                : (card.color_identity || 'Colorless');
            
            const price = card.prices_usd ? `$${parseFloat(card.prices_usd).toFixed(2)}` : 'N/A';
            
            // Get synergy score information
            const combinedScore = card.combined_synergy_score ? (card.combined_synergy_score * 100).toFixed(1) : 'N/A';
            const scoreBreakdown = card.score_breakdown || {};
            
            html += `
                <div class="card card-result mb-3">
                    <div class="card-body">
                        <h6 class="card-title">${card.name}</h6>
                        <div class="mb-2">
                            ${data.query_type === 'card_synergy_search' && combinedScore !== 'N/A' ? 
                                `<span class="synergy-score">Synergy: ${combinedScore}%</span>` : ''
                            }
                            <span class="mechanic-tag">${card.type_line || 'Unknown Type'}</span>
                            <span class="mechanic-tag">${colorIdentity}</span>
                            <span class="mechanic-tag">${price}</span>
                        </div>
                        
                        ${card.oracle_text ? `<p class="card-text small">${card.oracle_text}</p>` : ''}
                        
                        <!-- Score Breakdown -->
                        ${Object.keys(scoreBreakdown).length > 0 ? `
                            <div class="mt-2">
                                <small class="text-muted">
                                    <strong>Score Breakdown:</strong> 
                                    Text: ${(scoreBreakdown.text_similarity * 100).toFixed(0)}%, 
                                    Rules: ${(scoreBreakdown.rules_synergy * 100).toFixed(0)}%, 
                                    Mechanics: ${(scoreBreakdown.mechanic_synergy * 100).toFixed(0)}%
                                </small>
                            </div>
                        ` : ''}
                        
                        <!-- Synergy Explanations -->
                        ${card.synergy_explanations && card.synergy_explanations.length > 0 ? `
                            <div class="mt-2">
                                <small class="text-muted">
                                    <strong>Why it synergizes:</strong> ${card.synergy_explanations.join(', ')}
                                </small>
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
        });
    } else {
        html += '<div class="alert alert-info">No synergistic cards found for this query.</div>';
    }
    
    // Show not found cards if any
    if (data.not_found && data.not_found.length > 0) {
        html += `
            <div class="alert alert-warning mt-3">
                <strong>Note:</strong> Could not find: ${data.not_found.join(', ')}
            </div>
        `;
        
        if (data.suggestions && data.suggestions.length > 0) {
            html += '<h6 class="mt-3">Suggestions:</h6>';
            data.suggestions.slice(0, 3).forEach(suggestion => {
                html += `<span class="badge bg-light text-dark me-1">${suggestion.name}</span>`;
            });
        }
    }
    
    resultsDiv.innerHTML = html;
}

// Test function for enhanced universal search
function testEnhancedUniversalSearch() {
    document.getElementById('universalQuery').value = 'I need cards that synergize with Urza, Lord High Artificer';
    performUniversalSearch();
}

// Initialize event listeners when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Set up colorless selection handlers
    handleColorlessSelection('searchColors');
    
    // Enhanced Search Form
    document.getElementById('enhancedSearchForm')?.addEventListener('submit', function(e) {
        e.preventDefault();
        performEnhancedSearch();
    });
    
    // Card Synergy Form
    document.getElementById('synergyForm')?.addEventListener('submit', function(e) {
        e.preventDefault();
        findCardSynergies();
    });
    
    // Mechanics Analysis Form
    document.getElementById('mechanicsForm')?.addEventListener('submit', function(e) {
        e.preventDefault();
        analyzeCardMechanics();
    });
    
    // Combo Finder Form
    document.getElementById('comboForm')?.addEventListener('submit', function(e) {
        e.preventDefault();
        findCardCombos();
    });
    
    // Universal Search Form
    document.getElementById('universalForm')?.addEventListener('submit', function(e) {
        e.preventDefault();
        performUniversalSearch();
    });
    
    // Quick test buttons
    document.getElementById('testColorlessBtn')?.addEventListener('click', function() {
        document.getElementById('universalQuery').value = 'colorless artifacts that generate mana';
        performUniversalSearch();
    });
    
    document.getElementById('testSynergyBtn')?.addEventListener('click', function() {
        document.getElementById('seedCards').value = 'Sol Ring, Mana Vault, Grim Monolith';
        findCardSynergies();
    });
    
    document.getElementById('testMechanicsBtn')?.addEventListener('click', function() {
        document.getElementById('mechanicsCard').value = 'Lightning Bolt';
        analyzeCardMechanics();
    });
    
    // Enhanced Universal Search test button
    document.getElementById('testEnhancedUniversalBtn')?.addEventListener('click', function() {
        testEnhancedUniversalSearch();
    });
});