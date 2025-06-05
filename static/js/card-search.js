// card-search.js - Card Search Functionality
document.addEventListener('DOMContentLoaded', function() {
    // Card Statistics
    document.getElementById('cardStatsBtn').addEventListener('click', function() {
        toggleLoading('card-search', true);
        
        makeApiRequest(`${getApiUrl()}/api/card-stats`)
            .then(data => {
                displayCardSearchResult(data, 'Database Statistics');
                toggleLoading('card-search', false);
            })
            .catch(error => {
                toggleLoading('card-search', false);
                alert('Error getting card statistics: ' + error);
            });
    });
    
    // Card Search Form
    document.getElementById('cardSearchForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const cardName = document.getElementById('cardSearchName').value;
        searchCard(cardName);
    });
    
    // Quick Test Buttons
    document.getElementById('testKozilekBtn').addEventListener('click', function() {
        searchCard('Kozilek');
    });
    
    document.getElementById('testLightningBoltBtn').addEventListener('click', function() {
        searchCard('Lightning Bolt');
    });
    
    document.getElementById('testSolRingBtn').addEventListener('click', function() {
        searchCard('Sol Ring');
    });
    
    // Search card function
    function searchCard(cardName) {
        if (!cardName) {
            alert('Please enter a card name');
            return;
        }
        
        toggleLoading('card-search', true);
        
        makeApiRequest(`${getApiUrl()}/api/search-card?name=${encodeURIComponent(cardName)}`)
            .then(data => {
                displayCardSearchResult(data, `Search Results for "${cardName}"`);
                toggleLoading('card-search', false);
            })
            .catch(error => {
                toggleLoading('card-search', false);
                alert('Error searching for card: ' + error);
            });
    }
    
    // Display card search results in a formatted way
    function displayCardSearchResult(data, title) {
        const resultContainer = document.getElementById('cardSearchResult');
        const headerElement = document.querySelector('#cardSearchResultContainer .card-header');
        
        // Update header
        if (headerElement) {
            headerElement.textContent = title;
        }
        
        let html = '';
        
        if (data.success) {
            // Display exact match if found
            if (data.exact_match) {
                html += '<div class="alert alert-success"><strong>Exact Match Found!</strong></div>';
                html += formatCardDisplay(data.exact_match);
            }
            // Display similar matches
            else if (data.similar_matches && data.similar_matches.length > 0) {
                html += '<div class="alert alert-warning"><strong>No exact match found. Similar cards:</strong></div>';
                
                data.similar_matches.forEach((card, index) => {
                    html += formatCardDisplay(card, index + 1);
                    if (index < data.similar_matches.length - 1) {
                        html += '<hr>';
                    }
                });
            }
            // No matches found
            else {
                html += '<div class="alert alert-danger"><strong>No cards found</strong></div>';
                html += `<p>No cards found matching "${data.query}". Please check the spelling or try a different search term.</p>`;
            }
            
            // Display database statistics if available
            if (data.total_cards !== undefined) {
                html += '<div class="mt-4">';
                html += '<h6>Database Statistics</h6>';
                html += `<p><strong>Total Cards:</strong> ${data.total_cards}</p>`;
                html += `<p><strong>Database Connected:</strong> ${data.database_connected ? 'Yes' : 'No'}</p>`;
                
                if (data.sample_cards && data.sample_cards.length > 0) {
                    html += '<p><strong>Sample Cards:</strong></p>';
                    html += '<ul class="list-unstyled">';
                    data.sample_cards.forEach(card => {
                        html += `<li>• ${escapeHtml(card)}</li>`;
                    });
                    html += '</ul>';
                }
                
                if (data.kozilek_variants && data.kozilek_variants.length > 0) {
                    html += '<p><strong>Kozilek Variants Found:</strong></p>';
                    html += '<ul class="list-unstyled">';
                    data.kozilek_variants.forEach(card => {
                        html += `<li>• ${escapeHtml(card)}</li>`;
                    });
                    html += '</ul>';
                }
                html += '</div>';
            }
        } else {
            html += '<div class="alert alert-danger"><strong>Error:</strong></div>';
            html += `<p>${escapeHtml(data.error || 'Unknown error occurred')}</p>`;
            
            if (data.query) {
                html += `<p><strong>Query:</strong> ${escapeHtml(data.query)}</p>`;
            }
        }
        
        resultContainer.innerHTML = html;
        document.getElementById('cardSearchResultContainer').classList.remove('d-none');
    }
    
    // Format individual card display
    function formatCardDisplay(card, index) {
        let html = '';
        
        if (index) {
            html += `<h6 class="text-primary">Match ${index}</h6>`;
        }
        
        html += '<div class="card border-light mb-3">';
        html += '<div class="card-body">';
        
        // Card name
        html += `<h5 class="card-title text-success">${escapeHtml(card.name)}</h5>`;
        
        // Card ID
        if (card.id) {
            html += `<p class="text-muted small"><strong>ID:</strong> ${escapeHtml(card.id)}</p>`;
        }
        
        // Type line
        if (card.type_line) {
            html += `<p><strong>Type:</strong> ${escapeHtml(card.type_line)}</p>`;
        }
        
        // Color identity
        if (card.color_identity) {
            if (Array.isArray(card.color_identity)) {
                const colors = card.color_identity.length > 0 ? card.color_identity.join(', ') : 'Colorless';
                html += `<p><strong>Color Identity:</strong> ${escapeHtml(colors)}</p>`;
            } else {
                html += `<p><strong>Color Identity:</strong> ${escapeHtml(card.color_identity)}</p>`;
            }
        }
        
        // Oracle text
        if (card.oracle_text) {
            html += `<p><strong>Oracle Text:</strong></p>`;
            html += `<div class="bg-light p-2 rounded" style="white-space: pre-wrap;">${escapeHtml(card.oracle_text)}</div>`;
        }
        
        html += '</div>';
        html += '</div>';
        
        return html;
    }
    
    // Helper function to escape HTML and prevent XSS
    function escapeHtml(str) {
        if (!str) return '';
        return str
            .toString()
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
    
    // Toggle loading state for card search
    function toggleLoading(section, isLoading) {
        const loadingElement = document.getElementById(`${section}LoadingIndicator`);
        const resultContainer = document.getElementById(`${section}ResultContainer`);
        
        if (loadingElement) {
            if (isLoading) {
                loadingElement.classList.remove('d-none');
                if (resultContainer) {
                    resultContainer.classList.add('d-none');
                }
            } else {
                loadingElement.classList.add('d-none');
            }
        }
    }
});