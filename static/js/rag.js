// rag.js
document.addEventListener('DOMContentLoaded', function() {
    // RAG Status
    document.getElementById('ragStatusBtn').addEventListener('click', function() {
        toggleLoading('rag', true);
        
        makeApiRequest(`${getApiUrl()}/api/rag/status`)
            .then(data => {
                displayResult('rag', data);
                toggleLoading('rag', false);
            })
            .catch(error => {
                toggleLoading('rag', false);
                alert('Error checking RAG status: ' + error);
            });
    });
    
    // RAG Deck Recommendation
    document.getElementById('ragDeckRecommendationForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const strategy = document.getElementById('deckStrategy').value;
        const commander = document.getElementById('deckCommander').value;
        const colors = getSelectedColors('deckColors');
        const colorCombo = document.getElementById('deckColorCombo').value;
        const themes = document.getElementById('deckThemes').value;
        const excludeThemes = document.getElementById('deckExclude').value;
        const budget = document.getElementById('deckBudget').value;
        const cardLimit = document.getElementById('deckCardLimit').value;
        
        toggleLoading('rag', true);
        
        const requestData = {
            strategy: strategy
        };
        
        // Add optional parameters if provided
        if (commander) {
            requestData.commander = commander;
        }
        
        if (colors && colors.length > 0) {
            requestData.colors = colors;
        }
        
        if (colorCombo) {
            requestData.color_combo = colorCombo;
        }
        
        if (themes) {
            requestData.themes = themes;
        }
        
        if (excludeThemes) {
            requestData.exclude = excludeThemes;
        }
        
        if (budget) {
            requestData.budget = parseFloat(budget);
        }
        
        if (cardLimit) {
            requestData.card_limit = parseInt(cardLimit);
        }
        
        makeApiRequest(`${getApiUrl()}/api/rag/deck-recommendation`, 'POST', requestData)
            .then(data => {
                displayResult('rag', data);
                toggleLoading('rag', false);
            })
            .catch(error => {
                toggleLoading('rag', false);
                alert('Error getting deck recommendations: ' + error);
            });
    });
    
    // RAG Keyword Search
    document.getElementById('ragKeywordForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const keyword = document.getElementById('keywordQuery').value;
        const exactPhrase = document.getElementById('keywordExactPhrase').checked;
        const exclude = document.getElementById('keywordExclude').value;
        const limit = document.getElementById('keywordLimit').value;
        const cardType = document.getElementById('keywordType').value;
        const colors = getSelectedColors('keywordColors');
        const legality = document.getElementById('keywordLegality').value;
        
        toggleLoading('rag', true);
        
        let url = `${getApiUrl()}/api/rag/search-keyword?keyword=${encodeURIComponent(keyword)}&limit=${limit}`;
        
        // Add exact phrase parameter if checked
        if (exactPhrase) {
            url += `&exact_phrase=true`;
        }
        
        // Add filter parameters
        if (colors.length > 0) {
            url += `&colors=${encodeURIComponent(colors.join(','))}`;
        }
        
        if (exclude) {
            url += `&exclude=${encodeURIComponent(exclude)}`;
        }
        
        if (cardType) {
            url += `&type=${encodeURIComponent(cardType)}`;
        }
        
        if (legality) {
            if (legality === 'banned') {
                url += '&banned=true';
            } else {
                url += `&format=${encodeURIComponent(legality)}`;
            }
        }
        
        makeApiRequest(url)
            .then(data => {
                displayResult('rag', data);
                toggleLoading('rag', false);
            })
            .catch(error => {
                toggleLoading('rag', false);
                alert('Error searching with keyword: ' + error);
            });
    });
    
    // RAG Similar Text
    document.getElementById('ragSimilarTextForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const text = document.getElementById('textQuery').value;
        const exactPhrase = document.getElementById('textExactPhrase').checked;
        const exclude = document.getElementById('textExclude').value;
        const limit = document.getElementById('textLimit').value;
        const cardType = document.getElementById('textType').value;
        const colors = getSelectedColors('textColors');
        const legality = document.getElementById('textLegality').value;
        
        toggleLoading('rag', true);
        
        let url = `${getApiUrl()}/api/rag/similar-text?query=${encodeURIComponent(text)}&limit=${limit}`;
        
        // Add exact phrase parameter if checked
        if (exactPhrase) {
            url += `&exact_phrase=true`;
        }
        
        // Add filter parameters
        if (colors.length > 0) {
            url += `&colors=${encodeURIComponent(colors.join(','))}`;
        }
        
        if (exclude) {
            url += `&exclude=${encodeURIComponent(exclude)}`;
        }
        
        if (cardType) {
            url += `&type=${encodeURIComponent(cardType)}`;
        }
        
        if (legality) {
            if (legality === 'banned') {
                url += '&banned=true';
            } else {
                url += `&format=${encodeURIComponent(legality)}`;
            }
        }
        
        makeApiRequest(url)
            .then(data => {
                displayResult('rag', data);
                toggleLoading('rag', false);
            })
            .catch(error => {
                toggleLoading('rag', false);
                alert('Error searching similar text: ' + error);
            });
    });
    
    // RAG Similar Price
    document.getElementById('ragSimilarPriceForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const card = document.getElementById('priceCardName').value;
        const exclude = document.getElementById('priceExclude').value;
        const limit = document.getElementById('priceLimit').value;
        const cardType = document.getElementById('priceType').value;
        const legality = document.getElementById('priceLegality').value;
        
        toggleLoading('rag', true);
        
        let url = `${getApiUrl()}/api/rag/similar-price?card=${encodeURIComponent(card)}&limit=${limit}`;
        
        // Add filter parameters
        if (exclude) {
            url += `&exclude=${encodeURIComponent(exclude)}`;
        }
        
        if (cardType) {
            url += `&type=${encodeURIComponent(cardType)}`;
        }
        
        if (legality) {
            if (legality === 'banned') {
                url += '&banned=true';
            } else {
                url += `&format=${encodeURIComponent(legality)}`;
            }
        }
        
        makeApiRequest(url)
            .then(data => {
                displayResult('rag', data);
                toggleLoading('rag', false);
            })
            .catch(error => {
                toggleLoading('rag', false);
                alert('Error finding similar price cards: ' + error);
            });
    });
    
    // Helper function to get selected colors
    function getSelectedColors(containerId) {
        const colors = [];
        const container = document.getElementById(containerId);
        
        if (container) {
            const checkboxes = container.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach(checkbox => {
                if (checkbox.checked) {
                    colors.push(checkbox.value);
                }
            });
        }
        
        return colors;
    }
});