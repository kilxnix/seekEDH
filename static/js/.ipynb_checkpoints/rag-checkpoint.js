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
        toggleLoading('rag', true);
        
        const requestData = {
            strategy: strategy
        };
        
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
        const limit = document.getElementById('keywordLimit').value;
        const colors = getSelectedColors('keywordColors');
        
        toggleLoading('rag', true);
        
        let url = `${getApiUrl()}/api/rag/search-keyword?keyword=${encodeURIComponent(keyword)}&limit=${limit}`;
        
        // Add color filter if colors are selected
        if (colors.length > 0) {
            url += `&colors=${encodeURIComponent(colors.join(','))}`;
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
        const limit = document.getElementById('textLimit').value;
        const colors = getSelectedColors('textColors');
        
        toggleLoading('rag', true);
        
        let url = `${getApiUrl()}/api/rag/similar-text?query=${encodeURIComponent(text)}&limit=${limit}`;
        
        // Add color filter if colors are selected
        if (colors.length > 0) {
            url += `&colors=${encodeURIComponent(colors.join(','))}`;
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
        const limit = document.getElementById('priceLimit').value;
        toggleLoading('rag', true);
        
        makeApiRequest(`${getApiUrl()}/api/rag/similar-price?card=${encodeURIComponent(card)}&limit=${limit}`)
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