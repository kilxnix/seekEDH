// Deck Generation and Retrieval Functionality
document.addEventListener('DOMContentLoaded', function() {
    // Generate Deck
    document.getElementById('deckForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const strategy = document.getElementById('strategy').value;
        const commander = document.getElementById('commander').value;
        const bracket = document.getElementById('bracket').value;
        const maxPrice = document.getElementById('maxPrice').value;
        const landQuality = document.getElementById('landQuality').value;
        
        // Get selected generation method
        const generationMethod = document.querySelector('input[name="generationMethod"]:checked').value;
        
        toggleLoading('deck', true);
        
        // Prepare request data with correct parameter names
        const requestData = {
            strategy: strategy,
            bracket: parseInt(bracket),
            land_quality: landQuality
        };
        
        if (commander) {
            requestData.commander_name = commander;
        }
        
        if (maxPrice) {
            requestData.max_price = parseFloat(maxPrice);
        }
        
        // Determine endpoint based on generation method
        const endpoint = generationMethod === 'model' ? 
            `${getApiUrl()}/api/generate-deck-with-model` : 
            `${getApiUrl()}/api/generate-deck`;
        
        // Send API request
        makeApiRequest(endpoint, 'POST', requestData)
            .then(data => {
                // Handle redirects for model fallback
                if (!data.success && data.redirect) {
                    console.log("Model-based generation failed, falling back to RAG-based generation");
                    return makeApiRequest(`${getApiUrl()}${data.redirect}`, 'POST', requestData);
                }
                return data;
            })
            .then(data => {
                // Add commander suggestions handling
                if (!data.success && data.suggestions) {
                    let message = `Commander "${commander}" not found. Did you mean:\n`;
                    data.suggestions.forEach(name => {
                        message += `- ${name}\n`;
                    });
                    alert(message);
                } else {
                    displayResult('deck', data);
                }
                toggleLoading('deck', false);
            })
            .catch(error => {
                toggleLoading('deck', false);
                alert('Error generating deck: ' + error);
            });
    });
    
    // Get Deck (Regular Endpoint)
    document.getElementById('getDeckRegularBtn').addEventListener('click', function() {
        const deckId = document.getElementById('deckId').value;
        if (!deckId) {
            alert('Please enter a deck ID');
            return;
        }
        
        toggleLoading('getDeck', true);
        
        makeApiRequest(`${getApiUrl()}/deck/${deckId}`)
            .then(data => {
                displayResult('getDeck', data);
            })
            .catch(error => {
                toggleLoading('getDeck', false);
                alert('Error getting deck: ' + error);
            });
    });
    
    // Get Deck (API Endpoint)
    document.getElementById('getDeckApiBtn').addEventListener('click', function() {
        const deckId = document.getElementById('deckId').value;
        if (!deckId) {
            alert('Please enter a deck ID');
            return;
        }
        
        toggleLoading('getDeck', true);
        
        makeApiRequest(`${getApiUrl()}/api/deck/${deckId}`)
            .then(data => {
                displayResult('getDeck', data);
            })
            .catch(error => {
                toggleLoading('getDeck', false);
                alert('Error getting deck: ' + error);
            });
    });
});