// Price Embeddings Functionality

document.addEventListener('DOMContentLoaded', function() {
    // Price Embeddings Status
    document.getElementById('priceStatusBtn').addEventListener('click', function() {
        toggleLoading('price', true);
        
        makeApiRequest(`${getApiUrl()}/api/price-embeddings/status`)
            .then(data => {
                displayResult('price', data);
            })
            .catch(error => {
                toggleLoading('price', false);
                alert('Error checking price embeddings status: ' + error);
            });
    });
    
    // Generate Price Embeddings
    document.getElementById('priceGenerateBtn').addEventListener('click', function() {
        toggleLoading('price', true);
        
        makeApiRequest(`${getApiUrl()}/api/price-embeddings/generate`, 'POST', {})
            .then(data => {
                displayResult('price', data);
            })
            .catch(error => {
                toggleLoading('price', false);
                alert('Error generating price embeddings: ' + error);
            });
    });
    
    // Find Similar Cards by Price
    document.getElementById('priceSimilarForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const cardName = document.getElementById('similarCardName').value;
        const limit = document.getElementById('similarCardLimit').value;
        
        toggleLoading('price', true);
        
        makeApiRequest(`${getApiUrl()}/api/price-embeddings/similar?card=${encodeURIComponent(cardName)}&top_n=${limit}`)
            .then(data => {
                displayResult('price', data);
            })
            .catch(error => {
                toggleLoading('price', false);
                alert('Error finding similar cards: ' + error);
            });
    });
});