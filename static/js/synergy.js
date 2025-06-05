// synergy.js - Card Synergy Search Functionality
document.addEventListener('DOMContentLoaded', function() {
    // Synergy Search with Seed Cards
    document.getElementById('synergySeedCardsForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const seedCardsStr = document.getElementById('seedCards').value;
        if (!seedCardsStr) {
            alert('Please enter at least one card name');
            return;
        }
        
        const seedCards = seedCardsStr.split(',').map(card => card.trim()).filter(card => card);
        if (seedCards.length === 0) {
            alert('Please enter valid card names');
            return;
        }
        
        const count = Math.min(parseInt(document.getElementById('synergyCount').value || '10'), 30);
        
        toggleLoading('synergy', true);
        
        const requestData = {
            cards: seedCards,
            count: count
        };
        
        makeApiRequest(`${getApiUrl()}/api/rag/synergy-search`, 'POST', requestData)
            .then(data => {
                displayResult('synergy', data);
                toggleLoading('synergy', false);
            })
            .catch(error => {
                toggleLoading('synergy', false);
                alert('Error finding synergistic cards: ' + error);
            });
    });
    
    // Card Mechanics Analysis
    document.getElementById('mechanicsAnalysisForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const cardName = document.getElementById('cardToAnalyze').value;
        if (!cardName) {
            alert('Please enter a card name');
            return;
        }
        
        toggleLoading('synergy', true);
        
        makeApiRequest(`${getApiUrl()}/api/rag/mechanics-analysis?card=${encodeURIComponent(cardName)}`)
            .then(data => {
                displayResult('synergy', data);
                toggleLoading('synergy', false);
            })
            .catch(error => {
                toggleLoading('synergy', false);
                alert('Error analyzing card mechanics: ' + error);
            });
    });
    
    // Rules Interaction Search
    document.getElementById('rulesInteractionForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const cardsStr = document.getElementById('interactionCards').value;
        if (!cardsStr) {
            alert('Please enter at least two card names');
            return;
        }
        
        const cards = cardsStr.split(',').map(card => card.trim()).filter(card => card);
        if (cards.length < 2) {
            alert('Please enter at least two card names');
            return;
        }
        
        toggleLoading('synergy', true);
        
        const requestData = {
            cards: cards
        };
        
        makeApiRequest(`${getApiUrl()}/api/rag/rules-interaction`, 'POST', requestData)
            .then(data => {
                displayResult('synergy', data);
                toggleLoading('synergy', false);
            })
            .catch(error => {
                toggleLoading('synergy', false);
                alert('Error finding rules interactions: ' + error);
            });
    });
});