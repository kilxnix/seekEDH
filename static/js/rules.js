// rules.js - MTG Rules Search Functionality
document.addEventListener('DOMContentLoaded', function() {
    // Rules Status Check
    document.getElementById('rulesStatusBtn').addEventListener('click', function() {
        toggleLoading('rules', true);
        
        makeApiRequest(`${getApiUrl()}/api/rag/rules-search?query=status_check&limit=1`)
            .then(data => {
                displayResult('rules', data);
                toggleLoading('rules', false);
            })
            .catch(error => {
                toggleLoading('rules', false);
                alert('Error checking rules status: ' + error);
            });
    });
    
    // Rules Search Form
    document.getElementById('rulesSearchForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const query = document.getElementById('rulesQuery').value;
        const exactRule = document.getElementById('rulesExactRule').value;
        const limit = document.getElementById('rulesLimit').value;
        
        toggleLoading('rules', true);
        
        let url = `${getApiUrl()}/api/rag/rules-search?query=${encodeURIComponent(query)}&limit=${limit}`;
        
        if (exactRule) {
            url += `&rule_number=${encodeURIComponent(exactRule)}`;
        }
        
        makeApiRequest(url)
            .then(data => {
                // Format the rules results for better display
                if (data.success && data.rules) {
                    // Extract just the rules text to display in a more readable format
                    displayFormattedRules(data);
                } else {
                    displayResult('rules', data);
                }
                toggleLoading('rules', false);
            })
            .catch(error => {
                toggleLoading('rules', false);
                alert('Error searching rules: ' + error);
            });
    });
    
    // Rules Update Form
    document.getElementById('rulesUpdateForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const url = document.getElementById('rulesUrl').value;
        
        toggleLoading('rules', true);
        
        const requestData = {};
        if (url) {
            requestData.url = url;
        }
        
        makeApiRequest(`${getApiUrl()}/api/rag/rules-update`, 'POST', requestData)
            .then(data => {
                displayResult('rules', data);
                toggleLoading('rules', false);
            })
            .catch(error => {
                toggleLoading('rules', false);
                alert('Error updating rules: ' + error);
            });
    });
    
    // Helper function to format rules results
    function displayFormattedRules(data) {
        const resultsDiv = document.getElementById('rulesResult');
        
        // First create a header with info
        let html = `<div class="mb-3">
            <strong>Query:</strong> ${data.query}<br>
            <strong>Total matches:</strong> ${data.total}
        </div>`;
        
        // Create a list of rules
        html += '<div class="list-group">';
        data.rules.forEach(rule => {
            const score = rule.relevance_score ? ` (Score: ${rule.relevance_score.toFixed(2)})` : '';
            html += `<div class="list-group-item">
                <h5 class="mb-1">Rule ${rule.rule_number}${score}</h5>
                <p class="mb-1">${rule.text}</p>
                <small>Section: ${rule.section} - ${rule.title}</small>
            </div>`;
        });
        html += '</div>';
        
        resultsDiv.innerHTML = html;
        document.getElementById('rulesResultContainer').classList.remove('d-none');
    }
});