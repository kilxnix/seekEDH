// Database Operations Functionality

document.addEventListener('DOMContentLoaded', function() {
    // Database Status
    document.getElementById('dbStatusBtn').addEventListener('click', function() {
        toggleLoading('db', true);
        
        makeApiRequest(`${getApiUrl()}/api/database/status`)
            .then(data => {
                displayResult('db', data);
            })
            .catch(error => {
                toggleLoading('db', false);
                alert('Error checking database status: ' + error);
            });
    });
    
    // Database Config
    document.getElementById('dbConfigForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const url = document.getElementById('dbUrl').value;
        const key = document.getElementById('dbKey').value;
        
        toggleLoading('db', true);
        
        makeApiRequest(`${getApiUrl()}/api/config/database`, 'POST', { url, key })
            .then(data => {
                displayResult('db', data);
            })
            .catch(error => {
                toggleLoading('db', false);
                alert('Error updating database config: ' + error);
            });
    });
    
    // Initialize Database
    document.getElementById('dbInitBtn').addEventListener('click', function() {
        toggleLoading('db', true);
        
        makeApiRequest(`${getApiUrl()}/api/database/initialize`, 'POST', {})
            .then(data => {
                displayResult('db', data);
            })
            .catch(error => {
                toggleLoading('db', false);
                alert('Error initializing database: ' + error);
            });
    });
    
    // Import Data
    document.getElementById('dbImportBtn').addEventListener('click', function() {
        toggleLoading('db', true);
        
        makeApiRequest(`${getApiUrl()}/api/database/import`, 'POST', { include_price_embeddings: true, verbose: true })
            .then(data => {
                displayResult('db', data);
            })
            .catch(error => {
                toggleLoading('db', false);
                alert('Error importing data: ' + error);
            });
    });
});