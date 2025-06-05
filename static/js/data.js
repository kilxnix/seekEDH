// Data Operations Functionality

document.addEventListener('DOMContentLoaded', function() {
    // Data Status
    document.getElementById('dataStatusBtn').addEventListener('click', function() {
        toggleLoading('data', true);
        
        makeApiRequest(`${getApiUrl()}/api/data/status`)
            .then(data => {
                displayResult('data', data);
            })
            .catch(error => {
                toggleLoading('data', false);
                alert('Error checking data status: ' + error);
            });
    });
    
    // Update Data
    document.getElementById('dataUpdateForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const forceUpdate = document.getElementById('forceUpdate').checked;
        const skipEmbeddings = document.getElementById('skipEmbeddings').checked;
        
        toggleLoading('data', true);
        
        makeApiRequest(`${getApiUrl()}/api/data/update`, 'POST', { force: forceUpdate, skip_embeddings: skipEmbeddings })
            .then(data => {
                displayResult('data', data);
            })
            .catch(error => {
                toggleLoading('data', false);
                alert('Error updating data: ' + error);
            });
    });
});