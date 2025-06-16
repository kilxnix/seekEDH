// API Status Functionality

document.addEventListener('DOMContentLoaded', function() {
    // Check API Health
    document.getElementById('checkHealthBtn').addEventListener('click', function() {
        toggleLoading('status', true);
        
        makeApiRequest(`${getApiUrl()}/api/health`)
            .then(data => {
                displayResult('status', data);
            })
            .catch(error => {
                toggleLoading('status', false);
                alert('Error checking health: ' + error);
            });
    });
});