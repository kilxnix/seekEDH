// js/performance-tracker.js - Image Performance Monitoring

/**
 * Performance Tracker for MTG Image System
 * Monitors load times, success rates, and network performance
 */
class PerformanceTracker {
    constructor() {
        this.reset();
        this.networkInfo = this.getNetworkInfo();
        this.setupNetworkMonitoring();
    }
    
    /**
     * Reset all tracking metrics
     */
    reset() {
        this.startTime = Date.now();
        this.metrics = {
            totalImages: 0,
            loadedImages: 0,
            failedImages: 0,
            loadTimes: [],
            errors: [],
            networkConditions: []
        };
    }
    
    /**
     * Start tracking a new session
     */
    startSession(sessionName = 'default') {
        this.currentSession = sessionName;
        this.reset();
        console.log(`Performance tracking started for session: ${sessionName}`);
    }
    
    /**
     * Increment total image count
     */
    incrementTotal() {
        this.metrics.totalImages++;
        this.updateUI();
    }
    
    /**
     * Record successful image load
     */
    incrementLoaded(loadTime = null) {
        this.metrics.loadedImages++;
        
        if (loadTime) {
            this.metrics.loadTimes.push(loadTime);
        }
        
        this.updateUI();
    }
    
    /**
     * Record failed image load
     */
    incrementFailed(error = null) {
        this.metrics.failedImages++;
        
        if (error) {
            this.metrics.errors.push({
                timestamp: Date.now(),
                error: error.toString(),
                url: error.target?.src || 'unknown'
            });
        }
        
        this.updateUI();
    }
    
    /**
     * Record network condition change
     */
    recordNetworkChange(condition) {
        this.metrics.networkConditions.push({
            timestamp: Date.now(),
            condition: condition,
            effectiveType: this.networkInfo.effectiveType,
            downlink: this.networkInfo.downlink
        });
    }
    
    /**
     * Update performance stats in UI
     */
    updateUI() {
        const loadTime = Date.now() - this.startTime;
        const successRate = this.metrics.totalImages > 0 
            ? ((this.metrics.loadedImages / this.metrics.totalImages) * 100).toFixed(1)
            : 0;
        
        // Update performance stats display
        const elements = {
            loadTime: document.getElementById('loadTime'),
            imageCount: document.getElementById('imageCount'),
            successRate: document.getElementById('successRate')
        };
        
        if (elements.loadTime) elements.loadTime.textContent = loadTime;
        if (elements.imageCount) elements.imageCount.textContent = this.metrics.totalImages;
        if (elements.successRate) elements.successRate.textContent = successRate;
        
        // Show performance stats container
        const statsContainer = document.getElementById('performanceStats');
        if (statsContainer) {
            statsContainer.classList.remove('d-none');
        }
    }
    
    /**
     * Get current performance metrics
     */
    getMetrics() {
        const totalTime = Date.now() - this.startTime;
        const avgLoadTime = this.metrics.loadTimes.length > 0
            ? this.metrics.loadTimes.reduce((a, b) => a + b, 0) / this.metrics.loadTimes.length
            : 0;
        
        return {
            session: this.currentSession || 'default',
            totalTime,
            totalImages: this.metrics.totalImages,
            loadedImages: this.metrics.loadedImages,
            failedImages: this.metrics.failedImages,
            successRate: this.metrics.totalImages > 0 
                ? (this.metrics.loadedImages / this.metrics.totalImages) * 100 
                : 0,
            avgLoadTime,
            errors: this.metrics.errors,
            networkConditions: this.metrics.networkConditions,
            networkInfo: this.networkInfo
        };
    }
    
    /**
     * Get network information
     */
    getNetworkInfo() {
        if ('connection' in navigator) {
            const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
            return {
                effectiveType: connection.effectiveType || 'unknown',
                downlink: connection.downlink || 0,
                rtt: connection.rtt || 0,
                saveData: connection.saveData || false
            };
        }
        return {
            effectiveType: 'unknown',
            downlink: 0,
            rtt: 0,
            saveData: false
        };
    }
    
    /**
     * Setup network monitoring
     */
    setupNetworkMonitoring() {
        if ('connection' in navigator) {
            const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
            
            connection.addEventListener('change', () => {
                this.networkInfo = this.getNetworkInfo();
                this.recordNetworkChange('automatic');
                console.log('Network conditions changed:', this.networkInfo);
            });
        }
    }
    
    /**
     * Simulate network conditions for testing
     */
    simulateNetworkCondition(condition) {
        const conditions = {
            'fast': { effectiveType: '4g', downlink: 10, rtt: 50 },
            'slow': { effectiveType: '3g', downlink: 0.4, rtt: 400 },
            'offline': { effectiveType: 'none', downlink: 0, rtt: 0 }
        };
        
        if (conditions[condition]) {
            this.networkInfo = { ...this.networkInfo, ...conditions[condition] };
            this.recordNetworkChange(condition);
            console.log(`Simulating ${condition} network conditions:`, this.networkInfo);
        }
    }
    
    /**
     * Run performance test
     */
    async runPerformanceTest(config = {}) {
        const {
            cardCount = 25,
            imageSize = 'normal',
            testLazyLoading = true,
            testFallbacks = false
        } = config;
        
        this.startSession('performance_test');
        
        console.log(`Starting performance test: ${cardCount} cards, ${imageSize} size`);
        
        try {
            // Fetch test data
            const response = await fetch(`${getApiUrl()}/api/rag/enhanced-search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: 'creatures',
                    image_size: imageSize,
                    include_images: true,
                    top_k: cardCount
                })
            });
            
            const data = await response.json();
            
            if (!data.success || !data.cards) {
                throw new Error('Failed to fetch test data');
            }
            
            // Create test results display
            const testResults = this.createTestResultsDisplay(config);
            
            // Monitor image loading
            const imagePromises = data.cards.map((card, index) => {
                return new Promise((resolve) => {
                    if (card.image_url) {
                        const img = new Image();
                        const startTime = Date.now();
                        
                        img.onload = () => {
                            const loadTime = Date.now() - startTime;
                            this.incrementLoaded(loadTime);
                            resolve({ success: true, loadTime, card: card.name });
                        };
                        
                        img.onerror = () => {
                            this.incrementFailed();
                            resolve({ success: false, card: card.name });
                        };
                        
                        this.incrementTotal();
                        
                        // Add delay for lazy loading test
                        if (testLazyLoading) {
                            setTimeout(() => {
                                img.src = card.image_url;
                            }, index * 50); // Stagger loads
                        } else {
                            img.src = card.image_url;
                        }
                    } else {
                        resolve({ success: false, card: card.name, reason: 'No image URL' });
                    }
                });
            });
            
            // Wait for all images to load or fail
            const results = await Promise.all(imagePromises);
            
            // Update test results
            this.updateTestResults(testResults, results);
            
            return this.getMetrics();
            
        } catch (error) {
            console.error('Performance test failed:', error);
            throw error;
        }
    }
    
    /**
     * Create test results display
     */
    createTestResultsDisplay(config) {
        const resultsContainer = document.getElementById('performanceResults');
        if (!resultsContainer) return null;
        
        const html = `
            <div class="alert alert-info">
                <h6><i class="bi bi-play-circle"></i> Running Performance Test</h6>
                <p><strong>Configuration:</strong></p>
                <ul class="mb-0">
                    <li>Cards: ${config.cardCount}</li>
                    <li>Image Size: ${config.imageSize}</li>
                    <li>Lazy Loading: ${config.testLazyLoading ? 'Yes' : 'No'}</li>
                    <li>Test Fallbacks: ${config.testFallbacks ? 'Yes' : 'No'}</li>
                </ul>
            </div>
            
            <div class="mt-3">
                <h6>Real-time Results</h6>
                <div class="progress mb-2">
                    <div class="progress-bar" role="progressbar" style="width: 0%" id="testProgress"></div>
                </div>
                <div id="testStats">
                    <small class="text-muted">Preparing test...</small>
                </div>
            </div>
            
            <div id="detailedResults" class="mt-3" style="display: none;">
                <h6>Detailed Results</h6>
                <div id="resultsList"></div>
            </div>
        `;
        
        resultsContainer.innerHTML = html;
        return resultsContainer;
    }
    
    /**
     * Update test results display
     */
    updateTestResults(container, results) {
        if (!container) return;
        
        const metrics = this.getMetrics();
        const progressBar = container.querySelector('#testProgress');
        const testStats = container.querySelector('#testStats');
        const detailedResults = container.querySelector('#detailedResults');
        const resultsList = container.querySelector('#resultsList');
        
        // Update progress bar
        if (progressBar) {
            progressBar.style.width = '100%';
            progressBar.classList.add('bg-success');
        }
        
        // Update stats
        if (testStats) {
            testStats.innerHTML = `
                <div class="row text-center">
                    <div class="col-md-3">
                        <strong>${metrics.totalImages}</strong><br>
                        <small class="text-muted">Total Images</small>
                    </div>
                    <div class="col-md-3">
                        <strong>${metrics.loadedImages}</strong><br>
                        <small class="text-success">Loaded</small>
                    </div>
                    <div class="col-md-3">
                        <strong>${metrics.failedImages}</strong><br>
                        <small class="text-danger">Failed</small>
                    </div>
                    <div class="col-md-3">
                        <strong>${metrics.successRate.toFixed(1)}%</strong><br>
                        <small class="text-muted">Success Rate</small>
                    </div>
                </div>
                <hr>
                <div class="row text-center">
                    <div class="col-md-4">
                        <strong>${metrics.totalTime}ms</strong><br>
                        <small class="text-muted">Total Time</small>
                    </div>
                    <div class="col-md-4">
                        <strong>${metrics.avgLoadTime.toFixed(0)}ms</strong><br>
                        <small class="text-muted">Avg Load Time</small>
                    </div>
                    <div class="col-md-4">
                        <strong>${this.networkInfo.effectiveType}</strong><br>
                        <small class="text-muted">Network Type</small>
                    </div>
                </div>
            `;
        }
        
        // Show detailed results
        if (detailedResults && resultsList) {
            detailedResults.style.display = 'block';
            
            const successfulLoads = results.filter(r => r.success);
            const failedLoads = results.filter(r => !r.success);
            
            let detailsHtml = '';
            
            if (successfulLoads.length > 0) {
                detailsHtml += '<h6 class="text-success">Successful Loads</h6>';
                successfulLoads.forEach(result => {
                    detailsHtml += `
                        <div class="d-flex justify-content-between align-items-center border-bottom py-1">
                            <span class="small">${result.card}</span>
                            <span class="badge bg-success">${result.loadTime}ms</span>
                        </div>
                    `;
                });
            }
            
            if (failedLoads.length > 0) {
                detailsHtml += '<h6 class="text-danger mt-3">Failed Loads</h6>';
                failedLoads.forEach(result => {
                    detailsHtml += `
                        <div class="d-flex justify-content-between align-items-center border-bottom py-1">
                            <span class="small">${result.card}</span>
                            <span class="badge bg-danger">${result.reason || 'Load failed'}</span>
                        </div>
                    `;
                });
            }
            
            resultsList.innerHTML = detailsHtml;
        }
    }
    
    /**
     * Export performance data
     */
    exportData(format = 'json') {
        const metrics = this.getMetrics();
        
        if (format === 'json') {
            return JSON.stringify(metrics, null, 2);
        } else if (format === 'csv') {
            return this.metricsToCSV(metrics);
        }
        
        return metrics;
    }
    
    /**
     * Convert metrics to CSV format
     */
    metricsToCSV(metrics) {
        const headers = ['timestamp', 'session', 'totalImages', 'loadedImages', 'failedImages', 'successRate', 'avgLoadTime', 'networkType'];
        const row = [
            new Date().toISOString(),
            metrics.session,
            metrics.totalImages,
            metrics.loadedImages,
            metrics.failedImages,
            metrics.successRate.toFixed(2),
            metrics.avgLoadTime.toFixed(0),
            metrics.networkInfo.effectiveType
        ];
        
        return headers.join(',') + '\n' + row.join(',');
    }
    
    /**
     * Get performance recommendations
     */
    getRecommendations() {
        const metrics = this.getMetrics();
        const recommendations = [];
        
        if (metrics.successRate < 90) {
            recommendations.push({
                type: 'warning',
                message: 'Low success rate detected. Check image URLs and fallback handling.'
            });
        }
        
        if (metrics.avgLoadTime > 3000) {
            recommendations.push({
                type: 'warning',
                message: 'Slow average load time. Consider using smaller image sizes or implementing progressive loading.'
            });
        }
        
        if (this.networkInfo.effectiveType === '3g' || this.networkInfo.effectiveType === 'slow-2g') {
            recommendations.push({
                type: 'info',
                message: 'Slow network detected. Consider aggressive lazy loading and smaller image sizes.'
            });
        }
        
        if (metrics.errors.length > 0) {
            recommendations.push({
                type: 'error',
                message: `${metrics.errors.length} image errors detected. Check error logs for details.`
            });
        }
        
        return recommendations;
    }
}

// Initialize global performance tracker
window.PerformanceTracker = PerformanceTracker;
window.performanceTracker = new PerformanceTracker();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PerformanceTracker;
}