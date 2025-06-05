// js/visual-search.js - Visual Search functionality for MTG Cards

/**
 * Visual Search System for MTG Cards
 * Handles visual similarity search and description-based image search
 */
class VisualSearchSystem {
    constructor() {
        this.imageSystem = window.imageSystem;
        this.layoutManager = window.layoutManager;
        this.performanceTracker = window.performanceTracker;
        this.init();
    }
    
    /**
     * Initialize visual search system
     */
    init() {
        this.setupEventListeners();
    }
    
    /**
     * Setup event listeners for visual search forms
     */
    setupEventListeners() {
        // Visual similarity form
        const visualSimilarityForm = document.getElementById('visualSimilarityForm');
        if (visualSimilarityForm) {
            visualSimilarityForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.performVisualSimilaritySearch();
            });
        }
        
        // Visual description form
        const visualDescriptionForm = document.getElementById('visualDescriptionForm');
        if (visualDescriptionForm) {
            visualDescriptionForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.performVisualDescriptionSearch();
            });
        }
    }
    
    /**
     * Perform visual similarity search
     */
    async performVisualSimilaritySearch() {
        const cardName = document.getElementById('visualQueryCard')?.value;
        if (!cardName?.trim()) {
            this.showError('Please enter a card name');
            return;
        }
        
        this.showLoading('visualSpinner');
        
        if (this.performanceTracker) {
            this.performanceTracker.startSession('visual_similarity');
        }
        
        try {
            const response = await fetch(`${getApiUrl()}/api/image-embeddings/visual-similarity?card=${encodeURIComponent(cardName)}&top_k=${document.getElementById('visualSimilarityLimit')?.value || 10}&include_images=true`);
            const data = await response.json();
            
            if (data.success) {
                this.displayVisualResults(data, 'similarity');
            } else {
                this.showError(data.error || 'Visual similarity search failed');
            }
            
        } catch (error) {
            console.error('Visual similarity search error:', error);
            this.showError(`Search failed: ${error.message}`);
        } finally {
            this.hideLoading('visualSpinner');
        }
    }
    
    /**
     * Perform visual description search
     */
    async performVisualDescriptionSearch() {
        const description = document.getElementById('visualDescription')?.value;
        if (!description?.trim()) {
            this.showError('Please enter a visual description');
            return;
        }
        
        this.showLoading('visualSpinner');
        
        if (this.performanceTracker) {
            this.performanceTracker.startSession('visual_description');
        }
        
        try {
            const response = await fetch(`${getApiUrl()}/api/image-embeddings/search-by-description`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    description: description,
                    top_k: parseInt(document.getElementById('descriptionLimit')?.value) || 10,
                    include_images: true
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.displayVisualResults(data, 'description');
            } else {
                this.showError(data.error || 'Visual description search failed');
            }
            
        } catch (error) {
            console.error('Visual description search error:', error);
            this.showError(`Search failed: ${error.message}`);
        } finally {
            this.hideLoading('visualSpinner');
        }
    }
    
    /**
     * Display visual search results
     */
    displayVisualResults(data, searchType) {
        const resultsDiv = document.getElementById('visualResults');
        if (!resultsDiv) return;
        
        const cards = data.visually_similar_cards || data.matching_cards || data.cards || [];
        
        if (cards.length === 0) {
            resultsDiv.innerHTML = '<div class="alert alert-info">No visually similar cards found.</div>';
            return;
        }
        
        // Create header with search info
        let html = this.createResultsHeader(data, searchType, cards.length);
        
        // Use current layout manager settings
        const layout = this.layoutManager?.currentLayout || 'grid';
        
        switch (layout) {
            case 'grid':
                html += this.renderVisualGridLayout(cards, searchType);
                break;
            case 'list':
                html += this.renderVisualListLayout(cards, searchType);
                break;
            case 'compact':
                html += this.renderVisualCompactLayout(cards, searchType);
                break;
        }
        
        resultsDiv.innerHTML = html;
        
        // Initialize images after DOM update
        this.initializeVisualImages(cards, layout);
    }
    
    /**
     * Create results header
     */
    createResultsHeader(data, searchType, cardCount) {
        let headerText = '';
        
        if (searchType === 'similarity') {
            headerText = `Visual similarity to: <strong>${data.query_card || 'Unknown'}</strong>`;
        } else if (searchType === 'description') {
            headerText = `Cards matching: <strong>"${data.description || 'Unknown description'}"</strong>`;
        }
        
        return `
            <div class="mb-3">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        ${headerText}<br>
                        <small class="text-muted">Found ${cardCount} visually similar cards</small>
                    </div>
                    <div class="btn-group btn-group-sm" role="group">
                        <button type="button" class="btn btn-outline-secondary" onclick="window.visualSearch.sortResults('similarity')">
                            <i class="bi bi-sort-numeric-down"></i> Similarity
                        </button>
                        <button type="button" class="btn btn-outline-secondary" onclick="window.visualSearch.sortResults('name')">
                            <i class="bi bi-sort-alpha-down"></i> Name
                        </button>
                    </div>
                </div>
            </div>
            <hr>
        `;
    }
    
    /**
     * Render visual grid layout
     */
    renderVisualGridLayout(cards, searchType) {
        let html = '<div class="image-grid">';
        
        cards.forEach((card, index) => {
            const similarity = this.getSimilarityScore(card);
            const confidenceClass = this.getConfidenceClass(similarity);
            
            html += `
                <div class="image-grid-item" onclick="showImageModal('${card.name}', ${JSON.stringify(card).replace(/"/g, '&quot;')})">
                    <div class="position-relative">
                        <div id="visual-image-${this.sanitizeId(card.name)}" class="loading-skeleton">
                            <!-- Image will be loaded here -->
                        </div>
                        <div class="position-absolute top-0 end-0 m-2">
                            <span class="badge ${confidenceClass}">${similarity}%</span>
                        </div>
                    </div>
                    <div class="p-3">
                        <h6 class="card-title mb-2">${card.name}</h6>
                        <div class="mb-2">
                            <span class="mechanic-tag">${card.type_line || 'Unknown Type'}</span>
                            ${this.getPriceTag(card)}
                        </div>
                        <div class="progress mb-2" style="height: 4px;">
                            <div class="progress-bar ${confidenceClass.replace('bg-', 'bg-')}" 
                                 style="width: ${similarity}%" 
                                 title="Visual similarity: ${similarity}%"></div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        return html;
    }
    
    /**
     * Render visual list layout
     */
    renderVisualListLayout(cards, searchType) {
        let html = '';
        
        cards.forEach(card => {
            const similarity = this.getSimilarityScore(card);
            const confidenceClass = this.getConfidenceClass(similarity);
            
            html += `
                <div class="card card-result mb-3">
                    <div class="card-body">
                        <div class="card-result-list" onclick="showImageModal('${card.name}', ${JSON.stringify(card).replace(/"/g, '&quot;')})">
                            <div id="visual-image-${this.sanitizeId(card.name)}" class="loading-skeleton">
                                <!-- Image will be loaded here -->
                            </div>
                            <div class="flex-grow-1">
                                <div class="d-flex justify-content-between align-items-start mb-2">
                                    <h6 class="card-title mb-0">${card.name}</h6>
                                    <span class="badge ${confidenceClass}">${similarity}% similar</span>
                                </div>
                                <div class="mb-2">
                                    <span class="mechanic-tag">${card.type_line || 'Unknown Type'}</span>
                                    ${this.getPriceTag(card)}
                                </div>
                                ${card.oracle_text ? `<p class="card-text small mb-2">${card.oracle_text}</p>` : ''}
                                <div class="progress" style="height: 6px;">
                                    <div class="progress-bar ${confidenceClass.replace('bg-', 'bg-')}" 
                                         style="width: ${similarity}%" 
                                         title="Visual similarity: ${similarity}%"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        return html;
    }
    
    /**
     * Render visual compact layout
     */
    renderVisualCompactLayout(cards, searchType) {
        let html = '<div class="row g-2">';
        
        cards.forEach(card => {
            const similarity = this.getSimilarityScore(card);
            const confidenceClass = this.getConfidenceClass(similarity);
            
            html += `
                <div class="col-md-6">
                    <div class="card card-result">
                        <div class="card-body p-2">
                            <div class="d-flex align-items-center" onclick="showImageModal('${card.name}', ${JSON.stringify(card).replace(/"/g, '&quot;')})">
                                <div id="visual-image-${this.sanitizeId(card.name)}" class="loading-skeleton me-2">
                                    <!-- Image will be loaded here -->
                                </div>
                                <div class="flex-grow-1">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <h6 class="mb-1">${card.name}</h6>
                                        <span class="badge ${confidenceClass} badge-sm">${similarity}%</span>
                                    </div>
                                    <small class="text-muted">${card.type_line || 'Unknown Type'}</small>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        return html;
    }
    
    /**
     * Initialize images for visual search results
     */
    initializeVisualImages(cards, layout) {
        setTimeout(() => {
            cards.forEach(card => {
                const containerId = `visual-image-${this.sanitizeId(card.name)}`;
                const container = document.getElementById(containerId);
                
                if (container && this.imageSystem) {
                    const imageElement = this.imageSystem.createImageElement(card, 'normal', layout);
                    container.innerHTML = '';
                    container.appendChild(imageElement);
                    container.classList.remove('loading-skeleton');
                }
            });
        }, 50);
    }
    
    /**
     * Get similarity score from card data
     */
    getSimilarityScore(card) {
        // Try different possible similarity score fields
        if (card.visual_similarity_score) {
            return (card.visual_similarity_score * 100).toFixed(0);
        }
        if (card.similarity_score) {
            return (card.similarity_score * 100).toFixed(0);
        }
        if (card.similarity) {
            return (card.similarity * 100).toFixed(0);
        }
        if (card.score) {
            return (card.score * 100).toFixed(0);
        }
        
        // Default similarity
        return '85';
    }
    
    /**
     * Get confidence class based on similarity score
     */
    getConfidenceClass(similarity) {
        const score = parseInt(similarity);
        
        if (score >= 90) {
            return 'bg-success';
        } else if (score >= 75) {
            return 'bg-info';
        } else if (score >= 60) {
            return 'bg-warning';
        } else {
            return 'bg-secondary';
        }
    }
    
    /**
     * Get price tag HTML
     */
    getPriceTag(card) {
        const price = card.prices_usd || card.price_usd;
        if (price) {
            return `<span class="mechanic-tag">${parseFloat(price).toFixed(2)}</span>`;
        }
        return '<span class="mechanic-tag">N/A</span>';
    }
    
    /**
     * Sort visual search results
     */
    sortResults(sortBy) {
        const resultsDiv = document.getElementById('visualResults');
        if (!resultsDiv) return;
        
        const cards = this.extractCardsFromResults(resultsDiv);
        
        switch (sortBy) {
            case 'similarity':
                cards.sort((a, b) => {
                    const simA = parseInt(this.getSimilarityScore(a));
                    const simB = parseInt(this.getSimilarityScore(b));
                    return simB - simA; // Descending
                });
                break;
            case 'name':
                cards.sort((a, b) => a.name.localeCompare(b.name));
                break;
            case 'price':
                cards.sort((a, b) => {
                    const priceA = parseFloat(a.prices_usd || a.price_usd || 0);
                    const priceB = parseFloat(b.prices_usd || b.price_usd || 0);
                    return priceA - priceB; // Ascending
                });
                break;
        }
        
        // Re-render with sorted cards
        this.displayVisualResults({ cards }, 'sorted');
    }
    
    /**
     * Extract card data from current results
     */
    extractCardsFromResults(container) {
        const cards = [];
        
        container.querySelectorAll('[onclick*="showImageModal"]').forEach(element => {
            try {
                const onclickAttr = element.getAttribute('onclick');
                const match = onclickAttr.match(/showImageModal\('([^']+)', (.+)\)/);
                if (match) {
                    const cardData = JSON.parse(match[2].replace(/&quot;/g, '"'));
                    cards.push(cardData);
                }
            } catch (e) {
                console.warn('Could not extract card data from element:', e);
            }
        });
        
        return cards;
    }
    
    /**
     * Show loading spinner
     */
    showLoading(spinnerId) {
        const spinner = document.getElementById(spinnerId);
        if (spinner) {
            spinner.style.display = 'inline-block';
        }
    }
    
    /**
     * Hide loading spinner
     */
    hideLoading(spinnerId) {
        const spinner = document.getElementById(spinnerId);
        if (spinner) {
            spinner.style.display = 'none';
        }
    }
    
    /**
     * Show error message
     */
    showError(message) {
        const resultsDiv = document.getElementById('visualResults');
        if (resultsDiv) {
            resultsDiv.innerHTML = `
                <div class="alert alert-danger">
                    <i class="bi bi-exclamation-triangle"></i>
                    <strong>Error:</strong> ${message}
                </div>
            `;
        } else {
            alert(`Error: ${message}`);
        }
    }
    
    /**
     * Sanitize string for use as DOM ID
     */
    sanitizeId(str) {
        return str.replace(/[^a-zA-Z0-9]/g, '');
    }
    
    /**
     * Get visual search statistics
     */
    getStats() {
        return {
            lastSearchType: this.lastSearchType || 'none',
            lastResultCount: this.lastResultCount || 0,
            performanceMetrics: this.performanceTracker?.getMetrics() || null
        };
    }
    
    /**
     * Clear visual search results
     */
    clearResults() {
        const resultsDiv = document.getElementById('visualResults');
        if (resultsDiv) {
            resultsDiv.innerHTML = `
                <p class="text-muted text-center">
                    <i class="bi bi-eye"></i><br>
                    Use visual similarity search or describe what you're looking for...
                </p>
            `;
        }
    }
}

// Initialize global visual search system
window.VisualSearchSystem = VisualSearchSystem;
window.visualSearch = new VisualSearchSystem();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VisualSearchSystem;
}