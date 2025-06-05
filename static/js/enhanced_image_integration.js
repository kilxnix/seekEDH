// enhanced_image_integration.js - Production-Ready Image Integration Module

class MTGImageManager {
    constructor(options = {}) {
        this.apiUrl = options.apiUrl || 'http://localhost:5000';
        this.defaultImageSize = options.defaultImageSize || 'normal';
        this.enableLazyLoading = options.enableLazyLoading !== false;
        this.enablePlaceholders = options.enablePlaceholders !== false;
        this.maxRetries = options.maxRetries || 3;
        this.retryDelay = options.retryDelay || 1000;
        
        this.imageCache = new Map();
        this.loadingPromises = new Map();
        this.performanceMetrics = {
            loadTimes: [],
            failures: [],
            cacheHits: 0
        };
        
        this.init();
    }

    init() {
        this.setupIntersectionObserver();
        this.setupGlobalErrorHandling();
        console.log('MTG Image Manager initialized');
    }

    setupIntersectionObserver() {
        if ('IntersectionObserver' in window) {
            this.imageObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        this.loadImage(img);
                        this.imageObserver.unobserve(img);
                    }
                });
            }, {
                rootMargin: '50px 0px',
                threshold: 0.1
            });
        }
    }

    setupGlobalErrorHandling() {
        window.addEventListener('error', (event) => {
            if (event.target.tagName === 'IMG') {
                this.handleImageError(event.target);
            }
        }, true);
    }

    async enhanceSearchResults(searchData, options = {}) {
        const {
            includeImages = true,
            imageSize = this.defaultImageSize,
            containerSelector = null
        } = options;

        if (!includeImages || !searchData.cards) {
            return searchData;
        }

        // Process each card to ensure proper image integration
        const enhancedCards = await Promise.all(
            searchData.cards.map(card => this.enhanceCard(card, imageSize))
        );

        const enhancedData = {
            ...searchData,
            cards: enhancedCards
        };

        // If container specified, render immediately
        if (containerSelector) {
            this.renderCardsToContainer(enhancedCards, containerSelector);
        }

        return enhancedData;
    }

    async enhanceCard(card, imageSize = this.defaultImageSize) {
        const enhancedCard = { ...card };
        
        // Ensure image metadata is properly structured
        if (!enhancedCard.image_info) {
            enhancedCard.image_info = {
                image_url: enhancedCard.image_url || null,
                image_status: enhancedCard.image_url ? 'available' : 'unavailable',
                available_sources: [],
                file_size: null
            };
        }

        // Validate and optimize image URL
        const imageUrl = this.getOptimalImageUrl(enhancedCard, imageSize);
        enhancedCard.image_url = imageUrl;
        enhancedCard.image_info.optimal_url = imageUrl;

        return enhancedCard;
    }

    getOptimalImageUrl(card, size = this.defaultImageSize) {
        const imageInfo = card.image_info || {};
        
        // Priority order: storage > local > original
        if (imageInfo.storage_url) {
            return imageInfo.storage_url;
        }
        
        if (imageInfo.local_path) {
            // Convert local path to API endpoint
            return `${this.apiUrl}/api/images/serve/${encodeURIComponent(card.name)}?size=${size}`;
        }
        
        if (imageInfo.original_url) {
            return imageInfo.original_url;
        }
        
        // Fallback to direct image_url
        return card.image_url || null;
    }

    createCardElement(card, options = {}) {
        const {
            template = 'default',
            imageSize = this.defaultImageSize,
            enableHover = true,
            showImageStatus = false,
            className = 'mtg-card'
        } = options;

        const cardElement = document.createElement('div');
        cardElement.className = `${className} card mb-3`;
        cardElement.dataset.cardName = card.name;

        const imageUrl = this.getOptimalImageUrl(card, imageSize);
        const imageStatus = card.image_info?.image_status || 'unknown';

        // Create image element with lazy loading and error handling
        const imageHtml = this.createImageHtml(card, imageUrl, imageSize, {
            enableLazyLoading: this.enableLazyLoading,
            enablePlaceholder: this.enablePlaceholders,
            showStatus: showImageStatus
        });

        // Apply template
        switch (template) {
            case 'grid':
                cardElement.innerHTML = this.getGridTemplate(card, imageHtml, imageStatus);
                break;
            case 'list':
                cardElement.innerHTML = this.getListTemplate(card, imageHtml, imageStatus);
                break;
            case 'compact':
                cardElement.innerHTML = this.getCompactTemplate(card, imageHtml, imageStatus);
                break;
            default:
                cardElement.innerHTML = this.getDefaultTemplate(card, imageHtml, imageStatus);
        }

        // Add hover effects if enabled
        if (enableHover) {
            this.addHoverEffects(cardElement);
        }

        // Setup lazy loading
        if (this.enableLazyLoading) {
            const img = cardElement.querySelector('img[data-lazy]');
            if (img && this.imageObserver) {
                this.imageObserver.observe(img);
            }
        }

        return cardElement;
    }

    createImageHtml(card, imageUrl, imageSize, options = {}) {
        const {
            enableLazyLoading = true,
            enablePlaceholder = true,
            showStatus = false
        } = options;

        if (!imageUrl) {
            return enablePlaceholder ? this.getPlaceholderHtml(card) : '';
        }

        const lazyAttrs = enableLazyLoading ? 
            'data-lazy="true" loading="lazy"' : 
            `src="${imageUrl}"`;

        const statusBadge = showStatus && card.image_info?.image_status ? 
            `<span class="image-status-badge badge bg-${this.getStatusColor(card.image_info.image_status)}">${card.image_info.image_status}</span>` : 
            '';

        return `
            <div class="image-container position-relative">
                <img ${lazyAttrs}
                     ${enableLazyLoading ? '' : `src="${imageUrl}"`}
                     data-src="${imageUrl}"
                     data-card="${card.name}"
                     data-size="${imageSize}"
                     class="card-image img-fluid"
                     alt="${card.name}"
                     onerror="mtgImageManager.handleImageError(this)"
                     onload="mtgImageManager.handleImageLoad(this)">
                ${statusBadge}
            </div>
        `;
    }

    getDefaultTemplate(card, imageHtml, imageStatus) {
        return `
            <div class="row g-0">
                <div class="col-md-3">
                    ${imageHtml}
                </div>
                <div class="col-md-9">
                    <div class="card-body">
                        <h6 class="card-title">${card.name}</h6>
                        <div class="card-meta mb-2">
                            <span class="badge bg-secondary">${card.type_line || 'Unknown Type'}</span>
                            <span class="badge bg-primary">${this.getColorIdentityText(card.color_identity)}</span>
                            ${card.prices_usd ? `<span class="badge bg-success">$${parseFloat(card.prices_usd).toFixed(2)}</span>` : ''}
                        </div>
                        ${card.oracle_text ? `<p class="card-text small">${card.oracle_text}</p>` : ''}
                    </div>
                </div>
            </div>
        `;
    }

    getGridTemplate(card, imageHtml, imageStatus) {
        return `
            <div class="card-body text-center">
                ${imageHtml}
                <h6 class="card-title mt-2">${card.name}</h6>
                <div class="card-meta">
                    <small class="text-muted">${card.type_line || 'Unknown Type'}</small>
                </div>
            </div>
        `;
    }

    getListTemplate(card, imageHtml, imageStatus) {
        return `
            <div class="d-flex align-items-center p-3">
                <div class="flex-shrink-0 me-3" style="width: 80px;">
                    ${imageHtml}
                </div>
                <div class="flex-grow-1">
                    <h6 class="mb-1">${card.name}</h6>
                    <p class="mb-1 small">${card.type_line || 'Unknown Type'}</p>
                    <small class="text-muted">${this.getColorIdentityText(card.color_identity)}</small>
                </div>
                <div class="flex-shrink-0">
                    ${card.prices_usd ? `<span class="badge bg-success">$${parseFloat(card.prices_usd).toFixed(2)}</span>` : ''}
                </div>
            </div>
        `;
    }

    getCompactTemplate(card, imageHtml, imageStatus) {
        return `
            <div class="d-flex align-items-center p-2">
                <div class="flex-shrink-0 me-2" style="width: 50px; height: 35px; overflow: hidden; border-radius: 4px;">
                    ${imageHtml}
                </div>
                <div class="flex-grow-1">
                    <div class="fw-bold small">${card.name}</div>
                    <div class="text-muted" style="font-size: 0.75rem;">${card.type_line || 'Unknown Type'}</div>
                </div>
            </div>
        `;
    }

    getPlaceholderHtml(card) {
        return `
            <div class="card-image-placeholder d-flex align-items-center justify-content-center">
                <div class="text-center text-muted">
                    <i class="fas fa-image fa-2x mb-2"></i>
                    <div class="small">No Image</div>
                </div>
            </div>
        `;
    }

    addHoverEffects(cardElement) {
        cardElement.addEventListener('mouseenter', () => {
            const img = cardElement.querySelector('.card-image');
            if (img) {
                img.style.transform = 'scale(1.05)';
                img.style.transition = 'transform 0.2s ease';
            }
        });

        cardElement.addEventListener('mouseleave', () => {
            const img = cardElement.querySelector('.card-image');
            if (img) {
                img.style.transform = 'scale(1)';
            }
        });
    }

    async loadImage(img) {
        const src = img.dataset.src;
        if (!src || img.src === src) return;

        const cardName = img.dataset.card;
        const startTime = performance.now();

        try {
            // Check cache first
            if (this.imageCache.has(src)) {
                img.src = src;
                this.performanceMetrics.cacheHits++;
                return;
            }

            // Check if already loading
            if (this.loadingPromises.has(src)) {
                await this.loadingPromises.get(src);
                img.src = src;
                return;
            }

            // Create loading promise
            const loadingPromise = this.loadImageWithRetry(src);
            this.loadingPromises.set(src, loadingPromise);

            await loadingPromise;
            img.src = src;
            this.imageCache.set(src, true);

            const loadTime = performance.now() - startTime;
            this.performanceMetrics.loadTimes.push(loadTime);

            console.log(`Image loaded: ${cardName} in ${Math.round(loadTime)}ms`);

        } catch (error) {
            console.error(`Failed to load image for ${cardName}:`, error);
            this.handleImageError(img);
            this.performanceMetrics.failures.push({ cardName, error: error.message, timestamp: Date.now() });
        } finally {
            this.loadingPromises.delete(src);
        }
    }

    async loadImageWithRetry(src, attempt = 1) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            
            img.onload = () => resolve(img);
            img.onerror = () => {
                if (attempt < this.maxRetries) {
                    setTimeout(() => {
                        this.loadImageWithRetry(src, attempt + 1).then(resolve).catch(reject);
                    }, this.retryDelay * attempt);
                } else {
                    reject(new Error(`Failed to load after ${this.maxRetries} attempts`));
                }
            };
            
            img.src = src;
        });
    }

    handleImageError(img) {
        const cardName = img.dataset.card || 'Unknown';
        const container = img.closest('.image-container') || img.parentElement;
        
        if (this.enablePlaceholders && container) {
            container.innerHTML = `
                <div class="card-image-placeholder d-flex align-items-center justify-content-center">
                    <div class="text-center text-muted">
                        <i class="fas fa-exclamation-triangle fa-lg mb-1"></i>
                        <div class="small">Image Unavailable</div>
                    </div>
                </div>
            `;
        }

        console.warn(`Image load failed for ${cardName}`);
        
        // Emit custom event for error tracking
        window.dispatchEvent(new CustomEvent('mtg-image-error', {
            detail: { cardName, imageUrl: img.src || img.dataset.src }
        }));
    }

    handleImageLoad(img) {
        const cardName = img.dataset.card || 'Unknown';
        console.log(`Image loaded successfully: ${cardName}`);
        
        // Add loaded class for CSS transitions
        img.classList.add('loaded');
        
        // Emit custom event for load tracking
        window.dispatchEvent(new CustomEvent('mtg-image-loaded', {
            detail: { cardName, imageUrl: img.src }
        }));
    }

    renderCardsToContainer(cards, containerSelector, options = {}) {
        const container = document.querySelector(containerSelector);
        if (!container) {
            console.error(`Container not found: ${containerSelector}`);
            return;
        }

        const {
            template = 'default',
            clearContainer = true,
            animate = true
        } = options;

        if (clearContainer) {
            container.innerHTML = '';
        }

        const fragment = document.createDocumentFragment();
        
        cards.forEach((card, index) => {
            const cardElement = this.createCardElement(card, { template, ...options });
            
            if (animate) {
                cardElement.style.opacity = '0';
                cardElement.style.transform = 'translateY(20px)';
                setTimeout(() => {
                    cardElement.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
                    cardElement.style.opacity = '1';
                    cardElement.style.transform = 'translateY(0)';
                }, index * 100);
            }
            
            fragment.appendChild(cardElement);
        });

        container.appendChild(fragment);
    }

    getColorIdentityText(colorIdentity) {
        if (!colorIdentity || colorIdentity.length === 0) {
            return 'Colorless';
        }
        
        if (Array.isArray(colorIdentity)) {
            return colorIdentity.join('');
        }
        
        return colorIdentity;
    }

    getStatusColor(status) {
        switch (status) {
            case 'storage': return 'success';
            case 'local': return 'info';
            case 'original': return 'warning';
            case 'error': return 'danger';
            default: return 'secondary';
        }
    }

    // Enhanced search integration
    async performEnhancedSearch(query, options = {}) {
        const {
            filters = {},
            includeImages = true,
            imageSize = this.defaultImageSize,
            topK = 20
        } = options;

        const requestBody = {
            query,
            filters,
            include_images: includeImages,
            image_size: imageSize,
            top_k: topK
        };

        try {
            const response = await fetch(`${this.apiUrl}/api/rag/enhanced-search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });

            const data = await response.json();
            return await this.enhanceSearchResults(data, { imageSize });

        } catch (error) {
            console.error('Enhanced search failed:', error);
            throw error;
        }
    }

    // Visual similarity search
    async findVisualSimilarity(cardName, options = {}) {
        const { topK = 10, includeImages = true } = options;

        try {
            const response = await fetch(
                `${this.apiUrl}/api/image-embeddings/visual-similarity?card=${encodeURIComponent(cardName)}&top_k=${topK}&include_images=${includeImages}`
            );

            const data = await response.json();
            
            if (data.success && data.visually_similar_cards) {
                return await this.enhanceSearchResults({
                    success: true,
                    cards: data.visually_similar_cards,
                    query_type: 'visual_similarity',
                    query_card: cardName
                });
            }

            return data;

        } catch (error) {
            console.error('Visual similarity search failed:', error);
            throw error;
        }
    }

    // Description-based visual search
    async searchByDescription(description, options = {}) {
        const { topK = 10, includeImages = true } = options;

        try {
            const response = await fetch(`${this.apiUrl}/api/image-embeddings/search-by-description`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    description,
                    top_k: topK,
                    include_images: includeImages
                })
            });

            const data = await response.json();
            
            if (data.success && data.matching_cards) {
                return await this.enhanceSearchResults({
                    success: true,
                    cards: data.matching_cards,
                    query_type: 'description_search',
                    description
                });
            }

            return data;

        } catch (error) {
            console.error('Description search failed:', error);
            throw error;
        }
    }

    // Performance monitoring
    getPerformanceMetrics() {
        const loadTimes = this.performanceMetrics.loadTimes;
        
        return {
            totalImages: loadTimes.length,
            averageLoadTime: loadTimes.length > 0 ? loadTimes.reduce((a, b) => a + b, 0) / loadTimes.length : 0,
            minLoadTime: loadTimes.length > 0 ? Math.min(...loadTimes) : 0,
            maxLoadTime: loadTimes.length > 0 ? Math.max(...loadTimes) : 0,
            cacheHitRate: this.performanceMetrics.cacheHits / (loadTimes.length + this.performanceMetrics.cacheHits) * 100,
            failureRate: this.performanceMetrics.failures.length / (loadTimes.length + this.performanceMetrics.failures.length) * 100,
            failures: this.performanceMetrics.failures
        };
    }

    // Clear cache and reset metrics
    clearCache() {
        this.imageCache.clear();
        this.loadingPromises.clear();
        this.performanceMetrics = {
            loadTimes: [],
            failures: [],
            cacheHits: 0
        };
        console.log('Image cache and metrics cleared');
    }

    // Preload images for better performance
    async preloadImages(cards, imageSize = this.defaultImageSize) {
        const preloadPromises = cards.map(card => {
            const imageUrl = this.getOptimalImageUrl(card, imageSize);
            if (imageUrl && !this.imageCache.has(imageUrl)) {
                return this.loadImageWithRetry(imageUrl).then(() => {
                    this.imageCache.set(imageUrl, true);
                }).catch(error => {
                    console.warn(`Preload failed for ${card.name}:`, error);
                });
            }
            return Promise.resolve();
        });

        await Promise.allSettled(preloadPromises);
        console.log(`Preloaded ${preloadPromises.length} images`);
    }

    // Responsive image size selection
    getResponsiveImageSize() {
        const width = window.innerWidth;
        
        if (width < 576) return 'small';      // Mobile
        if (width < 768) return 'normal';     // Tablet
        if (width < 1200) return 'normal';    // Desktop
        return 'large';                       // Large desktop
    }

    // Update image sizes based on viewport
    updateImageSizes() {
        const newSize = this.getResponsiveImageSize();
        const images = document.querySelectorAll('img[data-card]');
        
        images.forEach(img => {
            if (img.dataset.size !== newSize) {
                const cardName = img.dataset.card;
                // Find card data and update image
                const cardElement = img.closest('[data-card-name]');
                if (cardElement) {
                    // Re-render with new size
                    img.dataset.size = newSize;
                    // Update src if not lazy loading
                    if (!img.dataset.lazy) {
                        const newSrc = img.dataset.src.replace(/size=[^&]*/, `size=${newSize}`);
                        img.src = newSrc;
                        img.dataset.src = newSrc;
                    }
                }
            }
        });
    }

    // Initialize responsive behavior
    initResponsive() {
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                this.updateImageSizes();
            }, 250);
        });
    }

    // Accessibility improvements
    enhanceAccessibility() {
        // Add ARIA labels and roles
        document.addEventListener('DOMContentLoaded', () => {
            const cardElements = document.querySelectorAll('.mtg-card');
            cardElements.forEach(card => {
                card.setAttribute('role', 'article');
                card.setAttribute('aria-label', `Magic card: ${card.dataset.cardName}`);
                
                const img = card.querySelector('.card-image');
                if (img) {
                    img.setAttribute('role', 'img');
                    img.setAttribute('aria-describedby', `${card.dataset.cardName}-description`);
                }
            });
        });
    }
}

// CSS Styles for enhanced image integration
const CSS_STYLES = `
<style>
.card-image {
    width: 100%;
    height: 200px;
    object-fit: cover;
    border-radius: 8px;
    transition: transform 0.2s ease, opacity 0.3s ease;
    background-color: #f8f9fa;
}

.card-image.loaded {
    opacity: 1;
}

.card-image:not(.loaded) {
    opacity: 0.7;
}

.card-image-placeholder {
    width: 100%;
    height: 200px;
    border: 2px dashed #dee2e6;
    border-radius: 8px;
    background-color: #f8f9fa;
    color: #6c757d;
    display: flex;
    align-items: center;
    justify-content: center;
}

.image-container {
    position: relative;
    overflow: hidden;
    border-radius: 8px;
}

.image-status-badge {
    position: absolute;
    top: 5px;
    right: 5px;
    font-size: 0.6rem;
    z-index: 10;
}

.mtg-card {
    transition: box-shadow 0.2s ease;
    border: 1px solid #dee2e6;
}

.mtg-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.card-meta .badge {
    margin-right: 0.25rem;
    font-size: 0.7rem;
}

/* Responsive grid layouts */
.mtg-cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

@media (max-width: 768px) {
    .mtg-cards-grid {
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    }
    
    .card-image, .card-image-placeholder {
        height: 150px;
    }
}

@media (max-width: 480px) {
    .mtg-cards-grid {
        grid-template-columns: 1fr;
    }
}

/* Loading animation */
.card-image[data-lazy]:not([src]) {
    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
    background-size: 200% 100%;
    animation: loading-shimmer 1.5s infinite;
}

@keyframes loading-shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* Accessibility improvements */
.card-image:focus {
    outline: 2px solid #007bff;
    outline-offset: 2px;
}

@media (prefers-reduced-motion: reduce) {
    .card-image,
    .mtg-card {
        transition: none;
    }
    
    .card-image[data-lazy]:not([src]) {
        animation: none;
        background: #f0f0f0;
    }
}
</style>
`;

// Auto-initialization and global instance
let mtgImageManager;

document.addEventListener('DOMContentLoaded', function() {
    // Initialize with default configuration
    mtgImageManager = new MTGImageManager({
        apiUrl: window.MTG_API_URL || 'http://localhost:5000',
        enableLazyLoading: true,
        enablePlaceholders: true
    });
    
    // Add CSS styles to head
    if (!document.querySelector('#mtg-image-styles')) {
        const styleElement = document.createElement('style');
        styleElement.id = 'mtg-image-styles';
        styleElement.innerHTML = CSS_STYLES.replace(/<\/?style>/g, '');
        document.head.appendChild(styleElement);
    }
    
    // Initialize responsive behavior
    mtgImageManager.initResponsive();
    mtgImageManager.enhanceAccessibility();
    
    console.log('MTG Image Manager initialized and ready');
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MTGImageManager;
} else if (typeof window !== 'undefined') {
    window.MTGImageManager = MTGImageManager;
}

// Integration helpers for existing enhanced_rag.js
window.enhanceSearchResultsWithImages = async function(data, containerSelector) {
    if (mtgImageManager && data.success && data.cards) {
        const enhancedData = await mtgImageManager.enhanceSearchResults(data);
        
        if (containerSelector) {
            mtgImageManager.renderCardsToContainer(
                enhancedData.cards, 
                containerSelector,
                { template: 'default', animate: true }
            );
        }
        
        return enhancedData;
    }
    return data;
};

window.renderMTGCards = function(cards, containerSelector, options = {}) {
    if (mtgImageManager) {
        mtgImageManager.renderCardsToContainer(cards, containerSelector, options);
    }
};

window.getMTGImageManager = function() {
    return mtgImageManager;
};