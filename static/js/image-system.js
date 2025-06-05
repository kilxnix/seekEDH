// js/image-system.js - Core MTG Card Image System

/**
 * MTG Card Image System
 * Handles image loading, fallbacks, lazy loading, and error handling
 */
class MTGImageSystem {
    constructor() {
        this.performanceTracker = window.PerformanceTracker || null;
        this.lazyLoadObserver = null;
        this.imageCache = new Map();
        this.errorRetryAttempts = new Map();
        this.maxRetryAttempts = 3;
        
        this.initializeLazyLoading();
        this.setupErrorHandling();
    }
    
    /**
     * Initialize lazy loading observer
     */
    initializeLazyLoading() {
        if ('IntersectionObserver' in window) {
            this.lazyLoadObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        this.loadImage(entry.target);
                        this.lazyLoadObserver.unobserve(entry.target);
                    }
                });
            }, {
                rootMargin: '50px 0px',
                threshold: 0.1
            });
        }
    }
    
    /**
     * Set up global error handling for images
     */
    setupErrorHandling() {
        window.addEventListener('error', (event) => {
            if (event.target && event.target.tagName === 'IMG') {
                this.handleImageError(event.target);
            }
        }, true);
    }
    
    /**
     * Create an image element with proper loading and error handling
     * @param {Object} card - Card data object
     * @param {string} size - Image size preference
     * @param {string} layout - Layout type (grid, list, compact)
     * @returns {HTMLElement} - Image container element
     */
    createImageElement(card, size = 'normal', layout = 'grid') {
        const container = document.createElement('div');
        container.className = 'image-container';
        container.style.position = 'relative';
        
        const imageUrl = this.getImageUrl(card);
        const imageStatus = this.getImageStatus(card);
        
        if (this.performanceTracker) {
            this.performanceTracker.incrementTotal();
        }
        
        if (imageUrl) {
            const img = this.createImg(card, imageUrl, size, layout);
            const statusIndicator = this.createStatusIndicator(imageStatus);
            
            container.appendChild(img);
            container.appendChild(statusIndicator);
            
            // Add to lazy loading if supported
            if (this.lazyLoadObserver) {
                img.dataset.src = imageUrl;
                img.src = this.getPlaceholderDataUrl();
                this.lazyLoadObserver.observe(img);
            } else {
                img.src = imageUrl;
            }
        } else {
            const placeholder = this.createPlaceholder(card.name);
            container.appendChild(placeholder);
        }
        
        return container;
    }
    
    /**
     * Create img element with proper attributes
     */
    createImg(card, imageUrl, size, layout) {
        const img = document.createElement('img');
        img.alt = card.name || 'MTG Card';
        img.className = this.getImageClass(layout);
        img.loading = 'lazy';
        img.dataset.cardName = card.name;
        img.dataset.imageStatus = this.getImageStatus(card);
        
        // Add click handler for modal
        img.addEventListener('click', () => {
            if (window.showImageModal) {
                window.showImageModal(card.name, card);
            }
        });
        
        // Add load handler
        img.addEventListener('load', () => {
            this.handleImageLoad(img);
        });
        
        // Add error handler
        img.addEventListener('error', () => {
            this.handleImageError(img);
        });
        
        return img;
    }
    
    /**
     * Create status indicator element
     */
    createStatusIndicator(status) {
        const indicator = document.createElement('div');
        indicator.className = `card-image-status status-${status}`;
        indicator.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        indicator.title = `Image source: ${status}`;
        return indicator;
    }
    
    /**
     * Create placeholder element
     */
    createPlaceholder(cardName) {
        const placeholder = document.createElement('div');
        placeholder.className = 'card-image-placeholder';
        placeholder.innerHTML = `
            <div>
                <i class="bi bi-image" style="font-size: 2rem;"></i><br>
                <span>Image not available</span>
                ${cardName ? `<br><small>${cardName}</small>` : ''}
            </div>
        `;
        return placeholder;
    }
    
    /**
     * Get image URL from card data with fallback priority
     */
    getImageUrl(card) {
        // Priority: image_url > image_info.image_url > fallback URLs
        if (card.image_url) {
            return card.image_url;
        }
        
        if (card.image_info?.image_url) {
            return card.image_info.image_url;
        }
        
        // Try different image URI sizes
        if (card.image_uris) {
            const imageUris = typeof card.image_uris === 'string' 
                ? JSON.parse(card.image_uris) 
                : card.image_uris;
            
            // Fallback order: normal > large > small > png
            const fallbackOrder = ['normal', 'large', 'small', 'png'];
            for (const size of fallbackOrder) {
                if (imageUris[size]) {
                    return imageUris[size];
                }
            }
        }
        
        return null;
    }
    
    /**
     * Get image status from card data
     */
    getImageStatus(card) {
        if (card.image_info?.image_status) {
            return card.image_info.image_status;
        }
        
        if (card.image_url) {
            // Determine status based on URL pattern
            if (card.image_url.includes('supabase') || card.image_url.includes('storage')) {
                return 'storage';
            } else if (card.image_url.startsWith('file://') || card.image_url.includes('localhost')) {
                return 'local';
            } else {
                return 'original';
            }
        }
        
        return 'not_available';
    }
    
    /**
     * Get appropriate CSS class based on layout
     */
    getImageClass(layout) {
        switch (layout) {
            case 'list':
            case 'compact':
                return 'card-image-small';
            case 'grid':
            default:
                return 'card-image';
        }
    }
    
    /**
     * Handle successful image load
     */
    handleImageLoad(img) {
        img.classList.add('image-loaded');
        img.classList.remove('image-loading');
        
        if (this.performanceTracker) {
            this.performanceTracker.incrementLoaded();
        }
        
        // Add load timer
        const timer = document.createElement('div');
        timer.className = 'image-load-timer';
        timer.textContent = `${Date.now() - (this.performanceTracker?.startTime || Date.now())}ms`;
        img.parentElement?.appendChild(timer);
        
        this.logImagePerformance(img.dataset.cardName, 'success');
    }
    
    /**
     * Handle image load error with retry logic
     */
    handleImageError(img) {
        const cardName = img.dataset.cardName || 'unknown';
        const currentAttempts = this.errorRetryAttempts.get(cardName) || 0;
        
        if (currentAttempts < this.maxRetryAttempts) {
            // Increment retry attempts
            this.errorRetryAttempts.set(cardName, currentAttempts + 1);
            
            // Try fallback URL or retry
            setTimeout(() => {
                const fallbackUrl = this.getFallbackUrl(img.src);
                if (fallbackUrl && fallbackUrl !== img.src) {
                    img.src = fallbackUrl;
                } else {
                    // Retry original URL
                    const originalSrc = img.src;
                    img.src = '';
                    setTimeout(() => img.src = originalSrc, 1000);
                }
            }, 1000 * (currentAttempts + 1)); // Exponential backoff
        } else {
            // Max retries reached, show error placeholder
            this.showErrorPlaceholder(img);
        }
        
        if (this.performanceTracker) {
            this.performanceTracker.incrementFailed();
        }
        
        this.logImagePerformance(cardName, 'error');
    }
    
    /**
     * Get fallback URL for failed image
     */
    getFallbackUrl(originalUrl) {
        // Try to get a different size or quality
        if (originalUrl.includes('normal')) {
            return originalUrl.replace('normal', 'small');
        } else if (originalUrl.includes('large')) {
            return originalUrl.replace('large', 'normal');
        }
        return null;
    }
    
    /**
     * Show error placeholder
     */
    showErrorPlaceholder(img) {
        img.classList.add('image-error');
        img.src = this.getErrorPlaceholderDataUrl();
        
        // Add retry button
        const retryBtn = document.createElement('button');
        retryBtn.className = 'image-retry';
        retryBtn.textContent = 'Retry';
        retryBtn.onclick = () => {
            this.retryImage(img);
            retryBtn.remove();
        };
        
        img.parentElement?.appendChild(retryBtn);
    }
    
    /**
     * Retry loading an image
     */
    retryImage(img) {
        const cardName = img.dataset.cardName;
        this.errorRetryAttempts.delete(cardName);
        img.classList.remove('image-error');
        
        // Reset src to trigger reload
        const originalSrc = img.dataset.src || img.src;
        img.src = '';
        setTimeout(() => img.src = originalSrc, 100);
    }
    
    /**
     * Load image (for lazy loading)
     */
    loadImage(img) {
        if (img.dataset.src) {
            img.classList.add('image-loading');
            img.src = img.dataset.src;
            img.removeAttribute('data-src');
        }
    }
    
    /**
     * Get placeholder data URL
     */
    getPlaceholderDataUrl() {
        return 'data:image/svg+xml;base64,' + btoa(`
            <svg width="200" height="280" xmlns="http://www.w3.org/2000/svg">
                <rect width="100%" height="100%" fill="#f8f9fa"/>
                <text x="50%" y="50%" font-family="Arial" font-size="14" fill="#6c757d" text-anchor="middle" dy=".3em">Loading...</text>
            </svg>
        `);
    }
    
    /**
     * Get error placeholder data URL
     */
    getErrorPlaceholderDataUrl() {
        return 'data:image/svg+xml;base64,' + btoa(`
            <svg width="200" height="280" xmlns="http://www.w3.org/2000/svg">
                <rect width="100%" height="100%" fill="#dee2e6"/>
                <text x="50%" y="40%" font-family="Arial" font-size="12" fill="#6c757d" text-anchor="middle" dy=".3em">Image</text>
                <text x="50%" y="60%" font-family="Arial" font-size="12" fill="#6c757d" text-anchor="middle" dy=".3em">Not Available</text>
            </svg>
        `);
    }
    
    /**
     * Log image performance metrics
     */
    logImagePerformance(cardName, status) {
        const timestamp = Date.now();
        console.log(`Image Performance - ${cardName}: ${status} at ${timestamp}`);
        
        // Send to analytics if available
        if (window.analytics?.track) {
            window.analytics.track('Image Load', {
                cardName,
                status,
                timestamp
            });
        }
    }
    
    /**
     * Preload images for better performance
     */
    preloadImages(imageUrls) {
        imageUrls.forEach(url => {
            if (!this.imageCache.has(url)) {
                const img = new Image();
                img.onload = () => this.imageCache.set(url, 'loaded');
                img.onerror = () => this.imageCache.set(url, 'error');
                img.src = url;
            }
        });
    }
    
    /**
     * Clear cache and reset error attempts
     */
    clearCache() {
        this.imageCache.clear();
        this.errorRetryAttempts.clear();
    }
    
    /**
     * Get system statistics
     */
    getStats() {
        return {
            cacheSize: this.imageCache.size,
            errorAttempts: this.errorRetryAttempts.size,
            observedImages: this.lazyLoadObserver ? 'enabled' : 'disabled'
        };
    }
}

// Initialize global image system
window.MTGImageSystem = MTGImageSystem;
window.imageSystem = new MTGImageSystem();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MTGImageSystem;
}