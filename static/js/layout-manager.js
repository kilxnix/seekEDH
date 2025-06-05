// js/layout-manager.js - Layout Management for MTG Card Displays

/**
 * Layout Manager for MTG Card Image System
 * Handles different layout modes (grid, list, compact) and responsive behavior
 */
class LayoutManager {
    constructor() {
        this.currentLayout = 'grid';
        this.imageSystem = window.imageSystem;
        this.init();
    }
    
    /**
     * Initialize layout manager
     */
    init() {
        this.setupLayoutToggleListeners();
        this.setupResponsiveBreakpoints();
        this.loadSavedLayout();
    }
    
    /**
     * Setup layout toggle event listeners
     */
    setupLayoutToggleListeners() {
        document.querySelectorAll('input[name="layoutOptions"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                if (e.target.checked) {
                    const layout = e.target.id.replace('Layout', '');
                    this.switchLayout(layout);
                }
            });
        });
    }
    
    /**
     * Setup responsive breakpoint handling
     */
    setupResponsiveBreakpoints() {
        // Create media query listeners for different screen sizes
        const breakpoints = {
            mobile: window.matchMedia('(max-width: 576px)'),
            tablet: window.matchMedia('(max-width: 768px)'),
            desktop: window.matchMedia('(min-width: 769px)')
        };
        
        Object.entries(breakpoints).forEach(([size, mq]) => {
            mq.addEventListener('change', () => {
                this.handleBreakpointChange(size, mq.matches);
            });
            
            // Check initial state
            if (mq.matches) {
                this.handleBreakpointChange(size, true);
            }
        });
    }
    
    /**
     * Handle responsive breakpoint changes
     */
    handleBreakpointChange(size, matches) {
        if (matches) {
            switch (size) {
                case 'mobile':
                    this.applyMobileOptimizations();
                    break;
                case 'tablet':
                    this.applyTabletOptimizations();
                    break;
                case 'desktop':
                    this.applyDesktopOptimizations();
                    break;
            }
        }
    }
    
    /**
     * Apply mobile-specific optimizations
     */
    applyMobileOptimizations() {
        // Force compact layout on mobile for better performance
        if (this.currentLayout === 'grid') {
            this.switchLayout('compact', false); // Don't save preference
        }
        
        // Reduce image quality on mobile
        this.updateImagePreferences({
            preferredSize: 'small',
            lazyLoadingThreshold: '25px'
        });
    }
    
    /**
     * Apply tablet-specific optimizations
     */
    applyTabletOptimizations() {
        this.updateImagePreferences({
            preferredSize: 'normal',
            lazyLoadingThreshold: '50px'
        });
    }
    
    /**
     * Apply desktop-specific optimizations
     */
    applyDesktopOptimizations() {
        this.updateImagePreferences({
            preferredSize: 'normal',
            lazyLoadingThreshold: '100px'
        });
    }
    
    /**
     * Switch between layout modes
     */
    switchLayout(layout, savePreference = true) {
        if (this.currentLayout === layout) return;
        
        const oldLayout = this.currentLayout;
        this.currentLayout = layout;
        
        // Update UI state
        this.updateLayoutToggle(layout);
        
        // Re-render current results if any exist
        this.reRenderCurrentResults(oldLayout, layout);
        
        // Save preference
        if (savePreference) {
            this.saveLayoutPreference(layout);
        }
        
        console.log(`Layout switched from ${oldLayout} to ${layout}`);
    }
    
    /**
     * Update layout toggle UI
     */
    updateLayoutToggle(layout) {
        const radio = document.getElementById(`${layout}Layout`);
        if (radio) {
            radio.checked = true;
        }
    }
    
    /**
     * Re-render current results with new layout
     */
    reRenderCurrentResults(oldLayout, newLayout) {
        const resultContainers = [
            'imageSearchResults',
            'visualResults',
            'searchResults',
            'synergyResults',
            'universalResults'
        ];
        
        resultContainers.forEach(containerId => {
            const container = document.getElementById(containerId);
            if (container && container.children.length > 1) {
                // Check if container has card results
                const cardResults = container.querySelectorAll('.card-result, .image-grid-item');
                if (cardResults.length > 0) {
                    this.convertLayout(container, oldLayout, newLayout);
                }
            }
        });
    }
    
    /**
     * Convert existing layout to new layout
     */
    convertLayout(container, oldLayout, newLayout) {
        const cards = this.extractCardData(container);
        if (cards.length > 0) {
            this.renderCards(container, cards, newLayout);
        }
    }
    
    /**
     * Extract card data from existing DOM elements
     */
    extractCardData(container) {
        const cards = [];
        
        // Extract from grid items
        container.querySelectorAll('.image-grid-item').forEach(item => {
            const titleElement = item.querySelector('.card-title, h6');
            const imageElement = item.querySelector('img');
            
            if (titleElement) {
                cards.push({
                    name: titleElement.textContent,
                    image_url: imageElement?.src,
                    type_line: item.querySelector('.text-muted')?.textContent || '',
                    element: item
                });
            }
        });
        
        // Extract from card results
        container.querySelectorAll('.card-result').forEach(item => {
            const titleElement = item.querySelector('.card-title, h6');
            const imageElement = item.querySelector('img');
            
            if (titleElement) {
                cards.push({
                    name: titleElement.textContent,
                    image_url: imageElement?.src,
                    type_line: item.querySelector('.text-muted')?.textContent || '',
                    oracle_text: item.querySelector('.card-text')?.textContent || '',
                    element: item
                });
            }
        });
        
        return cards;
    }
    
    /**
     * Render cards in specified layout
     */
    renderCards(container, cards, layout) {
        let html = '';
        
        switch (layout) {
            case 'grid':
                html = this.renderGridLayout(cards);
                break;
            case 'list':
                html = this.renderListLayout(cards);
                break;
            case 'compact':
                html = this.renderCompactLayout(cards);
                break;
        }
        
        // Find the results content area and update it
        const resultsArea = container.querySelector('.image-grid, .row') || container;
        if (resultsArea !== container) {
            resultsArea.outerHTML = html;
        } else {
            // Find content after the header/summary
            const summary = container.querySelector('.mb-3');
            if (summary) {
                // Remove everything after the summary
                let nextElement = summary.nextElementSibling;
                while (nextElement) {
                    const toRemove = nextElement;
                    nextElement = nextElement.nextElementSibling;
                    toRemove.remove();
                }
                // Add new content
                summary.insertAdjacentHTML('afterend', html);
            } else {
                container.innerHTML = html;
            }
        }
        
        // Re-initialize images for new layout
        this.initializeImagesInContainer(container, cards, layout);
    }
    
    /**
     * Render grid layout
     */
    renderGridLayout(cards) {
        let html = '<div class="image-grid">';
        
        cards.forEach(card => {
            html += `
                <div class="image-grid-item" onclick="showImageModal('${card.name}', ${JSON.stringify(card).replace(/"/g, '&quot;')})">
                    <div id="image-container-${this.sanitizeId(card.name)}" class="loading-skeleton">
                        <!-- Image will be loaded here -->
                    </div>
                    <div class="p-3">
                        <h6 class="card-title mb-2">${card.name}</h6>
                        <p class="small text-muted mb-0">${card.type_line || 'Unknown Type'}</p>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        return html;
    }
    
    /**
     * Render list layout
     */
    renderListLayout(cards) {
        let html = '';
        
        cards.forEach(card => {
            html += `
                <div class="card card-result">
                    <div class="card-body">
                        <div class="card-result-list" onclick="showImageModal('${card.name}', ${JSON.stringify(card).replace(/"/g, '&quot;')})">
                            <div id="image-container-${this.sanitizeId(card.name)}" class="loading-skeleton">
                                <!-- Image will be loaded here -->
                            </div>
                            <div class="flex-grow-1">
                                <h6 class="card-title">${card.name}</h6>
                                <p class="small text-muted">${card.type_line || 'Unknown Type'}</p>
                                ${card.oracle_text ? `<p class="card-text small">${card.oracle_text}</p>` : ''}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        return html;
    }
    
    /**
     * Render compact layout
     */
    renderCompactLayout(cards) {
        let html = '<div class="row g-2">';
        
        cards.forEach(card => {
            html += `
                <div class="col-md-6">
                    <div class="card card-result">
                        <div class="card-body p-2">
                            <div class="d-flex align-items-center" onclick="showImageModal('${card.name}', ${JSON.stringify(card).replace(/"/g, '&quot;')})">
                                <div id="image-container-${this.sanitizeId(card.name)}" class="loading-skeleton me-2">
                                    <!-- Image will be loaded here -->
                                </div>
                                <div class="flex-grow-1">
                                    <h6 class="mb-1">${card.name}</h6>
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
     * Initialize images in container after layout change
     */
    initializeImagesInContainer(container, cards, layout) {
        setTimeout(() => {
            cards.forEach(card => {
                const containerId = `image-container-${this.sanitizeId(card.name)}`;
                const imageContainer = document.getElementById(containerId);
                
                if (imageContainer && this.imageSystem) {
                    const imageElement = this.imageSystem.createImageElement(card, 'normal', layout);
                    imageContainer.innerHTML = '';
                    imageContainer.appendChild(imageElement);
                    imageContainer.classList.remove('loading-skeleton');
                }
            });
        }, 50);
    }
    
    /**
     * Sanitize string for use as DOM ID
     */
    sanitizeId(str) {
        return str.replace(/[^a-zA-Z0-9]/g, '');
    }
    
    /**
     * Update image preferences
     */
    updateImagePreferences(preferences) {
        // Apply preferences to existing images
        document.querySelectorAll('img[data-card-name]').forEach(img => {
            if (preferences.preferredSize && img.dataset.preferredSize !== preferences.preferredSize) {
                img.dataset.preferredSize = preferences.preferredSize;
            }
        });
        
        // Update lazy loading threshold
        if (preferences.lazyLoadingThreshold && this.imageSystem?.lazyLoadObserver) {
            // Would need to recreate observer with new threshold
            // This is a placeholder for more advanced implementation
        }
    }
    
    /**
     * Save layout preference to localStorage
     */
    saveLayoutPreference(layout) {
        try {
            localStorage.setItem('mtg-layout-preference', layout);
        } catch (e) {
            console.warn('Could not save layout preference:', e);
        }
    }
    
    /**
     * Load saved layout preference
     */
    loadSavedLayout() {
        try {
            const savedLayout = localStorage.getItem('mtg-layout-preference');
            if (savedLayout && ['grid', 'list', 'compact'].includes(savedLayout)) {
                this.switchLayout(savedLayout, false);
            }
        } catch (e) {
            console.warn('Could not load layout preference:', e);
        }
    }
    
    /**
     * Get current layout configuration
     */
    getLayoutConfig() {
        return {
            current: this.currentLayout,
            available: ['grid', 'list', 'compact'],
            responsive: {
                mobile: 'compact',
                tablet: 'list',
                desktop: 'grid'
            }
        };
    }
    
    /**
     * Reset to default layout
     */
    resetLayout() {
        this.switchLayout('grid');
        try {
            localStorage.removeItem('mtg-layout-preference');
        } catch (e) {
            console.warn('Could not clear layout preference:', e);
        }
    }
}

// Initialize global layout manager
window.LayoutManager = LayoutManager;
window.layoutManager = new LayoutManager();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LayoutManager;
}