// Enhanced source-specific themes matching backend
const SOURCE_THEMES = {
    "Reuters": { color: "ff6600", textColor: "ffffff", icon: "📰" },
    "BBC News": { color: "bb1919", textColor: "ffffff", icon: "📺" },
    "The Guardian": { color: "052962", textColor: "ffffff", icon: "🗞️" },
    "TechCrunch": { color: "00d084", textColor: "000000", icon: "💻" },
    "Associated Press": { color: "0066cc", textColor: "ffffff", icon: "📄" },
    "Tech Reuters": { color: "ff6600", textColor: "ffffff", icon: "🔧" },
    "EU News": { color: "003399", textColor: "ffffff", icon: "🇪🇺" },
    "Travel Guardian": { color: "052962", textColor: "ffffff", icon: "✈️" },
    "Tech Guardian": { color: "052962", textColor: "ffffff", icon: "💻" }
};

function createHighQualityFallback(source, headline, size = "800x400") {
    const theme = SOURCE_THEMES[source] || {
        color: "1a1f26",
        textColor: "8b949e", 
        icon: "📰"
    };
    
    const encodedSource = encodeURIComponent(source.replace(" ", "+"));
    return `https://via.placeholder.com/${size}/${theme.color}/${theme.textColor}?text=${theme.icon}+${encodedSource}`;
}

function enhanceImageUrl(url) {
    if (!url || url.includes('placeholder')) return url;
    
    // Enhance common image URLs for better quality
    let enhancedUrl = url;
    
    // Common resolution upgrades
    enhancedUrl = enhancedUrl
        .replace('_s.jpg', '_b.jpg')       // Small to big
        .replace('_m.jpg', '_b.jpg')       // Medium to big  
        .replace('thumbnail', 'large')      // Thumbnail to large
        .replace('150x150', '800x600')     // Small dimensions to large
        .replace('300x200', '800x600')     // Medium dimensions to large
        .replace('/thumb/', '/large/')     // Thumb directory to large
        .replace('?w=150', '?w=800')       // Width parameter
        .replace('?h=150', '?h=600')       // Height parameter
        .replace('&w=150', '&w=800')       // Width parameter
        .replace('&h=150', '&h=600');      // Height parameter
    
    // Add quality parameters for supported services
    if (enhancedUrl.includes('images.unsplash.com')) {
        enhancedUrl += (enhancedUrl.includes('?') ? '&' : '?') + 'q=80&w=800';
    } else if (enhancedUrl.includes('cdn') && !enhancedUrl.includes('?')) {
        enhancedUrl += '?quality=80&width=800';
    }
    
    return enhancedUrl;
}

function createPreviewCard(article) {
    const card = document.createElement('div');
    card.className = 'preview-card';
    card.onclick = () => window.location.href = `/article.html?id=${article.id}`;

    // Enhanced high-quality image handling
    let imageUrl = article.image_url;
    
    // Enhance image quality if it's a real image
    if (imageUrl && !imageUrl.includes('placeholder')) {
        imageUrl = enhanceImageUrl(imageUrl);
    }
    
    // Fallback if no image
    if (!imageUrl || imageUrl.trim() === '' || imageUrl === 'null') {
        console.warn(`No image URL for article: ${article.headline.substring(0, 50)}...`);
        imageUrl = createHighQualityFallback(article.source, article.headline, "800x400");
    }

    card.innerHTML = `
        <div class="card-image-container">
            <img src="${imageUrl}" 
                 alt="${article.headline}" 
                 class="card-image" 
                 loading="lazy"
                 onload="handleImageLoad(this)"
                 onerror="handleImageError(this, '${article.source}', '${article.headline.replace(/'/g, '\\\'')}')"
                 style="opacity: 0; transition: opacity 0.5s ease;">
            <div class="image-overlay">
                <span class="category-badge ${article.category.toLowerCase().replace(' ', '-')}">${article.category}</span>
            </div>
        </div>
        <div class="card-content">
            <h3>${article.headline}</h3>
            <p class="article-summary">${article.summary.substring(0, 120)}...</p>
        </div>
        <div class="card-footer">
            <span class="hotness-indicator">🔥 ${article.hotness}% Hot</span>
            <span>${article.source}</span>
        </div>
    `;

    return card;
}

// Handle successful image load with quality check
function handleImageLoad(img) {
    img.style.opacity = '1';
    
    // Check if image loaded properly (not broken)
    if (img.naturalWidth === 0 || img.naturalHeight === 0) {
        console.warn('Image appears broken, triggering fallback');
        img.onerror();
    }
}

// Enhanced error handling with quality fallbacks
function handleImageError(img, source, headline) {
    console.log(`Image failed to load: ${img.src.substring(0, 50)}... for ${headline.substring(0, 30)}...`);
    
    // Try different quality fallbacks before using placeholder
    const originalSrc = img.src;
    
    if (!originalSrc.includes('placeholder')) {
        // First try: Remove quality parameters that might be causing issues
        let fallbackUrl = originalSrc.split('?')[0];
        
        if (fallbackUrl !== originalSrc) {
            console.log('Trying without quality parameters...');
            img.src = fallbackUrl;
            img.onerror = () => {
                // Second try: Use high-quality placeholder
                const placeholderUrl = createHighQualityFallback(source, headline, "800x400");
                img.src = placeholderUrl;
                img.onerror = null;
                img.style.opacity = '1';
            };
            return;
        }
        
        // Final fallback: Use themed placeholder
        const placeholderUrl = createHighQualityFallback(source, headline, "800x400");
        img.src = placeholderUrl;
        img.onerror = null;
        img.style.opacity = '1';
    }
}

// Image preloading for better performance
function preloadImages(articles) {
    articles.slice(0, 6).forEach(article => {
        if (article.image_url && !article.image_url.includes('placeholder')) {
            const img = new Image();
            img.src = enhanceImageUrl(article.image_url);
        }
    });
}

// Enhanced loading states and performance
let allArticles = [];
let displayedArticles = 0;
const articlesPerPage = 9;

document.addEventListener('DOMContentLoaded', () => {
    fetchNews();
    document.getElementById('search-bar').addEventListener('input', debounce((e) => fetchNews(e.target.value), 300));
    document.getElementById('load-more').addEventListener('click', loadMoreArticles);
});

// Debounce function for search
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function fetchNews(searchTerm = '', category = 'All') {
    const url = `/api/news?search=${encodeURIComponent(searchTerm)}&category=${encodeURIComponent(category)}`;
    
    // Show loading state
    const grid = document.getElementById('news-grid');
    if (displayedArticles === 0) {
        grid.innerHTML = '<div class="loading"><div class="spinner"></div>Loading high-quality news...</div>';
    }
    
    fetch(url)
        .then(response => {
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response.json();
        })
        .then(data => {
            allArticles = data.articles;
            displayedArticles = 0;
            grid.innerHTML = '';
            renderCategorySelector(data.categories, category);
            
            // Show category counts
            updateCategoryCounts(data.categories);
            
            // Preload images for better performance
            preloadImages(allArticles);
            
            loadMoreArticles();
        })
        .catch(error => {
            console.error('Error fetching news:', error);
            grid.innerHTML = '<div class="error">Failed to load news. Please try again.</div>';
        });
}

function updateCategoryCounts(categories) {
    // Count articles per category
    const categoryCounts = {};
    allArticles.forEach(article => {
        categoryCounts[article.category] = (categoryCounts[article.category] || 0) + 1;
    });
    
    console.log('📊 Category Distribution:', categoryCounts);
}

function renderCategorySelector(categories, activeCategory) {
    const container = document.getElementById('category-selector');
    const icons = { 
        "All": '🌐', 
        "EU News": '🇪🇺', 
        "Tech": '💻', 
        "Travel": '✈️', 
        "General": '📰' 
    };
    
    // Count articles per category for display
    const categoryCounts = {};
    allArticles.forEach(article => {
        categoryCounts[article.category] = (categoryCounts[article.category] || 0) + 1;
    });
    categoryCounts["All"] = allArticles.length;
    
    container.innerHTML = categories.map(cat => `
        <button class="${activeCategory === cat ? 'active' : ''}" 
                onclick="fetchNews(document.getElementById('search-bar').value, '${cat}')">
            <span class="icon">${icons[cat] || '•'}</span> 
            ${cat}
            <span class="count">(${categoryCounts[cat] || 0})</span>
        </button>
    `).join('');
}

function loadMoreArticles() {
    const grid = document.getElementById('news-grid');
    const fragment = document.createDocumentFragment();
    const articlesToLoad = allArticles.slice(displayedArticles, displayedArticles + articlesPerPage);

    if (articlesToLoad.length === 0) {
        document.getElementById('load-more').style.display = 'none';
        if (displayedArticles === 0) {
            grid.innerHTML = '<div class="no-articles">No articles found for this category.</div>';
        }
        return;
    }

    articlesToLoad.forEach(article => {
        const card = createPreviewCard(article);
        fragment.appendChild(card);
    });
    
    grid.appendChild(fragment);
    displayedArticles += articlesToLoad.length;
    
    // Update load more button
    const loadMoreBtn = document.getElementById('load-more');
    if (displayedArticles >= allArticles.length) {
        loadMoreBtn.style.display = 'none';
    } else {
        loadMoreBtn.style.display = 'block';
        loadMoreBtn.textContent = `Load More (${allArticles.length - displayedArticles} remaining)`;
    }
}