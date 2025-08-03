// Source-specific themes matching backend
const SOURCE_THEMES = {
    "Reuters": { color: "ff6600", textColor: "ffffff", icon: "📰" },
    "BBC News": { color: "bb1919", textColor: "ffffff", icon: "📺" },
    "The Guardian": { color: "052962", textColor: "ffffff", icon: "🗞️" },
    "TechCrunch": { color: "00d084", textColor: "000000", icon: "💻" },
    "Associated Press": { color: "0066cc", textColor: "ffffff", icon: "📄" }
};

function createFallbackImage(source, headline, size = "400x180") {
    const theme = SOURCE_THEMES[source] || {
        color: "1a1f26",
        textColor: "8b949e", 
        icon: "📰"
    };
    
    const encodedSource = encodeURIComponent(source.replace(" ", "+"));
    return `https://via.placeholder.com/${size}/${theme.color}/${theme.textColor}?text=${theme.icon}+${encodedSource}`;
}

function createPreviewCard(article) {
    const card = document.createElement('div');
    card.className = 'preview-card';
    card.onclick = () => window.location.href = `/article.html?id=${article.id}`;

    // Enhanced image handling - GUARANTEED to show something
    let imageUrl = article.image_url;
    
    // Double-check that we have an image URL
    if (!imageUrl || imageUrl.trim() === '' || imageUrl === 'null' || imageUrl === 'undefined') {
        console.warn(`No image URL for article: ${article.headline.substring(0, 50)}...`);
        imageUrl = createFallbackImage(article.source, article.headline);
    }

    card.innerHTML = `
        <div class="card-image-container">
            <img src="${imageUrl}" 
                 alt="${article.headline}" 
                 class="card-image" 
                 loading="lazy"
                 onload="handleImageLoad(this)"
                 onerror="handleImageError(this, '${article.source}', '${article.headline.replace(/'/g, '\\\'')}')"
                 style="opacity: 0; transition: opacity 0.3s;">
        </div>
        <div class="card-content">
            <h3>${article.headline}</h3>
        </div>
        <div class="card-footer">
            <span class="hotness-indicator">🔥 ${article.hotness}% Hot</span>
            <span>${article.source}</span>
        </div>
    `;

    return card;
}

// Handle successful image load
function handleImageLoad(img) {
    img.style.opacity = '1';
}

// Enhanced error handling
function handleImageError(img, source, headline) {
    console.log(`Image failed to load for: ${headline.substring(0, 30)}... from ${source}`);
    
    // Set fallback image
    const fallbackUrl = createFallbackImage(source, headline);
    
    // Only change if not already a placeholder to prevent infinite loops
    if (!img.src.includes('placeholder')) {
        img.src = fallbackUrl;
        img.onerror = null; // Remove error handler to prevent loops
        img.style.opacity = '1'; // Show the fallback immediately
    }
}

// Enhanced image preloading
function preloadImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(url);
        img.onerror = () => reject(url);
        img.src = url;
    });
}

// Rest of the existing code with enhancements
let allArticles = [];
let displayedArticles = 0;
const articlesPerPage = 9;

document.addEventListener('DOMContentLoaded', () => {
    fetchNews();
    document.getElementById('search-bar').addEventListener('input', (e) => fetchNews(e.target.value));
    document.getElementById('load-more').addEventListener('click', loadMoreArticles);
});

function fetchNews(searchTerm = '', category = 'All') {
    const url = `/api/news?search=${encodeURIComponent(searchTerm)}&category=${encodeURIComponent(category)}`;
    
    // Show loading state
    const grid = document.getElementById('news-grid');
    if (displayedArticles === 0) {
        grid.innerHTML = '<div class="loading">Loading news...</div>';
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
            loadMoreArticles();
        })
        .catch(error => {
            console.error('Error fetching news:', error);
            grid.innerHTML = '<div class="error">Failed to load news. Please try again.</div>';
        });
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
    
    container.innerHTML = categories.map(cat => `
        <button class="${activeCategory === cat ? 'active' : ''}" 
                onclick="fetchNews(document.getElementById('search-bar').value, '${cat}')">
            <span class="icon">${icons[cat] || '•'}</span> ${cat}
        </button>
    `).join('');
}

function loadMoreArticles() {
    const grid = document.getElementById('news-grid');
    const fragment = document.createDocumentFragment();
    const articlesToLoad = allArticles.slice(displayedArticles, displayedArticles + articlesPerPage);

    if (articlesToLoad.length === 0) {
        document.getElementById('load-more').style.display = 'none';
        return;
    }

    articlesToLoad.forEach(article => {
        const card = createPreviewCard(article);
        fragment.appendChild(card);
    });
    
    grid.appendChild(fragment);
    displayedArticles += articlesToLoad.length;
    
    // Hide load more button if all articles are displayed
    const loadMoreBtn = document.getElementById('load-more');
    if (displayedArticles >= allArticles.length) {
        loadMoreBtn.style.display = 'none';
    } else {
        loadMoreBtn.style.display = 'block';
        loadMoreBtn.textContent = `Load More (${allArticles.length - displayedArticles} remaining)`;
    }
}