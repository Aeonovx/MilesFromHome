function createPreviewCard(article) {
    const card = document.createElement('div');
    card.className = 'preview-card';
    card.onclick = () => window.location.href = `/article.html?id=${article.id}`;

    // Enhanced image handling with multiple fallbacks
    let imageUrl = article.image_url;
    
    // If no image URL or it's empty, create a themed placeholder
    if (!imageUrl || imageUrl.trim() === '') {
        const sourceColors = {
            "Reuters": "2e5266",
            "BBC News": "bb1919", 
            "The Guardian": "052962",
            "TechCrunch": "00d084",
            "Associated Press": "0066cc"
        };
        
        const color = sourceColors[article.source] || "1a1f26";
        const encodedSource = encodeURIComponent(article.source);
        imageUrl = `https://via.placeholder.com/400x180/${color}/ffffff?text=${encodedSource}`;
    }

    card.innerHTML = `
        <div class="card-image-container">
            <img src="${imageUrl}" 
                 alt="${article.headline}" 
                 class="card-image" 
                 loading="lazy"
                 onerror="handleImageError(this, '${article.source}')">
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

// Global function to handle image loading errors
function handleImageError(img, source) {
    console.log(`Image failed to load for source: ${source}`);
    
    const sourceColors = {
        "Reuters": "2e5266",
        "BBC News": "bb1919", 
        "The Guardian": "052962",
        "TechCrunch": "00d084",
        "Associated Press": "0066cc"
    };
    
    const color = sourceColors[source] || "1a1f26";
    const encodedSource = encodeURIComponent(source);
    img.src = `https://via.placeholder.com/400x180/${color}/ffffff?text=${encodedSource}`;
    
    // Remove the onerror to prevent infinite loops
    img.onerror = null;
}

// Rest of the existing code
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
    fetch(url)
        .then(response => response.json())
        .then(data => {
            allArticles = data.articles;
            displayedArticles = 0;
            document.getElementById('news-grid').innerHTML = '';
            renderCategorySelector(data.categories, category);
            loadMoreArticles();
        })
        .catch(error => console.error('Error fetching news:', error));
}

function renderCategorySelector(categories, activeCategory) {
    const container = document.getElementById('category-selector');
    const icons = { "All": '🌐', "EU News": '🇪🇺', "Tech": '💻', "Travel": '✈️', "General": '📰' };
    container.innerHTML = categories.map(cat => `
        <button class="${activeCategory === cat ? 'active' : ''}" onclick="fetchNews(document.getElementById('search-bar').value, '${cat}')">
            <span class="icon">${icons[cat] || '•'}</span> ${cat}
        </button>
    `).join('');
}

function loadMoreArticles() {
    const grid = document.getElementById('news-grid');
    const fragment = document.createDocumentFragment();
    const articlesToLoad = allArticles.slice(displayedArticles, displayedArticles + articlesPerPage);

    articlesToLoad.forEach(article => fragment.appendChild(createPreviewCard(article)));
    grid.appendChild(fragment);
    displayedArticles += articlesToLoad.length;
    document.getElementById('load-more').style.display = displayedArticles >= allArticles.length ? 'none' : 'block';
}