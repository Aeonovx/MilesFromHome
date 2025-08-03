function createPreviewCard(article) {
    const card = document.createElement('div');
    card.className = 'preview-card';
    card.onclick = () => window.location.href = `/article.html?id=${article.id}`;

    // --- Bulletproof Image Handling ---
    let imageUrl = article.image_url;
    
    // 1. Check if the image URL is null, undefined, or an empty string.
    if (!imageUrl) {
        // If no image, use a placeholder.
        imageUrl = `https://via.placeholder.com/400x180/1a1f26/8b949e?text=MilesFromHome`;
        console.log(`Article ID ${article.id} ("${article.headline}") has no image. Using placeholder.`);
    }

    // 2. Build the HTML for the card.
    // This structure guarantees the <img> tag is ALWAYS present.
    card.innerHTML = `
        <img src="${imageUrl}" alt="${article.headline}" class="card-image" onerror="this.style.display='none'">
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


// Make sure the rest of your script.js file remains the same
// The functions below should already be in your file

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