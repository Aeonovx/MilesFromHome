let allArticles = [];
let displayedArticles = 0;
const articlesPerPage = 9;

document.addEventListener('DOMContentLoaded', () => {
    fetchNews();
    setupEventListeners();
});

function setupEventListeners() {
    document.getElementById('search-bar').addEventListener('input', (e) => filterAndDisplay(e.target.value));
    document.getElementById('load-more').addEventListener('click', loadMoreArticles);
    document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
    document.getElementById('subscribe-form').addEventListener('submit', handleSubscription);
}

function fetchNews(searchTerm = '', sourceFilter = '') {
    const url = `/api/news?search=${encodeURIComponent(searchTerm)}&source=${encodeURIComponent(sourceFilter)}`;
    fetch(url)
        .then(response => response.json())
        .then(data => {
            allArticles = data.articles;
            displayedArticles = 0;
            document.getElementById('news-grid').innerHTML = '';
            
            renderFilterButtons(data.sources, sourceFilter);
            renderTrendingTopics(data.trending);
            
            loadMoreArticles();
        })
        .catch(error => console.error('Error fetching news:', error));
}

function renderFilterButtons(sources, activeSource) {
    const container = document.getElementById('filter-buttons');
    container.innerHTML = `<button class="${!activeSource ? 'active' : ''}" onclick="fetchNews(document.getElementById('search-bar').value, '')">All</button>`;
    sources.forEach(source => {
        container.innerHTML += `<button class="${activeSource === source ? 'active' : ''}" onclick="fetchNews(document.getElementById('search-bar').value, '${source}')">${source}</button>`;
    });
}

function renderTrendingTopics(topics) {
    const container = document.getElementById('trending-topics');
    container.innerHTML = '<h3>Trending:</h3>';
    topics.forEach(topic => {
        container.innerHTML += `<a href="#" onclick="fetchNews('${topic}')">${topic}</a>`;
    });
}

function loadMoreArticles() {
    const grid = document.getElementById('news-grid');
    const fragment = document.createDocumentFragment();
    const articlesToLoad = allArticles.slice(displayedArticles, displayedArticles + articlesPerPage);

    articlesToLoad.forEach(article => {
        const card = createPreviewCard(article);
        fragment.appendChild(card);
    });

    grid.appendChild(fragment);
    displayedArticles += articlesToLoad.length;

    document.getElementById('load-more').style.display = displayedArticles >= allArticles.length ? 'none' : 'block';
}

function createPreviewCard(article) {
    const card = document.createElement('div');
    card.className = 'preview-card';
    card.onclick = () => window.location.href = `/article.html?id=${article.id}`;
    
    const hotnessColor = article.hotness > 90 ? 'var(--hot-color)' : '#ffb87a';
    const imageUrl = article.image_url || `https://via.placeholder.com/400x180/161b22/8b949e?text=MilesFromHome`;

    card.innerHTML = `
        <img src="${imageUrl}" alt="${article.headline}" class="card-image">
        <div class="card-content">
            <h3>${article.headline}</h3>
            <p>${article.summary.substring(0, 100)}...</p>
        </div>
        <div class="card-footer">
            <span class="hotness-indicator" style="color: ${hotnessColor};">🔥 ${article.hotness}% Hot</span>
            <span class="source">${article.source}</span>
        </div>
    `;
    return card;
}

function filterAndDisplay(searchTerm) {
    fetchNews(searchTerm, ''); // Simple implementation: refetches from backend
}

function toggleTheme() {
    document.documentElement.classList.toggle('light');
    document.documentElement.classList.toggle('dark');
}

function handleSubscription(event) {
    event.preventDefault();
    const email = document.getElementById('email-input').value;
    const messageEl = document.getElementById('subscribe-message');
    fetch('/api/subscribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
    })
    .then(res => res.json())
    .then(data => {
        messageEl.textContent = data.message;
        messageEl.style.color = 'green';
    })
    .catch(() => {
        messageEl.textContent = 'Subscription failed.';
        messageEl.style.color = 'red';
    });
}