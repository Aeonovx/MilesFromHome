document.addEventListener('DOMContentLoaded', () => {
    const urlParams = new URLSearchParams(window.location.search);
    const articleId = urlParams.get('id');

    if (articleId) {
        fetchArticle(articleId);
    }
    
    document.getElementById('theme-toggle').addEventListener('click', () => {
        document.documentElement.classList.toggle('light');
        document.documentElement.classList.toggle('dark');
    });
});

function fetchArticle(id) {
    fetch(`/api/news/${id}`)
        .then(response => response.json())
        .then(data => {
            renderArticle(data.article);
            renderRelatedArticles(data.related);
        })
        .catch(error => console.error('Error fetching article:', error));
}

function renderArticle(article) {
    document.title = `${article.headline} | MilesFromHome`;
    const container = document.getElementById('article-container');
    const imageUrl = article.image_url || `https://via.placeholder.com/800x400/161b22/8b949e?text=MilesFromHome`;

    container.innerHTML = `
        <img src="${imageUrl}" alt="${article.headline}" class="article-image">
        <h1>${article.headline}</h1>
        <div class="meta-info">
            <span>By <strong>${article.source}</strong></span>
            <span>${new Date(article.published).toLocaleDateString()}</span>
        </div>
        <div class="share-buttons">
            <span>Share:</span>
            <a href="https://wa.me/?text=${encodeURIComponent(article.headline + ' ' + article.link)}" target="_blank">WhatsApp</a>
            <a href="https://twitter.com/intent/tweet?url=${encodeURIComponent(article.link)}&text=${encodeURIComponent(article.headline)}" target="_blank">Twitter</a>
            <a href="https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(article.link)}" target="_blank">Facebook</a>
        </div>
        <div class="explained-text">
            ${marked.parse(article.explained_version)}
        </div>
        <a href="${article.link}" target="_blank" class="read-more-btn">Read Original Article &rarr;</a>
    `;
}

function renderRelatedArticles(articles) {
    const grid = document.getElementById('related-articles-grid');
    articles.forEach(article => {
        const card = document.createElement('div');
        card.className = 'preview-card';
        card.onclick = () => window.location.href = `/article.html?id=${article.id}`;
        card.innerHTML = `<img src="${article.image_url || 'https://via.placeholder.com/400x180'}" class="card-image"><div class="card-content"><h3>${article.headline}</h3></div>`;
        grid.appendChild(card);
    });
}