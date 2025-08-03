document.addEventListener('DOMContentLoaded', () => {
    const articleContainer = document.getElementById('article-container');
    const urlParams = new URLSearchParams(window.location.search);
    const articleId = urlParams.get('id');

    if (!articleId) {
        articleContainer.innerHTML = '<h1>Article not found</h1><p>No article ID was provided.</p>';
        return;
    }

    fetch(`/api/news/${articleId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Article not found');
            }
            return response.json();
        })
        .then(article => {
            document.title = `${article.headline} | MilesFromHome`; // Update page title
            articleContainer.innerHTML = `
                <img src="${article.image_url || 'https://via.placeholder.com/800x400?text=No+Image'}" alt="${article.headline}" class="article-image">
                <h1>${article.headline}</h1>
                <div class="meta-info">
                    <span>By <strong>${article.source}</strong></span> | <span>${new Date(article.published).toLocaleDateString()}</span>
                </div>
                <div class="explained-text">
                    ${marked.parse(article.explained_version)}
                </div>
            `;
        })
        .catch(error => {
            console.error('Error fetching article:', error);
            articleContainer.innerHTML = '<h1>Error</h1><p>Could not load the article. It may have been moved or deleted.</p>';
        });
});