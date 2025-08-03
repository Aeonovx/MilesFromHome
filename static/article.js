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
    
    // Enhanced image handling for article page
    let imageUrl = article.image_url;
    
    if (!imageUrl || imageUrl.trim() === '') {
        const sourceColors = {
            "Reuters": "ff6600",
            "BBC News": "bb1919", 
            "The Guardian": "052962",
            "TechCrunch": "00d084",
            "Associated Press": "0066cc",
            "Tech Reuters": "ff6600",
            "EU Guardian": "003399",
            "Travel Guardian": "ff6b35",
            "Tech Guardian": "052962"
        };
        
        const color = sourceColors[article.source] || "1a1f26";
        const encodedSource = encodeURIComponent(article.source);
        imageUrl = `https://via.placeholder.com/900x400/${color}/ffffff?text=${encodedSource}`;
    }

    // Format the date properly
    const publishedDate = article.published !== "N/A" 
        ? new Date(article.published).toLocaleDateString('en-US', { 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
          })
        : "Recent";

    container.innerHTML = `
        <div class="article-image-container">
            <img src="${imageUrl}" 
                 alt="${article.headline}" 
                 class="article-image"
                 onerror="handleArticleImageError(this, '${article.source}')">
        </div>
        <h1>${article.headline}</h1>
        <div class="meta-info">
            <span>By <strong>${article.source}</strong></span>
            <span>${publishedDate}</span>
        </div>
        <div class="share-buttons">
            <span>Share:</span>
            <a href="https://wa.me/?text=${encodeURIComponent(article.headline + ' - ' + window.location.href)}" target="_blank">WhatsApp</a>
            <a href="https://twitter.com/intent/tweet?url=${encodeURIComponent(window.location.href)}&text=${encodeURIComponent(article.headline)}" target="_blank">Twitter</a>
            <a href="https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(window.location.href)}" target="_blank">Facebook</a>
        </div>
        <div class="explained-text">
            ${marked.parse(article.explained_version || article.summary)}
        </div>
        <a href="${article.link}" target="_blank" class="read-more-btn">Read Original Article &rarr;</a>
    `;
}

function handleArticleImageError(img, source) {
    console.log(`Article image failed to load for source: ${source}`);
    
    const sourceColors = {
        "Reuters": "2e5266",
        "BBC News": "bb1919", 
        "The Guardian": "052962",
        "TechCrunch": "00d084",
        "Associated Press": "0066cc"
    };
    
    const color = sourceColors[source] || "1a1f26";
    const encodedSource = encodeURIComponent(source);
    img.src = `https://via.placeholder.com/800x400/${color}/ffffff?text=${encodedSource}`;
    img.onerror = null;
}

function renderRelatedArticles(articles) {
    const grid = document.getElementById('related-articles-grid');
    articles.forEach(article => {
        const card = document.createElement('div');
        card.className = 'preview-card';
        card.onclick = () => window.location.href = `/article.html?id=${article.id}`;
        
        let imageUrl = article.image_url;
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
                     class="card-image" 
                     alt="${article.headline}"
                     onerror="this.src='https://via.placeholder.com/400x180/1a1f26/ffffff?text=News'">
            </div>
            <div class="card-content">
                <h3>${article.headline}</h3>
            </div>
        `;
        grid.appendChild(card);
    });
}