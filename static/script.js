document.addEventListener('DOMContentLoaded', () => {
    const newsContainer = document.getElementById('news-container');

    fetch('/api/news')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                newsContainer.innerHTML = `<p>${data.error}</p>`;
                return;
            }
            data.forEach(article => {
                const articleElement = document.createElement('article');
                articleElement.className = 'news-card';

                // Only add an image element if an image URL exists
                const imageHtml = article.image_url
                    ? `<img src="${article.image_url}" alt="${article.headline}" class="news-image">`
                    : '';

                articleElement.innerHTML = `
                    ${imageHtml}
                    <div class="news-content">
                        <h2><a href="${article.link}" target="_blank">${article.headline}</a></h2>
                        <div class="explained-text">
                            ${marked.parse(article.explained_version)}
                        </div>
                        <div class="news-meta">
                            <span>${article.source}</span> | <span>${new Date(article.published).toLocaleString()}</span>
                        </div>
                    </div>
                `;
                newsContainer.appendChild(articleElement);
            });
        })
        .catch(error => {
            console.error('Error fetching news:', error);
            newsContainer.innerHTML = '<p>Could not load news. Please try again later.</p>';
        });
});