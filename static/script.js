document.addEventListener('DOMContentLoaded', () => {
    const newsGrid = document.getElementById('news-grid');

    fetch('/api/news')
        .then(response => response.json())
        .then(data => {
            if (data.error || !Array.isArray(data)) {
                newsGrid.innerHTML = `<p>Could not load news. Please try again later.</p>`;
                return;
            }
            data.forEach(article => {
                const card = document.createElement('div');
                card.className = 'preview-card';
                // Navigate to article page on click
                card.onclick = () => {
                    window.location.href = `/article.html?id=${article.id}`;
                };

                const hotnessColor = article.hotness > 90 ? '#ff7a7a' : '#ffb87a';
                const whatsappLink = `https://wa.me/?text=${encodeURIComponent(article.headline + " - " + article.link)}`;

                card.innerHTML = `
                    <img src="${article.image_url || 'https://via.placeholder.com/400x200?text=No+Image'}" alt="${article.headline}" class="card-image">
                    <div class="card-content">
                        <h3>${article.headline}</h3>
                        <p>${article.summary.substring(0, 100)}...</p>
                        <div class="card-footer">
                            <div class="hotness-indicator" style="color: ${hotnessColor};">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-fire" viewBox="0 0 16 16"><path d="M8 16c3.314 0 6-2 6-5.5 0-1.5-.5-4-2.5-6 .25 1.5-1.25 2-1.25 2C11 4 9 .5 6 0c.357 2 .5 4-2 6-1.25 1-2 2.729-2 4.5C2 14 4.686 16 8 16Zm0-1c-1.657 0-3-1-3-2.75 0-.75.25-2 1.25-3C6.125 10 7 10.5 7 10.5c-.375-1.25.5-3.25 2-3.5-.179 1-.25 2 1 3 .625.5 1 1.364 1 2.25C11 14 9.657 15 8 15Z"/></svg>
                                <span>${article.hotness}% Hot</span>
                            </div>
                            <a href="${whatsappLink}" class="whatsapp-share-btn" target="_blank" onclick="event.stopPropagation();">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-whatsapp" viewBox="0 0 16 16"><path d="M13.601 2.326A7.854 7.854 0 0 0 7.994 0C3.627 0 .068 3.558.064 7.926c0 1.399.366 2.76 1.057 3.965L0 16l4.204-1.102a7.933 7.933 0 0 0 3.79.965h.004c4.368 0 7.926-3.558 7.93-7.93A7.898 7.898 0 0 0 13.6 2.326zM7.994 14.521a6.573 6.573 0 0 1-3.356-.92l-.24-.144-2.494.654.666-2.433-.156-.251a6.56 6.56 0 0 1-1.007-3.505c0-3.626 2.957-6.584 6.591-6.584a6.56 6.56 0 0 1 4.66 1.931 6.557 6.557 0 0 1 1.928 4.66c-.004 3.639-2.961 6.592-6.592 6.592zm3.615-4.934c-.197-.099-1.17-.578-1.353-.646-.182-.065-.315-.099-.445.099-.133.197-.513.646-.627.775-.114.133-.232.148-.43.05-.197-.1-.836-.308-1.592-.985-.59-.525-.985-1.175-1.103-1.372-.114-.198-.011-.304.088-.403.087-.088.197-.232.296-.346.1-.114.133-.198.198-.33.065-.134.034-.248-.015-.347-.05-.1-.445-1.076-.612-1.47-.16-.389-.323-.335-.445-.34-.114-.007-.247-.007-.38-.007a.729.729 0 0 0-.529.247c-.182.198-.691.677-.691 1.654 0 .977.71 1.916.81 2.049.098.133 1.394 2.132 3.383 2.992.47.205.84.326 1.129.418.475.152.904.129 1.246.08.38-.058 1.171-.48 1.338-.943.164-.464.164-.86.114-.943-.049-.084-.182-.133-.38-.232z"/></svg>
                                <span>Share</span>
                            </a>
                        </div>
                    </div>
                `;
                newsGrid.appendChild(card);
            });
        })
        .catch(error => {
            console.error('Error fetching news:', error);
            newsGrid.innerHTML = '<p>Could not load news. Please try again later.</p>';
        });
});