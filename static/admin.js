document.addEventListener('DOMContentLoaded', () => {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            document.getElementById('total-articles').textContent = data.total_articles;
            document.getElementById('total-subscribers').textContent = data.subscribers;

            const sourcesContainer = document.getElementById('source-stats');
            for (const [source, count] of Object.entries(data.articles_per_source)) {
                const card = document.createElement('div');
                card.className = 'stat-card';
                card.innerHTML = `<h3>${source}</h3><p>${count}</p>`;
                sourcesContainer.appendChild(card);
            }
        })
        .catch(error => console.error('Error fetching stats:', error));
});