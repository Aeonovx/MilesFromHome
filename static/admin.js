document.addEventListener('DOMContentLoaded', () => {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            document.getElementById('total-articles').textContent = data.total_articles;
            document.getElementById('total-subscribers').textContent = data.subscribers;
            document.getElementById('last-updated').textContent = new Date(data.last_updated).toLocaleString();
            
            createChart('sourceChart', 'Articles by Source', data.articles_per_source);
            createChart('categoryChart', 'Articles by Category', data.articles_per_category);
        })
        .catch(error => console.error('Error fetching stats:', error));
});

function createChart(canvasId, label, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(data),
            datasets: [{
                label: label,
                data: Object.values(data),
                backgroundColor: ['#388bfd', '#f778ba', '#56d364', '#ffb87a', '#a371f7', '#6e7681'],
                borderColor: '#1a1f26',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#e6edf3' }
                }
            }
        }
    });
}