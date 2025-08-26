// Prediction page JS extracted from template
// Requires Chart.js loaded before this file

let perRaceChart = null;
function renderPerRaceChart(perRaceLabels, perRaceMae) {
  const perRaceCtx = document.getElementById('perRaceMae');
  if (!perRaceCtx || !perRaceLabels || !perRaceLabels.length) return;
  if (perRaceChart) { perRaceChart.destroy(); }
  perRaceChart = new Chart(perRaceCtx.getContext('2d'), {
    type: 'line',
    data: { labels: perRaceLabels, datasets: [{
      label: 'Per-Race MAE (lower is better)',
      data: perRaceMae,
      borderColor: '#e10600',
      backgroundColor: 'rgba(225, 6, 0, 0.15)',
      tension: 0.3,
    }]},
    options: { responsive: true, maintainAspectRatio: false }
  });
}

function toggleRaceDetails(idx) {
  const details = document.getElementById('details-' + idx);
  const arrow = document.getElementById('arrow-' + idx);
  const isOpen = details.style.display === 'block';
  document.querySelectorAll('[id^="details-"]').forEach(el => el.style.display = 'none');
  document.querySelectorAll('[id^="arrow-"]').forEach(el => el.classList.remove('rotated'));
  if (!isOpen) { details.style.display = 'block'; arrow.classList.add('rotated'); }
  if (window.predictionData) {
    renderPerRaceChart(window.predictionData.perRaceLabels, window.predictionData.perRaceMae);
  }
}

(function init() {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      if (window.predictionData) {
        renderPerRaceChart(window.predictionData.perRaceLabels, window.predictionData.perRaceMae);
      }
    });
  } else {
    if (window.predictionData) {
      renderPerRaceChart(window.predictionData.perRaceLabels, window.predictionData.perRaceMae);
    }
  }
})();

