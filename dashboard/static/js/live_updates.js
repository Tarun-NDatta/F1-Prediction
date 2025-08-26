// Live updates page JS extracted from template

// 2025 drivers lineup
const f1_2025_drivers = [
  "Max Verstappen", "Yuki Tsunoda",
  "Charles Leclerc", "Lewis Hamilton",
  "Lando Norris", "Oscar Piastri",
  "George Russell", "Kimi Antonelli",
  "Fernando Alonso", "Lance Stroll",
  "Pierre Gasly", "Jack Doohan",
  "Alexander Albon", "Carlos Sainz Jr.",
  "Nico Hu\u00FClkenberg", "Gabriel Bortoleto",
  "Isack Hadjar",
  "Oliver Bearman", "Esteban Ocon"
];

document.addEventListener('DOMContentLoaded', function() {
  const commentarySection = document.querySelector('.commentary-section');
  if (commentarySection) commentarySection.scrollTop = commentarySection.scrollHeight;
  const predictionSection = document.querySelector('.prediction-updates');
  if (predictionSection) predictionSection.scrollTop = predictionSection.scrollHeight;

  let raceInterval = null;
  let isRaceActive = false;
  let preRacePrediction = null;

  document.getElementById('startRaceBtn').addEventListener('click', startMockRace);
  document.getElementById('stopRaceBtn').addEventListener('click', stopMockRace);
  document.getElementById('safetyCarBtn').addEventListener('click', () => triggerEvent('safety_car'));
  document.getElementById('badPitStopBtn').addEventListener('click', () => triggerEvent('bad_pit_stop'));
  document.getElementById('weatherChangeBtn').addEventListener('click', () => triggerEvent('weather_change'));

  async function startMockRace() {
    const trackName = document.getElementById('trackSelect').value;
    const speed = document.getElementById('speedSelect').value;
    try {
      const response = await fetch('/api/mock-race/start/', {
        method: 'POST', headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCookie('csrftoken') },
        body: JSON.stringify({ event_name: trackName, simulation_speed: speed })
      });
      const data = await response.json();
      if (data.success) {
        isRaceActive = true; updateRaceStatus(`Race Started: ${trackName}`, 'success');
        document.getElementById('mockRaceDetails').classList.remove('d-none');
        const startingOrder = generateStartingGrid();
        generatePreRacePrediction(startingOrder);
        document.getElementById('startRaceBtn').classList.add('d-none');
        document.getElementById('stopRaceBtn').classList.remove('d-none');
        document.getElementById('interactiveEvents').classList.remove('d-none');
        document.getElementById('simulationControls').classList.add('mb-3');
        document.getElementById('commentaryFeed').innerHTML = '';
        document.getElementById('predictionFeed').innerHTML = '';
        const pollInterval = Math.max(1000, data.lap_interval * 1000);
        raceInterval = setInterval(pollRaceStatus, pollInterval);
        setTimeout(pollRaceStatus, 500);
        addCommentary('Race Control', `${trackName} simulation started! ${data.total_laps} laps ahead. Grid positions and pre-race predictions generated.`);
      } else {
        updateRaceStatus('Failed to start race', 'danger');
      }
    } catch (error) {
      console.error('Error starting race:', error);
      updateRaceStatus('Error starting race', 'danger');
    }
  }

  function generateStartingGrid() {
    const startingGrid = document.getElementById('startingGrid');
    const order = [...f1_2025_drivers].slice(0, 20).sort(() => Math.random() - 0.5);
    let gridHtml = '';
    order.forEach((driver, index) => {
      const position = index + 1; let positionClass = 'grid-position';
      if (position === 1) positionClass += ' pole';
      else if (position === 2) positionClass += ' front-row';
      else if (position <= 10) positionClass += ' top-10';
      gridHtml += `<span class="${positionClass}">P${position}: ${driver}</span>`;
    });
    startingGrid.innerHTML = gridHtml; return order;
  }

  function generatePreRacePrediction(order) {
    const drivers = order && order.length ? order : [...f1_2025_drivers].slice(0, 20);
    preRacePrediction = [...drivers];
    const preRaceContent = document.getElementById('preRacePredictionContent');
    let predictionHtml = '';
    preRacePrediction.slice(0, 10).forEach((driver, index) => {
      const position = index + 1;
      predictionHtml += `<div class="prediction-comparison"><span>P${position}: ${driver}</span><small class="text-muted">Pre-race model</small></div>`;
    });
    preRaceContent.innerHTML = predictionHtml;
  }

  async function stopMockRace() {
    try {
      await fetch('/api/mock-race/stop/', { method: 'POST', headers: { 'X-CSRFToken': getCookie('csrftoken') } });
      resetRaceUI();
    } catch (error) { console.error('Error stopping race:', error); }
  }

  async function pollRaceStatus() {
    if (!isRaceActive) return;
    try {
      const response = await fetch('/api/mock-race/status/');
      if (!response.ok) {
        if (response.status === 500) { addCommentary('System Warning', 'Temporary server issue - retrying...'); return; }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      if (data.success && data.race_data) {
        const raceData = data.race_data;
        if (raceData.status === 'running') {
          updateRaceStatus(`Lap ${raceData.current_lap}/${raceData.total_laps} - ML: ${raceData.ml_updates_active ? 'Active' : 'Stopped'}`, raceData.ml_updates_active ? 'success' : 'warning');
          const mlBadge = document.getElementById('mlStatus');
          if (!raceData.ml_updates_active) { mlBadge.classList.remove('d-none','bg-secondary'); mlBadge.classList.add('bg-warning'); mlBadge.textContent = 'Final Prediction Made'; }
          document.getElementById('eventsRemaining').textContent = `${raceData.max_events - raceData.events_used} remaining`;
          const canAddEvents = raceData.can_add_events;
          document.getElementById('safetyCarBtn').disabled = !canAddEvents;
          document.getElementById('badPitStopBtn').disabled = !canAddEvents;
          document.getElementById('weatherChangeBtn').disabled = !canAddEvents;
        } else if (raceData.status === 'finished') {
          updateRaceStatus('Race Finished!', 'primary'); clearInterval(raceInterval); isRaceActive = false; setTimeout(showFinalResults, 2000);
        }
        if (raceData.commentary && raceData.commentary.length > 0) {
          raceData.commentary.forEach(comment => { if (comment.lap === raceData.current_lap) addCommentary(`Lap ${comment.lap}`, comment.message.replace(`Lap ${comment.lap}: `, '')); });
        }
        if (raceData.ml_updates_active && raceData.ml_predictions && Object.keys(raceData.ml_predictions).length > 0) {
          updateMLPredictions(raceData.ml_predictions, raceData.race_state);
        } else if (raceData.ml_updates_active && raceData.current_lap % 2 === 0) {
          updateMockPredictions(raceData);
        }
      } else {
        addCommentary('System Error', 'Failed to get race data from server');
      }
    } catch (error) {
      console.error('Error polling race status:', error); addCommentary('Connection Error', `Failed to connect to server: ${error.message}`);
      if (String(error).includes('404') || String(error).includes('No active race')) { updateRaceStatus('Race stopped due to error', 'danger'); resetRaceUI(); }
    }
  }

  async function triggerEvent(eventType) {
    try {
      const randomDriver = f1_2025_drivers[Math.floor(Math.random() * f1_2025_drivers.length)];
      const response = await fetch('/api/mock-race/event/', { method: 'POST', headers: { 'Content-Type':'application/json', 'X-CSRFToken': getCookie('csrftoken') }, body: JSON.stringify({ event_type: eventType, target_lap: null, driver_name: eventType === 'bad_pit_stop' ? randomDriver : null }) });
      const data = await response.json();
      if (data.success) {
        addCommentary('Event Scheduled', data.message); document.getElementById('eventsRemaining').textContent = `${data.events_remaining} remaining`;
        if (data.events_remaining === 0) { document.getElementById('safetyCarBtn').disabled = true; document.getElementById('badPitStopBtn').disabled = true; document.getElementById('weatherChangeBtn').disabled = true; }
      } else { addCommentary('Event Error', data.error); }
    } catch (error) { console.error('Error triggering event:', error); }
  }

  async function showFinalResults() {
    try {
      const response = await fetch('/api/mock-race/results/');
      const data = await response.json();
      if (data.success) {
        addCommentary('Final Results', 'Race completed! Check the prediction comparison below.');
        let resultsHtml = '<div class="final-results">';
        resultsHtml += '<h6>Final Race Results</h6>';
        data.final_results.slice(0,10).forEach(result => {
          resultsHtml += `<div class="prediction-item"><div class="prediction-driver">P${result.position}: ${result.driver}</div><div class="prediction-change">${result.pit_stops} pit stops</div></div>`;
        });
        if (data.prediction_comparison && Object.keys(data.prediction_comparison).length > 0) {
          resultsHtml += '<hr><h6>ML Prediction Accuracy</h6>';
          const comparison = data.prediction_comparison;
          if (comparison.models_compared) {
            Object.keys(comparison.models_compared).forEach(modelName => {
              const modelData = comparison.models_compared[modelName];
              resultsHtml += `<div class=\"prediction-item\"><div class=\"prediction-driver\">${modelName.replace('_',' ').toUpperCase()}</div><div class=\"prediction-change\">MAE: <strong>${modelData.mae}</strong> | Accuracy: <strong>${modelData.accuracy_within_1}%</strong></div></div>`;
            });
            if (comparison.best_model) {
              resultsHtml += `<div class=\"alert alert-success mt-2\"><strong>Best Model:</strong> ${comparison.best_model.name.replace('_',' ').toUpperCase()} (MAE: ${comparison.best_model.mae})</div>`;
            }
          }
          if (comparison.final_prediction_lap) { resultsHtml += `<small class=\"text-muted\">Final predictions made at lap ${comparison.final_prediction_lap}</small>`; }
        } else { resultsHtml += '<hr><p class="text-muted">No ML prediction comparison available</p>'; }
        resultsHtml += '</div>'; document.getElementById('predictionFeed').innerHTML = resultsHtml;
      }
    } catch (error) { console.error('Error getting final results:', error); }
  }

  function addCommentary(time, message) {
    const commentaryFeed = document.getElementById('commentaryFeed');
    const commentaryItem = document.createElement('div'); commentaryItem.className = 'commentary-item';
    commentaryItem.innerHTML = `<div class=\"commentary-time\">${time}</div><div class=\"commentary-text\">${message}</div>`;
    commentaryFeed.appendChild(commentaryItem);
    const commentarySection = document.getElementById('commentarySection'); commentarySection.scrollTop = commentarySection.scrollHeight;
    const items = commentaryFeed.children; if (items.length > 20) commentaryFeed.removeChild(items[0]);
  }

  function updateMLPredictions(mlPredictions, raceState) {
    const predictionFeed = document.getElementById('predictionFeed');
    const positions = raceState.driver_positions.slice(0, 10);
    let predictionsHtml = '<div class="ml-predictions-header"><h6>Live ML Predictions</h6></div>';
    Object.keys(mlPredictions).forEach(modelName => {
      const modelData = mlPredictions[modelName]; const predictions = modelData.predictions || [];
      predictionsHtml += `<div class=\"model-predictions mb-3\"><h6 class=\"model-name\">${modelName.replace('_',' ').toUpperCase()}</h6>`;
      predictions.slice(0,5).forEach((pred, index) => {
        const driver = positions[index]; const currentPos = driver ? driver.position : index + 1;
        const predictedPosition = Math.max(1, Math.min(20, Math.round(pred)));
        const change = predictedPosition - currentPos; const changeStr = change > 0 ? `+${change}` : change.toString();
        const changeClass = change > 0 ? 'text-danger' : change < 0 ? 'text-success' : 'text-muted';
        predictionsHtml += `<div class=\"prediction-item\"><div class=\"prediction-driver\">${driver ? driver.driver_name : f1_2025_drivers[index] || 'Unknown'} <small class=\"text-muted\">(Currently P${currentPos})</small></div><div class=\"prediction-change\">Predicted: <strong>P${predictedPosition}</strong> <span class=\"${changeClass}\">(${changeStr === '0' ? 'No change' : changeStr})</span></div></div>`;
      });
      predictionsHtml += '</div>';
    });
    predictionFeed.innerHTML = predictionsHtml; updateLivePredictionTimeline(mlPredictions, positions);
  }

  function updateLivePredictionTimeline(mlPredictions, positions) {
    const livePredictions = document.getElementById('livePredictions');
    const livePredictionContent = document.getElementById('livePredictionContent');
    livePredictions.classList.remove('d-none');
    const firstModel = Object.keys(mlPredictions)[0];
    if (firstModel && mlPredictions[firstModel].predictions) {
      const predictions = mlPredictions[firstModel].predictions; let timelineHtml = '';
      predictions.slice(0,5).forEach((pred, index) => {
        const driver = positions[index];
        const predictedPosition = Math.max(1, Math.min(20, Math.round(pred)));
        timelineHtml += `<div class=\"prediction-comparison\"><span>P${predictedPosition}: ${driver ? driver.driver_name : f1_2025_drivers[index] || 'Unknown'}</span><small class=\"text-muted\">Live update</small></div>`;
      });
      livePredictionContent.innerHTML = timelineHtml;
    }
  }

  function updateMockPredictions(raceData) {
    const predictionFeed = document.getElementById('predictionFeed');
    const positions = raceData.race_state.driver_positions.slice(0, 5);
    let predictionsHtml = '<div class="mock-predictions-header"><h6>Mock Predictions</h6></div>';
    positions.forEach((driver, index) => {
      const change = Math.random() > 0.5 ? '+' : '-'; const changeValue = Math.floor(Math.random() * 3) + 1; const probability = 85 - (index * 10) + Math.floor(Math.random() * 10);
      predictionsHtml += `<div class=\"prediction-item\"><div class=\"prediction-driver\">P${driver.position}: ${driver.driver_name}</div><div class=\"prediction-change\">Position probability: <strong>${probability}%</strong> (${change}${changeValue} from start)</div></div>`;
    });
    predictionFeed.innerHTML = predictionsHtml;
  }

  function updateRaceStatus(message, type) {
    const statusElement = document.getElementById('raceStatus');
    statusElement.className = `alert alert-${type} mb-0`;
    const iconMap = { 'success': 'fas fa-circle text-success', 'warning': 'fas fa-circle text-warning', 'danger': 'fas fa-circle text-danger', 'primary': 'fas fa-checkered-flag text-primary' };
    statusElement.innerHTML = `<i class="${iconMap[type] || 'fas fa-circle text-secondary'} me-1"></i>${message}`;
  }

  function resetRaceUI() {
    isRaceActive = false; preRacePrediction = null; if (raceInterval) { clearInterval(raceInterval); raceInterval = null; }
    document.getElementById('mockRaceDetails').classList.add('d-none');
    document.getElementById('startRaceBtn').classList.remove('d-none');
    document.getElementById('stopRaceBtn').classList.add('d-none');
    document.getElementById('interactiveEvents').classList.add('d-none');
    document.getElementById('mlStatus').classList.add('d-none');
    updateRaceStatus('No active simulation', 'secondary');
    document.getElementById('safetyCarBtn').disabled = false;
    document.getElementById('badPitStopBtn').disabled = false;
    document.getElementById('weatherChangeBtn').disabled = false;
    document.getElementById('eventsRemaining').textContent = '2 remaining';
    document.getElementById('preRacePredictionContent').innerHTML = '<em>Available after race starts</em>';
    document.getElementById('livePredictions').classList.add('d-none');
    document.getElementById('livePredictionContent').innerHTML = '<em>Updates during race</em>';
  }

  function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
      const cookies = document.cookie.split(';');
      for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        if (cookie.substring(0, name.length + 1) === (name + '=')) {
          cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
          break;
        }
      }
    }
    return cookieValue;
  }
});

