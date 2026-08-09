'use strict';

function $(id) { return document.getElementById(id); }

function toast(message, type = 'success', duration = 3200) {
  const el = $('toast');
  el.textContent = message;
  el.className = `toast ${type}`;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, duration);
}

async function apiCall(url, options = {}) {
  const res = await fetch(url, { ...options });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText || 'Request failed');
  return data;
}

let running = false;

async function checkModel() {
  try {
    const s = await apiCall('/api/status');
    const badge = $('modelBadge');
    if (s.model_loaded) {
      badge.textContent = 'Model loaded';
      badge.className = 'badge badge-good';
    } else {
      badge.textContent = 'No model found';
      badge.className = 'badge badge-bad';
      if (s.model_error) toast(s.model_error, 'error', 6000);
    }
  } catch (e) { /* ignore */ }
}

async function startDetection() {
  $('btnStart').disabled = true;
  $('btnStart').textContent = 'Starting…';
  $('videoPlaceholder').textContent = 'Opening camera — this can take a few seconds…';
  try {
    await apiCall('/api/start', { method: 'POST' });
    running = true;
    $('videoFeed').src = '/video_feed?t=' + Date.now();
    $('videoPlaceholder').style.display = 'none';
    $('btnStop').disabled = false;
    toast('Detection started.');
  } catch (err) {
    toast(err.message, 'error', 6000);
    $('btnStart').disabled = false;
    $('videoPlaceholder').textContent = 'Camera is off';
  } finally {
    $('btnStart').textContent = 'Start Detection';
  }
}

async function stopDetection() {
  await apiCall('/api/stop', { method: 'POST' }).catch(() => {});
  running = false;
  $('videoFeed').src = '';
  $('videoPlaceholder').style.display = 'flex';
  $('videoPlaceholder').textContent = 'Camera is off';
  $('btnStart').disabled = false;
  $('btnStop').disabled = true;
  toast('Detection stopped.');
}

async function reloadModel() {
  try {
    await apiCall('/api/reload_model', { method: 'POST' });
    toast('Model reloaded.');
    checkModel();
  } catch (err) {
    toast(err.message, 'error', 6000);
  }
}

$('btnStart').addEventListener('click', startDetection);
$('btnStop').addEventListener('click', stopDetection);
$('btnReload').addEventListener('click', reloadModel);

async function poll() {
  checkModel();
  if (!running) return;
  try {
    const s = await apiCall('/api/status');

    if (s.camera_error) {
      toast(s.camera_error, 'error', 6000);
      running = false;
      $('videoFeed').src = '';
      $('videoPlaceholder').style.display = 'flex';
      $('videoPlaceholder').textContent = 'Camera error — see message above';
      $('btnStart').disabled = false;
      $('btnStop').disabled = true;
      return;
    }

    const labelEl = $('resultLabel');
    const confFill = $('confFill');
    const confLabel = $('confLabel');

    if (s.stalled) {
      labelEl.textContent = 'Camera stalled — Stop then Start again';
      labelEl.className = 'result-label low';
      return;
    }

    if (!s.hand_detected) {
      labelEl.textContent = 'No hand detected';
      labelEl.className = 'result-label low';
      confFill.style.width = '0%';
      confLabel.textContent = '0% confidence';
      return;
    }

    if (!s.label) {
      labelEl.textContent = 'Reading gesture…';
      labelEl.className = 'result-label low';
      return;
    }

    const pct = Math.round((s.confidence || 0) * 100);
    confFill.style.width = pct + '%';
    confLabel.textContent = `${pct}% confidence`;

    if (s.low_confidence) {
      labelEl.textContent = 'Low Confidence';
      labelEl.className = 'result-label low';
    } else {
      labelEl.textContent = s.label;
      labelEl.className = 'result-label';
    }
  } catch (e) { /* ignore transient errors */ }
}

setInterval(poll, 300);
poll();
