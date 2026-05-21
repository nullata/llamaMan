// Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Stats Modal - per-instance rollups derived from the request log
// -------------------------------------------------------------------------

function fmtNum(n) {
  if (n == null) return '–';
  return Number(n).toLocaleString();
}

function fmtTps(n) {
  return n == null ? '–' : `${Number(n).toFixed(1)} t/s`;
}

function fmtMs(n) {
  if (n == null) return '–';
  return n >= 1000 ? `${(n / 1000).toFixed(2)} s` : `${Math.round(n)} ms`;
}

function statTile(label, value, sub) {
  return `<div class="stat-tile">
    <div class="stat-value">${value}</div>
    <div class="stat-label">${label}</div>
    ${sub ? `<div class="stat-sub">${sub}</div>` : ''}
  </div>`;
}

function renderStats(data) {
  const body = document.getElementById('stats-body');

  if (!data.recording) {
    body.innerHTML = `<div class="stats-empty">
      <i class="fa-solid fa-chart-line"></i>
      <p>Request logging is off. Enable it in Settings → Application to start
      collecting per-request stats.</p>
    </div>`;
    return;
  }

  if (!data.turn_count) {
    body.innerHTML = `<div class="stats-empty">
      <i class="fa-solid fa-hourglass-half"></i>
      <p>No recorded requests for this instance yet. Stats appear here once it
      handles traffic.</p>
    </div>`;
    return;
  }

  const totalTokens = (data.prompt_tokens || 0) + (data.completion_tokens || 0);
  const tiles = [
    statTile('Requests', fmtNum(data.turn_count),
      data.error_count ? `<span class="text-danger">${fmtNum(data.error_count)} errors</span>` : ''),
    statTile('Avg throughput', fmtTps(data.avg_tokens_per_sec),
      data.max_tokens_per_sec != null ? `peak ${fmtTps(data.max_tokens_per_sec)}` : ''),
    statTile('Avg TTFT', fmtMs(data.avg_ttft_ms)),
    statTile('Avg latency', fmtMs(data.avg_duration_ms)),
    statTile('Completion tokens', fmtNum(data.completion_tokens),
      `${fmtNum(data.prompt_tokens)} prompt`),
    statTile('Total tokens', fmtNum(totalTokens),
      data.streamed_count ? `${fmtNum(data.streamed_count)} streamed` : ''),
  ];

  let span = '';
  if (data.first_seen_at && data.last_seen_at) {
    const first = new Date(data.first_seen_at).toLocaleString();
    const last = new Date(data.last_seen_at).toLocaleString();
    span = `<div class="stats-span">First ${first} · Last ${last}</div>`;
  }

  body.innerHTML = `<div class="stats-grid">${tiles.join('')}</div>${span}`;
}

async function openStatsModal(instId, modelName) {
  const modal = document.getElementById('stats-modal');
  document.getElementById('stats-modal-title').textContent =
    modelName ? `Stats - ${modelName}` : 'Stats';
  document.getElementById('stats-body').textContent = 'Loading…';
  modal.classList.add('open');
  try {
    const res = await apiFetch(`/api/request-log/stats?inst_id=${encodeURIComponent(instId)}`);
    renderStats(await res.json());
  } catch (e) {
    document.getElementById('stats-body').textContent = 'Error loading stats';
  }
}

function closeStats() {
  document.getElementById('stats-modal').classList.remove('open');
}

const closeStatsBtn = document.getElementById('btn-close-stats');
const statsModalEl = document.getElementById('stats-modal');
if (closeStatsBtn && statsModalEl) {
  closeStatsBtn.addEventListener('click', closeStats);
  statsModalEl.addEventListener('click', (e) => {
    if (e.target === statsModalEl) closeStats();
  });
}
