// Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Boot - event wiring, intervals, initial loads
// -------------------------------------------------------------------------
const refreshModelsBtn = document.getElementById('btn-refresh-models');
if (refreshModelsBtn) refreshModelsBtn.addEventListener('click', loadModels);

// Downloads sidebar section toggle
const dlSectionToggle = document.getElementById('dl-section-toggle');
if (dlSectionToggle) dlSectionToggle.addEventListener('click', () => {
  const panel = document.getElementById('downloads-panel');
  const collapsed = panel.classList.toggle('hidden');
  dlSectionToggle.classList.toggle('collapsed', collapsed);
  saveSectionState('downloads', collapsed);
});

const modelSearch = document.getElementById('model-search');
if (modelSearch) modelSearch.addEventListener('input', renderModels);

const refreshGpuBtn = document.getElementById('btn-refresh-gpu');
if (refreshGpuBtn) refreshGpuBtn.addEventListener('click', loadGpuInfo);

const refreshSystemBtn = document.getElementById('btn-refresh-system');
if (refreshSystemBtn) refreshSystemBtn.addEventListener('click', loadSystemInfo);

// Instance list refresh (every 5s, includes status from background poller)
setInterval(pollInstances, 5000);

// Container resource usage (CPU/memory) refresh (every 3s)
setInterval(pollContainerStats, 3000);

// Download status refresh (every 3s)
setInterval(pollDownloads, 3000);

// System info refresh (every 10s)
setInterval(loadSystemInfo, 10000);

// GPU VRAM refresh (every 10s)
setInterval(loadGpuInfo, 10000);

// Cleanup metadata refresh (every 60s)
setInterval(refreshCleanupLastRan, 60000);

// Cluster registry refresh (every 5s; drives per-node System/GPU cards and
// the Cluster tab. No-op rendering when clustering is disabled.)
if (typeof loadClusterNodes === 'function') {
  setInterval(loadClusterNodes, 5000);
}

// Initial load
loadModels();
if (typeof loadClusterNodes === 'function') loadClusterNodes();
loadSystemInfo();
loadGpuInfo();
pollInstances().then(() => { updatePortSuggestion(); pollContainerStats(); });

const params = new URLSearchParams(window.location.search);
const presetModelPath = params.get('model_path');
if (presetModelPath && modelPathField) {
  modelPathField.value = presetModelPath;
  if (typeof setActiveTab === 'function') setActiveTab('settings', 'launch');
  updateGpuLayersTotal(presetModelPath);
}

pollDownloads();
loadSettings();
loadApiKeys();
loadImages();
if (typeof populateLaunchImageSelect === 'function') populateLaunchImageSelect();

// Cluster grouping fields (alias + fallback) under "Share queue with same model".
// Only meaningful when share-queue is on; revealed AND cleared on the toggle so
// the launch form stays uncluttered AND a stale alias typed before the toggle
// was flipped off can't sneak through on submit. The backend re-enforces this
// invariant too (launch_instance + preset save force-empty when share_queue
// is false), so the UI gating is defense in depth, not the only check.
function updateShareQueueClusterRow() {
  const t = document.getElementById('f-share-queue');
  const row = document.getElementById('f-share-queue-cluster');
  if (!t || !row) return;
  const on = t.checked;
  row.hidden = !on;
  const groupIn = document.getElementById('f-share-queue-group');
  const fbIn = document.getElementById('f-share-queue-fallback');
  if (groupIn) {
    groupIn.disabled = !on;
    if (!on) groupIn.value = '';
  }
  if (fbIn) {
    fbIn.disabled = !on;
    if (!on) fbIn.checked = false;
  }
}
const shareQueueToggle = document.getElementById('f-share-queue');
if (shareQueueToggle) shareQueueToggle.addEventListener('change', updateShareQueueClusterRow);
updateShareQueueClusterRow();

// Node selector changes (cluster mode): refetch the per-node data on switch.
const imagesNodeSel = document.getElementById('images-node');
if (imagesNodeSel) imagesNodeSel.addEventListener('change', loadImages);
const launchNodeSel = document.getElementById('f-node');
if (launchNodeSel) launchNodeSel.addEventListener('change', () => {
  if (typeof onLaunchNodeChanged === 'function') onLaunchNodeChanged();
  else updatePortSuggestion();
});
const downloadNodeSel = document.getElementById('d-node');
if (downloadNodeSel) downloadNodeSel.addEventListener('change', refreshDownloadDiskSpace);

// Info-tip clipping fallback: when a centered tooltip would extend past the
// viewport, switch to anchoring it on the icon's right (or left) edge.
// Tooltip max-width is 240px; centered means up to 120px on either side.
function updateInfoTipClipping(tip) {
  const rect = tip.getBoundingClientRect();
  const center = rect.left + rect.width / 2;
  const halfMax = 120;
  const overflowsRight = center + halfMax > window.innerWidth;
  const overflowsLeft = center - halfMax < 0;
  tip.classList.toggle('clip-right', overflowsRight);
  tip.classList.toggle('clip-left', !overflowsRight && overflowsLeft);
}
document.addEventListener('mouseover', (e) => {
  const tip = e.target.closest && e.target.closest('.info-tip');
  if (tip) updateInfoTipClipping(tip);
});
document.addEventListener('focusin', (e) => {
  const tip = e.target.closest && e.target.closest('.info-tip');
  if (tip) updateInfoTipClipping(tip);
});
