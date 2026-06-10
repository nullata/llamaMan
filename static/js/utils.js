// Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Shared state
// -------------------------------------------------------------------------
let instances = {};          // id -> instance obj
let downloads = {};          // id -> download obj
let containerStats = {};     // id -> {cpu_pct, mem_used_mb, mem_limit_mb}
let peerContainerStats = {}; // remote-instance stats pulled from peers (cluster)
let _peerStatsTick = 0;      // throttles how often peers are polled for stats
let huggingFaceTokens = [];  // saved Hugging Face tokens (safe metadata only)
let selectedModelPath = null;
let allModels = [];           // cached model list for filtering

// -------------------------------------------------------------------------
// Collapsible sections (persisted in localStorage)
// -------------------------------------------------------------------------
const SECTION_STORAGE_KEY = 'llamaman-sections';
const TAB_STORAGE_KEY = 'llamaman-tabs';

function loadSectionStates() {
  try { return JSON.parse(localStorage.getItem(SECTION_STORAGE_KEY)) || {}; }
  catch { return {}; }
}

function loadTabStates() {
  try { return JSON.parse(localStorage.getItem(TAB_STORAGE_KEY)) || {}; }
  catch { return {}; }
}

function saveSectionState(section, collapsed) {
  const states = loadSectionStates();
  states[section] = collapsed;
  localStorage.setItem(SECTION_STORAGE_KEY, JSON.stringify(states));
}

function saveTabState(group, tab) {
  const states = loadTabStates();
  states[group] = tab;
  localStorage.setItem(TAB_STORAGE_KEY, JSON.stringify(states));
}

function setActiveTab(group, tab) {
  const root = document.querySelector(`.tabbed-card[data-tab-group="${group}"]`);
  if (!root) return;

  let found = false;
  root.querySelectorAll('.settings-tab[data-tab]').forEach(btn => {
    const active = btn.dataset.tab === tab;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', active ? 'true' : 'false');
    btn.setAttribute('tabindex', active ? '0' : '-1');
    if (active) found = true;
  });

  if (!found) return;

  root.querySelectorAll('.tab-panel[data-tab-panel]').forEach(panel => {
    panel.hidden = panel.dataset.tabPanel !== tab;
  });

  // Show/hide the favorite star button only on the launch tab
  const starBtn = root.querySelector('.settings-tab-star');
  if (starBtn) starBtn.hidden = tab !== 'launch';

  saveTabState(group, tab);
}

function toggleSection(section) {
  const heading = document.querySelector(`.collapsible-heading[data-section="${section}"]`);
  const body = document.querySelector(`.collapsible-body[data-section="${section}"]`);
  if (!heading || !body) return;
  const collapsed = body.classList.toggle('hidden');
  heading.classList.toggle('collapsed', collapsed);
  saveSectionState(section, collapsed);
}

function restoreSectionStates() {
  const states = loadSectionStates();
  // Main content sections
  document.querySelectorAll('.collapsible-heading[data-section]').forEach(heading => {
    const section = heading.dataset.section;
    const body = document.querySelector(`.collapsible-body[data-section="${section}"]`);
    if (!body) return;
    if (states[section]) {
      body.classList.add('hidden');
      heading.classList.add('collapsed');
    }
    heading.addEventListener('click', () => toggleSection(section));
  });
  // Sidebar downloads section
  const dlToggle = document.getElementById('dl-section-toggle');
  const dlPanel = document.getElementById('downloads-panel');
  if (dlToggle && dlPanel) {
    if (states.downloads) {
      dlPanel.classList.add('hidden');
      dlToggle.classList.add('collapsed');
    }
  }
}

function initTabs() {
  const tabStates = loadTabStates();
  document.querySelectorAll('.tabbed-card[data-tab-group]').forEach(root => {
    const group = root.dataset.tabGroup;
    const buttons = [...root.querySelectorAll('.settings-tab[data-tab]')];
    if (buttons.length === 0) return;

    buttons.forEach((btn, index) => {
      btn.addEventListener('click', () => setActiveTab(group, btn.dataset.tab));
      btn.addEventListener('keydown', (e) => {
        if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
        e.preventDefault();
        const direction = e.key === 'ArrowRight' ? 1 : -1;
        const next = buttons[(index + direction + buttons.length) % buttons.length];
        next.focus();
        setActiveTab(group, next.dataset.tab);
      });
    });

    const savedTab = tabStates[group];
    const defaultTab = root.dataset.defaultTab || buttons[0].dataset.tab;
    const initialTab = buttons.some(btn => btn.dataset.tab === savedTab) ? savedTab : defaultTab;
    setActiveTab(group, initialTab);
  });
}

initTabs();
restoreSectionStates();

// -------------------------------------------------------------------------
// Confirm Modal
// -------------------------------------------------------------------------
function showConfirm(title, message) {
  return new Promise((resolve) => {
    const overlay = document.getElementById('confirm-modal');
    document.getElementById('confirm-modal-title').textContent = title;
    document.getElementById('confirm-modal-message').textContent = message;
    overlay.classList.add('open');

    const cleanup = (result) => {
      overlay.classList.remove('open');
      btnOk.removeEventListener('click', onOk);
      btnCancel.removeEventListener('click', onCancel);
      btnClose.removeEventListener('click', onCancel);
      overlay.removeEventListener('click', onOverlay);
      resolve(result);
    };
    const onOk = () => cleanup(true);
    const onCancel = () => cleanup(false);
    const onOverlay = (e) => { if (e.target === overlay) cleanup(false); };

    const btnOk = document.getElementById('btn-confirm-ok');
    const btnCancel = document.getElementById('btn-confirm-cancel');
    const btnClose = document.getElementById('btn-confirm-close');
    btnOk.addEventListener('click', onOk);
    btnCancel.addEventListener('click', onCancel);
    btnClose.addEventListener('click', onCancel);
    overlay.addEventListener('click', onOverlay);
  });
}

// -------------------------------------------------------------------------
// Utilities
// -------------------------------------------------------------------------
async function apiFetch(url, opts) {
  const res = await fetch(url, opts);
  if (res.status === 401) { window.location.href = '/login'; return null; }
  return res;
}

async function readApiResponse(res) {
  if (!res) return {};
  const contentType = res.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return await res.json();
  }
  const text = await res.text();
  return {
    error: text ? text.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim() : 'Unexpected server response',
  };
}

// Encode a file path for use in a URL - encodes each segment but keeps slashes.
function encodePathForUrl(p) {
  return p.split('/').map(encodeURIComponent).join('/');
}

function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

function formatUptime(startedAt) {
  const secs = Math.floor(Date.now() / 1000 - startedAt);
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`;
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  return `${h}h ${m}m`;
}

async function nextAvailablePort() {
  try {
    const res = await apiFetch('/api/next-port');
    const data = await res.json();
    return data.port || 8000;
  } catch (e) {
    return 8000;
  }
}

function escHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function clampPercent(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value));
}

function renderMeterSvg({ meterClass = '', toneClass = '', percent = 0, meterHeight = 14, cornerRadius = 3 } = {}) {
  const classes = ['meter-svg', meterClass, toneClass].filter(Boolean).join(' ');
  const clampedPercent = clampPercent(percent);
  return `
    <svg class="${classes}" viewBox="0 0 100 ${meterHeight}" preserveAspectRatio="none" aria-hidden="true" focusable="false">
      <rect class="meter-track" x="0" y="0" width="100" height="${meterHeight}" rx="${cornerRadius}" ry="${cornerRadius}"></rect>
      <rect class="meter-fill" x="0" y="0" width="${clampedPercent}" height="${meterHeight}" rx="${cornerRadius}" ry="${cornerRadius}"></rect>
    </svg>
  `;
}
