// Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// App shell - view routing, rail, theme
//
// The page is one server-rendered document holding every view as a
// <section class="view">; the hash picks which one is visible. No framework,
// no fetch-per-navigation: every view's DOM (and therefore every existing
// poller, form binding and ID) stays alive across navigation, so switching
// views never interrupts in-flight work like a model hash or a download.
// -------------------------------------------------------------------------

const VIEWS = ['dashboard', 'instances', 'launch', 'models', 'downloads', 'settings'];
const VIEW_TITLES = {
  dashboard: 'Dashboard',
  instances: 'Instances',
  launch: 'Launch Instance',
  models: 'Model Library',
  downloads: 'Downloads',
  settings: 'Settings',
};

function currentView() {
  const m = (window.location.hash || '').match(/^#\/([a-z]+)/);
  const v = m && m[1];
  return VIEWS.includes(v) ? v : 'dashboard';
}

function showView(view) {
  document.querySelectorAll('.view').forEach(sec => {
    sec.hidden = sec.dataset.view !== view;
  });
  document.querySelectorAll('[data-view-link]').forEach(a => {
    a.classList.toggle('active', a.dataset.viewLink === view);
  });
  const crumb = document.getElementById('view-title');
  if (crumb) crumb.textContent = VIEW_TITLES[view] || 'Dashboard';
  // Scroll back to the top of the content column on navigation.
  const main = document.getElementById('main');
  if (main) main.scrollTop = 0;
}

window.addEventListener('hashchange', () => showView(currentView()));

// Deep-link helper used by other modules (e.g. "Use" on a finished download).
function gotoView(view) {
  if (window.location.hash === `#/${view}`) showView(view);
  else window.location.hash = `#/${view}`;
}

// -------------------------------------------------------------------------
// Rail collapse (persisted)
// -------------------------------------------------------------------------
const RAIL_COLLAPSED_KEY = 'llamaman-rail-collapsed';

function applyRailCollapsed(collapsed) {
  document.body.classList.toggle('rail-collapsed', collapsed);
  const btn = document.getElementById('btn-rail-toggle');
  if (btn) {
    const icon = btn.querySelector('i');
    if (icon) icon.className = collapsed ? 'fa-solid fa-angles-right' : 'fa-solid fa-angles-left';
  }
}

const railToggleBtn = document.getElementById('btn-rail-toggle');
if (railToggleBtn) {
  railToggleBtn.addEventListener('click', () => {
    const collapsed = !document.body.classList.contains('rail-collapsed');
    applyRailCollapsed(collapsed);
    try { localStorage.setItem(RAIL_COLLAPSED_KEY, collapsed ? '1' : '0'); } catch (e) { /* ignore */ }
  });
}
try {
  if (localStorage.getItem(RAIL_COLLAPSED_KEY) === '1') applyRailCollapsed(true);
} catch (e) { /* ignore */ }

// -------------------------------------------------------------------------
// Theme toggle (data-theme on <html>; the inline head script applies the
// stored value before first paint, so this only handles switching).
// -------------------------------------------------------------------------
const THEME_KEY = 'llamaman-theme';

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const btn = document.getElementById('btn-theme-toggle');
  if (btn) {
    const icon = btn.querySelector('i');
    if (icon) icon.className = theme === 'light' ? 'fa-solid fa-sun' : 'fa-solid fa-moon';
    btn.title = theme === 'light' ? 'Switch to dark theme' : 'Switch to light theme';
  }
}

const themeToggleBtn = document.getElementById('btn-theme-toggle');
if (themeToggleBtn) {
  themeToggleBtn.addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
    applyTheme(next);
    try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* ignore */ }
  });
}
applyTheme(document.documentElement.dataset.theme || 'dark');

// -------------------------------------------------------------------------
// Download-modal buttons outside the modal itself (view page headers)
// -------------------------------------------------------------------------
document.querySelectorAll('[data-opens-download-modal]').forEach(btn => {
  btn.addEventListener('click', () => {
    if (typeof openDownloadModal === 'function') openDownloadModal();
  });
});

// -------------------------------------------------------------------------
// Rail badges - counts that justify a view's existence without opening it.
// Reads the shared state objects (utils.js) that the existing pollers already
// maintain, so this adds no extra network traffic.
// -------------------------------------------------------------------------
function updateRailBadges() {
  const instEl = document.getElementById('rail-inst-count');
  if (instEl) {
    const n = Object.values(instances || {}).filter(i => i.status !== 'stopped').length;
    instEl.textContent = n;
    instEl.hidden = n === 0;
  }
  const dlEl = document.getElementById('rail-dl-count');
  if (dlEl) {
    const active = Object.values(downloads || {}).filter(d =>
      d.status === 'downloading' || d.status === 'moving' || d.status === 'queued');
    dlEl.textContent = active.length;
    dlEl.hidden = active.length === 0;
  }
}
setInterval(updateRailBadges, 2000);

// Initial routing + badges
showView(currentView());
updateRailBadges();
