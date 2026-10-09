// Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Model archive (core/archive.py): Archive / Restore buttons in the model
// library and the moves' progress in the Downloads panel.
//
// Only shown when the node has ARCHIVE_DIR set. Moves are this node's: the
// archive volume is mounted on it, so in cluster mode the buttons appear for
// the local node's library (the Target node being this node).
// -------------------------------------------------------------------------

let archiveState = { enabled: false, available: false, reason: '', jobs: [] };
let _archiveJobStatus = {};   // job id -> last seen status (to refresh models on completion)

function _archiveLocalTarget() {
  const cs = window.clusterState;
  if (!(cs && cs.enabled)) return true;
  const target = (typeof getLaunchNode === 'function') ? getLaunchNode() : null;
  return !target || target === cs.self_id;
}

function _relUnder(path, base) {
  return (base && path && path.startsWith(base + '/')) ? path.slice(base.length + 1) : null;
}

// The queued/moving job that covers this model (archive side or restore side).
function archiveJobFor(m) {
  for (const job of archiveState.jobs || []) {
    if (job.status !== 'queued' && job.status !== 'moving') continue;
    for (const base of [job.src_base, job.dst_base]) {
      const rel = _relUnder(m.path, base);
      if (rel == null) continue;
      if ((job.unit || []).some(u => rel === u || rel.startsWith(u + '/'))) return job;
    }
  }
  return null;
}

function archiveButtonHtml(m) {
  if (!archiveState.enabled || m.engine || !_archiveLocalTarget()) return '';
  const job = archiveJobFor(m);
  if (job) {
    const verb = job.kind === 'archive' ? 'Archiving' : 'Restoring';
    return `<button class="btn-archive-model btn-archive-busy" disabled title="${verb}… (see Downloads)"><span class="spinner"></span></button>`;
  }
  if (m.archived) {
    return `<button class="btn-restore-model" title="Restore to the models volume"><i class="fa-solid fa-box-open"></i></button>`;
  }
  const off = !archiveState.available;
  const title = off ? `Archive unavailable: ${archiveState.reason}` : 'Move to the archive volume';
  return `<button class="btn-archive-model" title="${escHtml(title)}"${off ? ' disabled' : ''}><i class="fa-solid fa-box-archive"></i></button>`;
}

async function _startArchiveMove(kind, m) {
  const url = kind === 'archive' ? '/api/archive' : '/api/archive/restore';
  try {
    const res = await apiFetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: m.path }),
    });
    const data = await readApiResponse(res);
    if (res.ok) {
      toast(`${kind === 'archive' ? 'Archiving' : 'Restoring'} ${data.name} - progress in Downloads`, 'success');
      await pollDownloads();
      renderModels();
    } else {
      toast(`${kind === 'archive' ? 'Archive' : 'Restore'} failed: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}

async function archiveModel(m) {
  const ok = await showConfirm('Archive model',
    `Move ${m.name} (and the rest of its folder) to the archive volume? It can't be launched until it is restored. Presets are kept.`);
  if (ok) await _startArchiveMove('archive', m);
}

async function restoreModel(m) {
  await _startArchiveMove('restore', m);
}

// Called from pollDownloads: fetch this node's archive state and add its
// moves to the Downloads panel's map.
async function mergeArchiveJobs(map) {
  let data;
  try {
    const res = await apiFetch('/api/archive', (typeof pollOpts === 'function') ? pollOpts() : undefined);
    if (!res || !res.ok) return;
    data = await res.json();
  } catch (e) {
    return;
  }
  const wasEnabled = archiveState.enabled;
  const prevActive = (archiveState.jobs || []).filter(j => j.status === 'queued' || j.status === 'moving').length;
  archiveState = data;
  let finished = false;
  (data.jobs || []).forEach(job => {
    const prev = _archiveJobStatus[job.id];
    if (prev && prev !== job.status && !['queued', 'moving'].includes(job.status)) {
      finished = true;
      if (job.status === 'completed') {
        toast(`${job.kind === 'archive' ? 'Archived' : 'Restored'} ${job.name}`
          + (job.warning ? ` (${job.warning})` : ''), job.warning ? 'info' : 'success');
      } else if (job.status === 'failed') {
        toast(`${job.kind === 'archive' ? 'Archive' : 'Restore'} of ${job.name} failed: ${job.error}`, 'error');
      }
    }
    _archiveJobStatus[job.id] = job.status;
    const cs = window.clusterState;
    map[`arch-${job.id}`] = {
      ...job,
      id: `arch-${job.id}`,
      job_id: job.id,
      _archive: true,
      started_at: job.created_at,
      _node_name: (cs && cs.selfName) || null,
    };
  });
  const nowActive = (data.jobs || []).filter(j => j.status === 'queued' || j.status === 'moving').length;
  if (finished) {
    if (typeof loadModels === 'function') loadModels();
  } else if (wasEnabled !== data.enabled || prevActive !== nowActive) {
    if (typeof renderModels === 'function') renderModels();
  }
}

function _fmtBytes(n) {
  if (!n) return '0 B';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i >= 3 ? 1 : 0)} ${u[i]}`;
}

function updateArchiveItem(item, job) {
  item.dataset.id = job.id;
  item.classList.add('dl-item-archive');
  const top = item.querySelector('.dl-item-top');
  let spinner = top.querySelector('.spinner');
  if (job.status === 'moving') {
    if (!spinner) {
      spinner = document.createElement('span');
      spinner.className = 'spinner';
      top.prepend(spinner);
    }
  } else if (spinner) {
    spinner.remove();
  }

  const verb = job.kind === 'archive' ? 'Archive' : 'Restore';
  const icon = job.kind === 'archive' ? 'fa-box-archive' : 'fa-box-open';
  const name = item.querySelector('.dl-item-name');
  name.innerHTML = `<i class="fa-solid ${icon}"></i> ${escHtml(verb)}: ${escHtml(job.name)}`;
  name.title = `${job.src_base} → ${job.dst_base}`;

  const status = item.querySelector('.dl-status');
  status.textContent = job.status;
  status.className = `dl-status dl-status-${job.status === 'moving' ? 'downloading' : job.status}`;

  let prog = item.querySelector('.arch-progress');
  if (!prog) {
    prog = document.createElement('div');
    prog.className = 'arch-progress meta';
    item.insertBefore(prog, item.querySelector('.dl-item-actions'));
  }
  const total = job.bytes_total || 0;
  const done = job.bytes_done || 0;
  const pct = total > 0 ? Math.min(100, Math.round(done * 100 / total)) : (job.status === 'completed' ? 100 : 0);
  let text = `${_fmtBytes(done)} / ${_fmtBytes(total)}`;
  if (job.status === 'moving' && job.speed > 0) {
    const eta = Math.max(0, Math.round((total - done) / job.speed));
    text += ` · ${_fmtBytes(job.speed)}/s · ${eta >= 60 ? Math.round(eta / 60) + ' min' : eta + ' s'} left`;
  }
  if (job.method === 'rename') text = `${_fmtBytes(total)} · moved instantly (same disk)`;
  if (job.error) text += ` · ${job.error}`;
  if (job.warning) text += ` · ${job.warning}`;
  prog.innerHTML = `
    <div class="arch-progress-bar"><div style="width:${pct}%"></div></div>
    <span class="arch-progress-text${job.error ? ' text-danger' : ''}">${escHtml(text)}</span>`;

  const actions = item.querySelector('.dl-item-actions');
  actions.innerHTML = '';
  const mk = (cls, html) => {
    const b = document.createElement('button');
    b.className = cls;
    b.dataset.job = job.job_id;
    b.innerHTML = html;
    actions.appendChild(b);
  };
  if (job.status === 'queued' || job.status === 'moving') {
    mk('btn-xs danger btn-arch-cancel', '<i class="fa-solid fa-ban"></i> Cancel');
  } else {
    mk('btn-xs danger btn-arch-remove', '<i class="fa-solid fa-trash"></i> Remove');
  }
}

function bindArchiveItemButtons(panel) {
  panel.querySelectorAll('.btn-arch-cancel').forEach(btn => {
    btn.addEventListener('click', async () => {
      const ok = await showConfirm('Cancel move', 'Stop this move? The model stays where it was and the partial copy is deleted.');
      if (ok) await _archiveJobAction(btn.dataset.job, 'Move cancelled');
    });
  });
  panel.querySelectorAll('.btn-arch-remove').forEach(btn => {
    btn.addEventListener('click', () => _archiveJobAction(btn.dataset.job, 'Removed from the list'));
  });
}

async function _archiveJobAction(jobId, msg) {
  try {
    const res = await apiFetch(`/api/archive/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
    if (res.ok) toast(msg, 'info');
    await pollDownloads();
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}
