<div class="section leaderboard-card">
  <div class="lb-header">
    <div>
      <h2 id="leaderboard">Leaderboard</h2>
      <div id="updated" class="lb-updated">Last updated: —</div>
    </div>
    <div class="lb-legend">
      <span class="pill gold">🥇 Top 1</span>
      <span class="pill silver">🥈 Top 2</span>
      <span class="pill bronze">🥉 Top 3</span>
    </div>
  </div>

  <table id="board" class="display compact nowrap" style="width:100%">
    <thead>
      <tr>
        <th>Rank</th>
        <th>Model</th>
        <th>Overall</th>
        <th>Quality</th>
        <th>Novelty</th>
        <th>Diversity</th>
        <th>Date</th>
        <th>Paper</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>
</div>

<script>
(async function(){
  const updatedEl = document.getElementById('updated');
  const table = document.getElementById('board');
  const tbody = table.querySelector('tbody');

  function fmt(p){ return (p*100).toFixed(1)+'%'; }

  // 1) Load data with error handling
  let data;
  try {
    const res = await fetch('assets/data/leaderboard.json', { cache: 'no-cache' });
    if (!res.ok) throw new Error('Fetch ' + res.status);
    data = await res.json();
  } catch (e) {
    updatedEl.textContent = 'Last updated: — (failed to load data)';
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;opacity:.7;">Could not load <code>assets/data/leaderboard.json</code>.</td></tr>`;
    console.error('Leaderboard fetch error:', e);
    return;
  }

  updatedEl.textContent = 'Last updated: ' + (data.meta?.last_updated ?? '—');

  // 2) Build rows (also works without DataTables)
  const rows = (data.rows ?? []).map((r,i)=>{
    const rank = i+1, medal = rank===1?'🥇':rank===2?'🥈':rank===3?'🥉':'';
    return [
      `<span class="rank ${rank<=3?'top3':''}">${medal || rank}</span>`,
      r.model ?? '',
      `<span class="num">${fmt(r.overall ?? 0)}</span>`,
      `<span class="num">${fmt(r.quality ?? 0)}</span>`,
      `<span class="num">${fmt(r.novelty ?? 0)}</span>`,
      `<span class="num">${fmt(r.diversity ?? 0)}</span>`,
      `<span class="mono">${r.date ?? ''}</span>`,
      r.paper ? `<a class="pill link" href="${r.paper}" target="_blank" rel="noopener">link</a>` : ''
    ];
  });

  // Fallback: if DataTables isn’t present, render plain HTML rows
  if (typeof window.DataTable === 'undefined') {
    tbody.innerHTML = rows.map(cells => `<tr>${cells.map(c=>`<td>${c}</td>`).join('')}</tr>`).join('');
    console.warn('DataTable not found — rendered plain table.');
    return;
  }

  // 3) Enhance with DataTables
  new DataTable('#board', {
    data: rows,
    order: [[0,'asc']],
    pageLength: 25,
    dom: '<"lb-top"lf>t<"lb-bottom"ip>',
    language: { search: '', lengthMenu: '_MENU_ entries per page' },
    columnDefs: [
      { targets: [0,2,3,4,5], className: 'dt-right' },
      { targets: [1], className: 'dt-left' },
      { targets: [6,7], className: 'dt-center' }
    ]
  });

  const input = document.querySelector('.dataTables_filter input');
  if (input) input.placeholder = 'Search models…';
})();
</script>
