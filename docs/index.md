<!-- Hero -->
<div class="hero">
    <h1>CreativityPrism</h1>
    <p class="sub">A benchmark for creative reasoning in LLMs — Quality · Novelty · Diversity</p>
    <div class="btns">
        <a href="." target="_blank" rel="noopener">📄 arXiv (coming soon!)</a>
        <a href="https://www.kaggle.com/datasets/a916e717ce0375e9ca3e31a53d1f4b37f29266057027e3bfb0140b31cc6dcc21" target="_blank" rel="noopener">🗂 data</a>
        <a href="https://github.com/joeyhou/creativityprism" target="_blank" rel="noopener">💻 code</a>
    </div>
</div>

<!-- KPIs (edit the numbers or compute them later) -->
<div class="kpis">
    <div class="card"><div class="num">17</div><div class="label">LLMs</div></div>
    <div class="card"><div class="num">9</div><div class="label">Datasets</div></div>
    <div class="card"><div class="num">3</div><div class="label">Dimensions</div></div>
    <div class="card"><div class="num">21</div><div class="label">Metrics</div></div>
</div>


## What is CreativityPrism?
INTRODUCTION

---

<div class="section">
    <h2 id="overview">Overview</h2>
    <figure>
        <img src="assets/img/CreativePrism_light.png" alt="CreativityPrism overview diagram">
        <figcaption>Domains, datasets, and metric dimensions.</figcaption>
    </figure>
</div>



<div class="section leaderboard-card">
  <div class="lb-header">
    <div>
      <h2 id="leaderboard">Leaderboard</h2>
      <div id="updated" class="lb-updated">Last updated: —</div>
    </div>
  </div>

  <table id="board" class="paper-table" style="width:100%">
    <thead>
      <tr>
        <th class="th-model" data-key="model" data-type="text">Model</th>
        <th data-key="overall" data-type="num">Overall</th>
        <th data-key="quality" data-type="num">Quality</th>
        <th data-key="novelty" data-type="num">Novelty</th>
        <th data-key="diversity" data-type="num">Diversity</th>
        <th data-key="cw" data-type="num">Creative&nbsp;Writing</th>
        <th data-key="dt" data-type="num">Divergent&nbsp;Thinking</th>
        <th data-key="lr" data-type="num">Logical&nbsp;Reasoning</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>
</div>

---

<div class="section">
    <h2 id="headline-results">Overall Results</h2>
    <figure>
        <img src="assets/img/plot-perf-overall.png" alt="Overall performance bar chart">
        <!-- <figcaption> Overall performance across evaluated models.</figcaption> -->
    </figure>
</div>

## Detailed Results
=== "Results by Creativity Dimension"

    === "Quality"
        <figure>
          <img src="assets/img/plot-perf-quality.png" alt="Quality performance">
        </figure>

    === "Novelty"
        <figure>
          <img src="assets/img/plot-perf-novelty.png" alt="Novelty performance">
        </figure>

    === "Diversity"
        <figure>
          <img src="assets/img/plot-perf-diversity.png" alt="Diversity performance">
        </figure>

=== "Results by Domain"

    === "Creative writing"
        <figure>
          <img src="assets/img/plot-perf-story.png" alt="Creative writing performance">
        </figure>

    === "Divergent thinking"
        <figure>
          <img src="assets/img/plot-perf-psyc.png" alt="Divergent thinking performance">
        </figure>

    === "Logical reasoning"
        <figure>
          <img src="assets/img/plot-perf-logical.png" alt="Logical reasoning performance">
        </figure>


<script>
(async function(){
  const url = 'assets/data/leaderboard.json';
  const res = await fetch(url, { cache: 'no-cache' });
  const data = await res.json();

  // Update timestamp
  document.getElementById('updated').textContent =
    'Last updated: ' + (data.meta?.last_updated ?? '—');

  // Helpers
  const fmt3 = (x)=> (x ?? 0).toFixed(3).replace(/^0(?=\.)/, '');  // -> ".721"
  const cols = ['overall','quality','novelty','diversity','cw','dt','lr'];
  const groupsOrder = data.group_order ?? Array.from(new Set((data.rows||[]).map(r=>r.group)));

  // Build grouped structure
  const groups = groupsOrder.map(g => ({
    name: g,
    rows: (data.rows || []).filter(r => r.group === g)
  }));

  // Compute max per group/column (for bold)
  function computeMaxByGroup(groups) {
    const maxByGroup = {};
    for (const g of groups) {
      maxByGroup[g.name] = {};
      for (const c of cols) {
        maxByGroup[g.name][c] = Math.max(...g.rows.map(r => +r[c] || -Infinity));
      }
    }
    return maxByGroup;
  }

  // Render function
  function render(sortKey=null, dir='desc') {
    const tbody = document.querySelector('#board tbody');
    tbody.innerHTML = '';

    // sort per group if a key is given
    if (sortKey) {
      for (const g of groups) {
        g.rows.sort((a,b) => {
          const va = a[sortKey], vb = b[sortKey];
          if (va == null && vb == null) return 0;
          if (va == null) return 1;
          if (vb == null) return -1;
          if (typeof va === 'number' && typeof vb === 'number') {
            return dir === 'asc' ? va - vb : vb - va;
          }
          // text sort
          return dir === 'asc' ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
        });
      }
    }

    const maxByGroup = computeMaxByGroup(groups);

    for (const g of groups) {
      // Group header row (full-span)
      const gh = document.createElement('tr');
      gh.className = 'group-row';
      gh.innerHTML = `<td colspan="8">${g.name}</td>`;
      tbody.appendChild(gh);

      // Data rows
      for (const r of g.rows) {
        const tr = document.createElement('tr');
        tr.innerHTML = [
          `<td class="model">${r.model}</td>`,
          ...cols.map(c => {
            const v = +r[c];
            const s = isFinite(v) ? fmt3(v) : '';
            const isBest = isFinite(v) && v === maxByGroup[g.name][c];
            return `<td class="num ${isBest?'best':''}">${isBest ? '<strong>'+s+'</strong>' : s}</td>`;
          })
        ].join('');
        tbody.appendChild(tr);
      }
    }
  }

  // Initial render (paper order)
  render();

  // Header click sorting (within groups)
  const ths = Array.from(document.querySelectorAll('#board thead th'));
  const state = { key: null, dir: 'desc' };
  ths.forEach(th => {
    const key = th.dataset.key;
    const type = th.dataset.type || 'num';
    if (!key) return;

    th.classList.add('sortable');
    th.addEventListener('click', () => {
      if (state.key === key) {
        state.dir = (state.dir === 'desc') ? 'asc' : 'desc';
      } else {
        state.key = key;
        state.dir = (type === 'text') ? 'asc' : 'desc'; // sensible default
      }

      // visual indicator
      ths.forEach(t => t.classList.remove('sorted-asc','sorted-desc'));
      th.classList.add(state.dir === 'asc' ? 'sorted-asc' : 'sorted-desc');

      render(state.key, state.dir);
    });
  });
})();
</script>
