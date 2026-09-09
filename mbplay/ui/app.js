'use strict';

const $ = (id) => document.getElementById(id);
const api = async (path, opts) => {
  const resp = await fetch(path, opts);
  if (!resp.ok) {
    let detail = await resp.text();
    try { detail = JSON.parse(detail).detail || detail; } catch (e) { /* текст как есть */ }
    throw new Error(detail);
  }
  return resp.status === 204 ? null : resp.json();
};
const post = (path, body) => api(path, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}),
});

const fmt = (v, digits = 3) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : Number(v).toFixed(digits);
const pct = (v) => (v === null || v === undefined) ? '—' : (Number(v) * 100).toFixed(1) + '%';
const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const bits = (nats) => nats === null || nats === undefined ? null : nats / Math.LN2;

let META = null;
let CORPORA = [];

/* ---------- вкладки ---------- */
document.querySelectorAll('nav button').forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll('nav button').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    btn.classList.add('active');
    $('tab-' + btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'runs') loadRuns();
    if (btn.dataset.tab === 'repair') loadBaseRuns();
  };
});

/* ---------- график ---------- */
function chart(series, opts) {
  opts = opts || {};
  const W = 640, H = 200, padL = 46, padR = 12, padT = 12, padB = 24;
  const all = series.flatMap((s) => s.points);
  if (!all.length) return '<p class="note">нет точек</p>';
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (opts.yFloor !== undefined) y0 = Math.min(y0, opts.yFloor);
  if (x1 === x0) x1 = x0 + 1;
  const pad = (y1 - y0) * 0.08 || 0.1;
  y0 -= pad; y1 += pad;
  const X = (x) => padL + (x - x0) / (x1 - x0) * (W - padL - padR);
  const Y = (y) => H - padB - (y - y0) / (y1 - y0) * (H - padT - padB);

  const grid = [0, 0.25, 0.5, 0.75, 1].map((f) => {
    const y = y0 + f * (y1 - y0);
    return `<line x1="${padL}" y1="${Y(y).toFixed(1)}" x2="${W - padR}" y2="${Y(y).toFixed(1)}" stroke="#2a2f3a"/>
            <text x="4" y="${(Y(y) + 4).toFixed(1)}" fill="#949bab" font-size="10">${y.toFixed(2)}</text>`;
  }).join('');

  const paths = series.map((s) => {
    const d = s.points.map((p, i) => (i ? 'L' : 'M') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1)).join(' ');
    const dash = s.dashed ? ' stroke-dasharray="4 3"' : '';
    return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="1.6"${dash}/>`;
  }).join('');

  const legend = series.map((s) =>
    `<span><i style="background:${s.color}"></i>${esc(s.label)}</span>`).join('');

  return `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    ${grid}${paths}
    <text x="${W - padR}" y="${H - 6}" fill="#949bab" font-size="10" text-anchor="end">${x1} шагов</text>
  </svg><div class="legend">${legend}</div>`;
}

/* ---------- опрос задачи ---------- */
async function runJob(kind, body, barId, statusId, onDone) {
  const bar = $(barId), status = $(statusId);
  status.className = 'status';
  status.textContent = 'отправляем…';
  bar.style.width = '0';
  let job;
  try {
    job = await post('/api/experiments/' + kind, body);
  } catch (e) {
    status.className = 'status failed';
    status.textContent = 'ошибка запуска: ' + e.message;
    return;
  }
  const poll = setInterval(async () => {
    let j;
    try { j = await api('/api/jobs/' + job.job_id); } catch (e) { return; }
    bar.style.width = (j.progress * 100).toFixed(1) + '%';
    status.textContent = j.message || j.status;
    if (j.status === 'done') {
      clearInterval(poll);
      status.textContent = 'готово: ' + j.run_id;
      const run = await api('/api/runs/' + j.run_id);
      onDone(run);
      loadBaseRuns();
    } else if (j.status === 'failed') {
      clearInterval(poll);
      status.className = 'status failed';
      status.textContent = 'ошибка: ' + j.error;
    }
  }, 1000);
}

/* ---------- общие блоки результата ---------- */
function sourceBlock(run) {
  const s = run.source || {};
  const cls = (s.data_class || '').toLowerCase();
  return `<div class="card"><h2>Прогон ${esc(run.run_id)}</h2>
    <dl class="kv">
      <dt>Данные</dt><dd>${esc(s.title || '—')} <span class="tag ${cls}">${esc(s.data_class || '?')}</span></dd>
      <dt>Основание</dt><dd>${esc(s.license || '—')}</dd>
      <dt>Символов</dt><dd>${s.used_chars ?? '—'}</dd>
      <dt>sha256 текста</dt><dd>${esc((s.used_sha256 || '').slice(0, 16))}</dd>
      <dt>Код MagicBrain</dt><dd>${esc((run.code || {}).magicbrain_sha || '?')}${(run.code || {}).magicbrain_dirty === 'dirty' ? ' (dirty)' : ''}</dd>
      <dt>Время</dt><dd>${fmt(run.wall_time_sec, 1)} с</dd>
    </dl>
    <button class="small" onclick="location.href='/api/runs/${esc(run.run_id)}/export'">Скачать JSON</button>
  </div>`;
}

function networkBlock(net) {
  if (!net) return '';
  const dh = net.delay_histogram || {};
  return `<div class="card"><h2>Сеть, выросшая из генома</h2>
    <dl class="kv">
      <dt>Геном</dt><dd>${esc(net.genome)}</dd>
      <dt>Нейронов N</dt><dd>${net.N}</dd>
      <dt>Связность K</dt><dd>${net.K}</dd>
      <dt>Рёбер</dt><dd>${net.edges}</dd>
      <dt>Активных за шаг</dt><dd>${net.k_active} (${pct(net.target_rate)})</dd>
      <dt>Тормозных</dt><dd>${pct(net.p_inhib)}</dd>
      <dt>Рёбер по задержкам</dt><dd>${[1, 2, 3, 4, 5].map((d) => d + ':' + (dh[d] ?? 0)).join('  ')}</dd>
    </dl></div>`;
}

function caveatsBlock(run) {
  if (!run.caveats || !run.caveats.length) return '';
  return `<div class="card"><h2>Границы результата</h2>
    <ul class="caveats">${run.caveats.map((c) => `<li>${esc(c)}</li>`).join('')}</ul></div>`;
}

/* ---------- обучение ---------- */
function renderTrain(run) {
  const m = run.metrics;
  const b = m.baselines_bits;
  const verdict = m.beats_bigram
    ? '<span class="verdict ok">лучше биграммы</span>'
    : (m.beats_unigram ? '<span class="verdict">лучше униграммы, но хуже биграммы</span>'
                       : '<span class="verdict bad">хуже униграммы</span>');
  const series = [
    { label: 'train loss (окно)', color: '#6ea8fe', points: run.curve.map((p) => [p.step, p.loss]) },
  ];
  if (run.holdout_curve && run.holdout_curve.length) {
    series.push({ label: 'отложенный хвост', color: '#59c98a', points: run.holdout_curve.map((p) => [p.step, p.holdout_nats]) });
  }
  series.push({
    label: 'биграмма', color: '#e8b451', dashed: true,
    points: [[run.curve[0].step, m.baselines_nats.bigram], [run.curve[run.curve.length - 1].step, m.baselines_nats.bigram]],
  });

  const sub = [
    { label: 'частота разрядов', color: '#6ea8fe', points: run.curve.map((p) => [p.step, p.firing_rate]) },
    { label: 'средняя |W|', color: '#e07a7a', points: run.curve.map((p) => [p.step, p.mean_abs_w]) },
  ];

  $('train-result').innerHTML = sourceBlock(run) + `
    <div class="card"><h2>Результат на отложенном хвосте</h2>
      <table>
        <tr><th>Модель</th><th class="num">бит/символ</th><th class="num">нат/символ</th></tr>
        <tr><td>равномерная</td><td class="num">${fmt(b.uniform)}</td><td class="num">${fmt(m.baselines_nats.uniform)}</td></tr>
        <tr><td>униграмма</td><td class="num">${fmt(b.unigram)}</td><td class="num">${fmt(m.baselines_nats.unigram)}</td></tr>
        <tr><td>биграмма</td><td class="num">${fmt(b.bigram)}</td><td class="num">${fmt(m.baselines_nats.bigram)}</td></tr>
        <tr><td>сеть до обучения</td><td class="num">${fmt(m.holdout_bits_before)}</td><td class="num">${fmt(m.holdout_nats_before)}</td></tr>
        <tr><td><b>сеть после обучения</b></td><td class="num"><b>${fmt(m.holdout_bits_after)}</b></td><td class="num"><b>${fmt(m.holdout_nats_after)}</b></td></tr>
      </table>
      <p style="margin:10px 0 0">Вердикт: ${verdict}. Скорость: ${fmt(m.steps_per_second, 0)} шагов/с.</p>
    </div>
    <div class="card"><h2>Кривая обучения</h2>${chart(series)}</div>
    <div class="card"><h2>Состояние субстрата</h2>${chart(sub)}</div>
    <div class="card"><h2>Сгенерированный образец</h2><pre>${esc(run.sample)}</pre>
      <button class="small" style="margin-top:10px" onclick="publishModel('${esc(run.run_id)}')">Отдать модель в magicbrain-api</button>
      <div class="status" id="publish-status"></div>
    </div>` + networkBlock(run.network);
}

async function publishModel(runId) {
  const el = $('publish-status');
  el.textContent = 'регистрируем…';
  try {
    const r = await post(`/api/runs/${runId}/publish`);
    const s = await post('/api/sample', { model_id: r.model_id, seed_text: 'The ', n_tokens: 120 });
    el.textContent = 'magicbrain-api сгенерировал: ' + (s.generated_text || '').slice(0, 160);
  } catch (e) {
    el.className = 'status failed';
    el.textContent = 'не вышло: ' + e.message;
  }
}

/* ---------- нейрогенез ---------- */
function renderNeurogenesis(run) {
  const stages = (run.stages || []).map((st) => {
    const rows = Object.entries(st.metrics || {}).map(([k, v]) => {
      const val = (typeof v === 'object' && v !== null)
        ? `<pre style="max-height:160px">${esc(JSON.stringify(v, null, 1))}</pre>`
        : esc(typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(4)) : v);
      return `<dt>${esc(k)}</dt><dd>${val}</dd>`;
    }).join('');
    const text = st.text ? `<pre>${esc(st.text.slice(0, 1200))}</pre>` : '';
    return `<div class="stage"><div class="head"><b>${esc(st.title)}</b><span>${fmt(st.seconds, 2)} с</span></div>
      <dl class="kv">${rows}</dl>${text}</div>`;
  }).join('');

  const series = [{ label: 'train loss', color: '#6ea8fe', points: run.curve.map((p) => [p.step, p.loss]) }];
  $('ng-result').innerHTML = sourceBlock(run) +
    `<div class="card"><h2>Стадии пайплайна</h2><div class="stages">${stages}</div></div>
     <div class="card"><h2>Кривая обучения развитой сети</h2>${chart(series)}</div>` +
    networkBlock(run.network) + caveatsBlock(run);
}

/* ---------- самовосстановление ---------- */
function renderRepair(run) {
  const m = run.metrics;
  const colors = ['#6ea8fe', '#59c98a', '#e8b451', '#e07a7a', '#b48ee0'];
  const series = m.by_damage.map((d, i) => ({
    label: `повреждение ${pct(d.damage_frac)}`,
    color: colors[i % colors.length],
    points: [[0, d.post_damage_nats]].concat(d.recovery_curve.map((p) => [p.step, p.holdout_nats])),
  }));
  series.push({
    label: 'до повреждения', color: '#949bab', dashed: true,
    points: [[0, m.pre_damage_nats], [Math.max(...m.by_damage.map((d) => d.recovery_curve.at(-1)?.step || 1)), m.pre_damage_nats]],
  });

  const rows = m.by_damage.map((d) => `<tr>
    <td class="num">${pct(d.damage_frac)}</td>
    <td class="num">${d.weights_zeroed}</td>
    <td class="num">${fmt(d.pre_damage_nats)}</td>
    <td class="num">${fmt(d.post_damage_nats)}</td>
    <td class="num">${fmt(d.final_nats)}</td>
    <td class="num">${d.recovery_ratio === null ? '—' : pct(d.recovery_ratio)}</td>
    <td class="num">${fmt(d.residual_nats)}</td></tr>`).join('');

  $('rp-result').innerHTML = sourceBlock(run) + `
    <div class="card"><h2>Восстановление после повреждения</h2>
      <table>
        <tr><th class="num">доля</th><th class="num">весов обнулено</th><th class="num">до</th>
            <th class="num">после удара</th><th class="num">после восстановления</th>
            <th class="num">восстановлено</th><th class="num">остаточный ущерб</th></tr>
        ${rows}
      </table>
      <p class="note" style="margin-top:8px">Все значения — нат/символ на отложенном хвосте.
      «Восстановлено» = какая доля просадки отыграна дообучением.</p>
    </div>
    <div class="card"><h2>Кривые восстановления</h2>${chart(series)}</div>` +
    networkBlock(run.network) + caveatsBlock(run);
}

/* ---------- ACT ---------- */
function renderAct(run) {
  const a = run.sides.act, n = run.sides.numpy, mb = run.microbenchmark, m = run.metrics;
  const series = [
    { label: 'numpy', color: '#6ea8fe', points: n.curve.map((p) => [p.step, p.loss]) },
    { label: 'ACT', color: '#59c98a', points: a.curve.map((p) => [p.step, p.loss]) },
  ];
  const drift = [
    { label: 'дрейф компенсации w_slow (ACT)', color: '#59c98a', points: a.act_snapshots.map((s) => [s.step, s.w_slow_compensation_ratio]) },
  ];
  $('act-result').innerHTML = sourceBlock(run) + `
    <div class="card"><h2>Обучение: одинаковые условия, разная арифметика</h2>
      <table>
        <tr><th></th><th class="num">numpy</th><th class="num">ACT</th></tr>
        <tr><td>ACT реально активен</td><td class="num">${n.act_active}</td><td class="num">${a.act_active}</td></tr>
        <tr><td>отложенный хвост, нат/символ</td><td class="num">${fmt(n.holdout_nats_after)}</td><td class="num">${fmt(a.holdout_nats_after)}</td></tr>
        <tr><td>train loss в конце</td><td class="num">${fmt(n.train_loss_last)}</td><td class="num">${fmt(a.train_loss_last)}</td></tr>
        <tr><td>время, с</td><td class="num">${fmt(n.seconds, 2)}</td><td class="num">${fmt(a.seconds, 2)}</td></tr>
        <tr><td>шагов/с</td><td class="num">${fmt(n.steps_per_second, 0)}</td><td class="num">${fmt(a.steps_per_second, 0)}</td></tr>
      </table>
      <p style="margin:10px 0 0">ACT медленнее в <b>${fmt(m.act_slowdown_x, 2)}×</b>,
      разница качества <b>${fmt(m.holdout_delta_nats, 4)}</b> нат/символ
      (${fmt((m.holdout_delta_relative || 0) * 100, 2)}% — знак «минус» значит, что с ACT лучше).</p>
    </div>
    <div class="card"><h2>Микробенчмарк: где компенсация действительно решает</h2>
      <table>
        <tr><th>операция</th><th class="num">ошибка numpy</th><th class="num">ошибка ACT</th><th class="num">во сколько раз точнее</th></tr>
        <tr><td>сумма ${mb.sum.n} плохо обусловленных чисел</td>
            <td class="num">${mb.sum.naive_abs_error.toExponential(3)}</td>
            <td class="num">${mb.sum.act_abs_error.toExponential(3)}</td>
            <td class="num">${mb.sum.act_abs_error > 0 ? fmt(mb.sum.naive_abs_error / mb.sum.act_abs_error, 1) : '∞'}</td></tr>
        <tr><td>скалярное произведение (${mb.dot.n})</td>
            <td class="num">${mb.dot.naive_abs_error.toExponential(3)}</td>
            <td class="num">${mb.dot.act_abs_error.toExponential(3)}</td>
            <td class="num">${mb.dot.act_abs_error > 0 ? fmt(mb.dot.naive_abs_error / mb.dot.act_abs_error, 1) : '∞'}</td></tr>
      </table>
      <p class="note" style="margin-top:8px">Эталон — math.fsum. Данные подобраны так, чтобы наивная сумма ошибалась:
      на весах сети масштаба N≈1000 такого разброса величин нет, поэтому в таблице выше разницы почти не видно.</p>
    </div>
    <div class="card"><h2>Кривые обучения</h2>${chart(series)}</div>
    <div class="card"><h2>Дрейф компенсированной суммы весов</h2>${chart(drift)}</div>` +
    caveatsBlock(run);
}

const RENDER = { train: renderTrain, neurogenesis: renderNeurogenesis, 'self-repair': renderRepair, act: renderAct };

/* ---------- кнопки ---------- */
$('train-go').onclick = () => {
  const text = $('train-text').value.trim();
  runJob('train', {
    corpus_id: text ? null : $('train-corpus').value,
    text: text || null,
    steps: +$('train-steps').value,
    max_chars: +$('train-chars').value,
    genome: $('train-genome').value.trim(),
    seed: $('train-seed').value ? +$('train-seed').value : null,
    use_act: $('train-act').value === 'true',
  }, 'train-bar', 'train-status', renderTrain);
};

$('ng-go').onclick = () => runJob('neurogenesis', {
  corpus_id: $('ng-corpus').value,
  strategy: $('ng-strategy').value,
  genome_length: +$('ng-len').value,
  steps: +$('ng-steps').value,
  max_chars: +$('ng-chars').value,
  use_cppn: $('ng-cppn').value === 'true',
  attractor_probes: +$('ng-probes').value,
}, 'ng-bar', 'ng-status', renderNeurogenesis);

$('rp-go').onclick = () => {
  const base = $('rp-base').value;
  runJob('self-repair', {
    base_run_id: base || null,
    corpus_id: base ? null : $('rp-corpus').value,
    pretrain_steps: +$('rp-pre').value,
    recovery_steps: +$('rp-rec').value,
    damage_fracs: $('rp-fracs').value.split(',').map((s) => parseFloat(s.trim())).filter((x) => x > 0 && x < 1),
  }, 'rp-bar', 'rp-status', renderRepair);
};

$('act-go').onclick = () => runJob('act', {
  corpus_id: $('act-corpus').value,
  steps: +$('act-steps').value,
  max_chars: +$('act-chars').value,
  seed: +$('act-seed').value,
}, 'act-bar', 'act-status', renderAct);

/* ---------- списки ---------- */
async function loadRuns() {
  const data = await api('/api/runs');
  const tbody = $('runs-table').querySelector('tbody');
  tbody.innerHTML = `<tr><th>прогон</th><th>сценарий</th><th>данные</th><th class="num">с</th><th></th></tr>` +
    data.runs.map((r) => `<tr onclick="openRun('${esc(r.run_id)}')">
      <td>${esc(r.run_id)}</td><td>${esc(r.kind)}</td>
      <td>${esc((r.source || {}).title || '')} <span class="tag ${((r.source || {}).data_class || '').toLowerCase()}">${esc((r.source || {}).data_class || '')}</span></td>
      <td class="num">${fmt(r.wall_time_sec, 1)}</td>
      <td><button class="small" onclick="event.stopPropagation();deleteRun('${esc(r.run_id)}')">удалить</button></td>
    </tr>`).join('');
}

async function openRun(runId) {
  const run = await api('/api/runs/' + runId);
  const render = RENDER[run.kind];
  const target = $('runs-result');
  if (!render) { target.innerHTML = `<div class="card"><pre>${esc(JSON.stringify(run, null, 2))}</pre></div>`; return; }
  const saved = { train: 'train-result', neurogenesis: 'ng-result', 'self-repair': 'rp-result', act: 'act-result' }[run.kind];
  // Рендерим в свою вкладку и показываем её.
  render(run);
  target.innerHTML = `<div class="card"><h2>Открыт прогон ${esc(runId)}</h2>
    <p class="note">Показан на вкладке сценария «${esc(run.kind)}».</p></div>`;
  document.querySelector(`nav button[data-tab="${{ train: 'train', neurogenesis: 'neurogenesis', 'self-repair': 'repair', act: 'act' }[run.kind]}"]`).click();
  void saved;
}

async function deleteRun(runId) {
  await api('/api/runs/' + runId, { method: 'DELETE' });
  loadRuns(); loadBaseRuns();
}

async function loadBaseRuns() {
  const data = await api('/api/runs?kind=train');
  const sel = $('rp-base');
  const current = sel.value;
  sel.innerHTML = '<option value="">— обучить заново —</option>' +
    data.runs.filter((r) => r.has_model).map((r) =>
      `<option value="${esc(r.run_id)}">${esc(r.run_id)}</option>`).join('');
  sel.value = current;
}

async function loadCorpora() {
  const data = await api('/api/corpora');
  CORPORA = data.corpora;
  const opts = CORPORA.map((c) => `<option value="${esc(c.id)}">${esc(c.title)}</option>`).join('');
  ['train-corpus', 'ng-corpus', 'rp-corpus', 'act-corpus'].forEach((id) => { $(id).innerHTML = opts; });
  $('corpora-table').innerHTML =
    `<tr><th>id</th><th>что это</th><th>класс</th><th class="num">символов</th><th class="num">словарь</th><th>основание</th></tr>` +
    CORPORA.map((c) => `<tr onclick="showCorpus('${esc(c.id)}')">
      <td>${esc(c.id)}</td><td>${esc(c.title)}</td>
      <td><span class="tag ${esc(c.data_class.toLowerCase())}">${esc(c.data_class)}</span></td>
      <td class="num">${c.chars}</td><td class="num">${c.vocab_size}</td>
      <td>${esc(c.license)}</td></tr>`).join('');
}

async function showCorpus(id) {
  const data = await api(`/api/corpora/${id}/preview`);
  const c = data.corpus;
  const extra = Object.entries(c).filter(([k]) =>
    !['id', 'title', 'lang', 'kind', 'chars', 'bytes', 'vocab_size', 'sha256', 'data_class', 'license'].includes(k));
  $('corpus-preview').style.display = 'block';
  $('corpus-preview').innerHTML = `<h2>${esc(c.title)}</h2>
    <dl class="kv">
      <dt>sha256</dt><dd>${esc(c.sha256)}</dd>
      ${c.url ? `<dt>источник</dt><dd>${esc(c.url)}</dd>` : ''}
      ${extra.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(JSON.stringify(v))}</dd>`).join('')}
    </dl>
    <pre>${esc(data.preview.slice(0, 2000))}</pre>`;
}

$('runs-refresh').onclick = loadRuns;

/* ---------- старт ---------- */
(async function init() {
  META = await api('/api/meta');
  $('meta').textContent =
    `magicbrain ${META.code.magicbrain_sha}${META.code.magicbrain_dirty === 'dirty' ? '+dirty' : ''} · ` +
    `balansis ${META.code.balansis} · numpy ${META.code.numpy} · режим ${META.disclosure}`;
  if (META.public_mode) {
    document.body.insertAdjacentHTML('afterbegin',
      '<div class="banner" style="margin:12px 20px">Открытый режим: сценарии доступны все, ' +
      'сохранённые прогоны удалять нельзя.</div>');
  }
  await loadCorpora();
  await loadBaseRuns();
  await loadRuns();
})();
