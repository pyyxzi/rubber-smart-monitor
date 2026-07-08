const $ = id => document.getElementById(id);
const pct = v => v != null ? (v * 100).toFixed(2) + '%' : 'N/A';
const today = () => new Date().toISOString().slice(0, 10);
const CLS = ['无病(0)', '轻度(1)', '中度(2)', '重度(3)'];
let figsLoaded = false;
let lastPrediction = null;

/* ── 页面导航 ──────────────────────────────────────────────────────────────── */
function go(name, el) {
  document.querySelectorAll('.pg').forEach(p => p.classList.remove('on'));
  document.querySelectorAll('#sidebar a').forEach(a => a.classList.remove('on'));
  $('pg-' + name).classList.add('on');
  el.classList.add('on');
  if (name === 'figures' && !figsLoaded) { figsLoaded = true; loadFigs(); }
}

/* ── 训练结果 ──────────────────────────────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', async () => {
  const d5 = $('f5'); d5.value = d5.defaultValue = today();
  try {
    const r = await fetch('/api/results');
    if (!r.ok) throw new Error((await r.json()).error);
    const data = await r.json();
    const bm = data.models[data.best_model];
    $('stat-cards').innerHTML = [
      ['最优模型', data.best_model], ['准确率', pct(bm.accuracy)],
      ['宏F1', pct(bm.f1_macro)],   ['宏AUC', pct(bm.auc_macro)],
    ].map(([l, v]) =>
      `<div class="stat-box"><div class="lbl">${l}</div><div class="val">${v}</div></div>`
    ).join('');
    $('tbody').innerHTML = Object.entries(data.models).map(([n, m]) => {
      const best = n === data.best_model;
      return `<tr class="${best ? 'best' : ''}">
        <td>${n}${best ? '<span class="badge">最优</span>' : ''}</td>
        <td>${pct(m.accuracy)}</td><td>${pct(m.f1_macro)}</td><td>${pct(m.f1_weighted)}</td>
        <td>${pct(m.precision_macro)}</td><td>${pct(m.recall_macro)}</td>
        <td>${m.auc_macro != null ? pct(m.auc_macro) : 'N/A'}</td>
        <td style="color:#999">${m.time ? m.time.toFixed(1) + 's' : '—'}</td></tr>`;
    }).join('');
    $('loading').style.display = 'none';
    $('wrap').style.display = '';
  } catch (e) {
    $('loading').style.display = 'none';
    $('err').style.display = '';
    $('err').textContent = e.message;
  }
});

/* ── 图表展示 ──────────────────────────────────────────────────────────────── */
function makeFigItems(files, prefix) {
  return files.map(f =>
    `<div class="fig-item" onclick="$('lbimg').src='/${prefix}/${f}';$('lightbox').classList.add('on')">
      <img src="/${prefix}/${f}" loading="lazy" alt="">
      <p>${f.replace(/^\d+[a-z]?_/, '').replace(/_/g, ' ').replace(/\.png$/, '')}</p>
    </div>`).join('');
}

async function loadFigs() {
  $('fig-tip').style.display = 'none';
  const [dataFiles, picFiles] = await Promise.all([
    fetch('/api/figures').then(r => r.json()),
    fetch('/api/pic-figures').then(r => r.json()),
  ]);
  if (dataFiles.length) {
    $('figs-data').innerHTML = makeFigItems(dataFiles, 'figures');
    $('fig-data-wrap').style.display = '';
  }
  if (picFiles.length) {
    $('figs-pic').innerHTML = makeFigItems(picFiles, 'pic-figures');
    $('fig-pic-wrap').style.display = '';
  }
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') $('lightbox').classList.remove('on');
});

/* ── 数据预测 ──────────────────────────────────────────────────────────────── */
async function predictData() {
  const btn = $('pbtn');
  btn.disabled = true; btn.textContent = '预测中...';
  try {
    const v = id => +$(id).value;
    const resp = await fetch('/api/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        temperature_2m_mean:       v('f0') || 24.7,
        relative_humidity_mean:    v('f1') || 80.3,
        leaf_wetness_hours:        v('f2') || 8.0,
        precip_sum_7d:             v('f3') || 30.1,
        consecutive_suitable_days: v('f4') || 5,
        obs_date: $('f5').value || today(),
      }),
    });
    const d = await resp.json();
    if (d.error) throw new Error(d.error);
    $('pred-tip').style.display = 'none';
    $('pred-out').style.display = '';
    $('plv').textContent = d.label;
    const dr = d.derived || {};
    $('pinfo').innerHTML =
      `使用模型：${d.model}<br>` +
      `监测日期：${dr.obs_date || '-'} | 月份：${dr.month || '-'}月 | 物候敏感度：${dr.phenology_sensitivity ?? '-'}`;
    $('pbars').innerHTML = CLS.map(l => {
      const p = (d.probs[l] || 0) * 100;
      return `<div class="prob-row">
        <span class="prob-name">${l}</span>
        <div class="prob-track"><div class="prob-fill" style="width:${p.toFixed(1)}%"></div></div>
        <span class="prob-pct">${p.toFixed(1)}%</span></div>`;
    }).join('');
    $('ai-out').style.display = 'none';
    $('aibtn').disabled = false; $('aibtn').textContent = 'AI 诊断分析';
    lastPrediction = {
      temperature: $('f0').value, humidity: $('f1').value,
      leaf_wetness: $('f2').value, precip: $('f3').value,
      consecutive_days: $('f4').value, obs_date: $('f5').value || today(),
      month: dr.month, phenology: dr.phenology_sensitivity,
      label: d.label,
      probs: Object.fromEntries(CLS.map(l => [l, d.probs[l] || 0])),
    };
  } catch (e) { alert('预测失败: ' + e.message); }
  finally { btn.disabled = false; btn.textContent = '开始预测'; }
}

function resetForm() {
  ['f0','f1','f2','f3','f4','f5'].forEach(id => { const e=$(id); e.value=e.defaultValue; });
  $('pred-out').style.display = 'none';
  $('pred-tip').style.display = '';
  lastPrediction = null;
}

/* ── AI诊断 / 防治措施 ────────────────────────────────────────────────────── */
async function aiAnalyze(mode) {
  if (!lastPrediction) return;
  const isAnalyze = mode !== 'prevention';
  const btn = isAnalyze ? $('aibtn') : $('prevbtn');
  const label = isAnalyze ? 'AI 诊断分析' : '防治措施';
  $('aibtn').disabled = true; $('prevbtn').disabled = true;
  btn.innerHTML = label + '<span class="ai-spinner"></span>';
  $('ai-out').style.display = '';
  $('ai-text').textContent = isAnalyze ? '正在生成诊断报告...' : '正在生成防治方案...';
  try {
    const resp = await fetch('/api/ai-analyze', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...lastPrediction, mode }),
    });
    const d = await resp.json();
    if (d.error) throw new Error(d.error);
    $('ai-text').textContent = d.reply;
  } catch (e) {
    $('ai-text').textContent = '请求失败：' + e.message;
  } finally {
    $('aibtn').disabled = false; $('aibtn').textContent = 'AI 诊断分析';
    $('prevbtn').disabled = false; $('prevbtn').textContent = '防治措施';
  }
}

/* ── AI聊天 ───────────────────────────────────────────────────────────────── */
let chatHistory = [];

function appendMsg(role, text) {
  const box = $('chat-msgs');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = `<div class="bubble">${text.replace(/</g, '&lt;')}</div>`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return div;
}

async function sendChat() {
  const input = $('chat-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  $('chat-send').disabled = true;

  appendMsg('user', text);
  chatHistory.push({ role: 'user', content: text });

  const aiDiv = appendMsg('ai', '...');
  try {
    const resp = await fetch('/api/ai-chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: chatHistory }),
    });
    const d = await resp.json();
    if (d.error) throw new Error(d.error);
    aiDiv.querySelector('.bubble').textContent = d.reply;
    chatHistory.push({ role: 'assistant', content: d.reply });
  } catch (e) {
    aiDiv.querySelector('.bubble').textContent = '请求失败：' + e.message;
  } finally {
    $('chat-send').disabled = false;
    input.focus();
  }
}

/* ── 图片检测 ──────────────────────────────────────────────────────────────── */
const dropZone = $('dropZone'), imgFile = $('imgFile');
let selectedFile = null;

['dragenter', 'dragover'].forEach(e =>
  dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.add('drag-over'); }));
['dragleave', 'drop'].forEach(e =>
  dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.remove('drag-over'); }));
dropZone.addEventListener('drop', ev => {
  if (ev.dataTransfer.files.length) handleFile(ev.dataTransfer.files[0]);
});
imgFile.addEventListener('change', () => {
  if (imgFile.files.length) handleFile(imgFile.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  const r = new FileReader();
  r.onload = e => {
    $('previewImg').src = e.target.result;
    $('imgPreview').style.display = 'block';
    $('imgBtnRow').style.display = 'flex';
    $('img-out').style.display = 'none';
    $('img-tip').style.display = '';
  };
  r.readAsDataURL(file);
}

async function detectImage() {
  if (!selectedFile) return;
  const btn = $('imgBtn');
  btn.disabled = true; btn.textContent = '检测中...';
  $('imgSpinner').style.display = 'block';
  $('img-out').style.display = 'none';
  try {
    const fd = new FormData(); fd.append('file', selectedFile);
    const resp = await fetch('/api/predict-image', { method: 'POST', body: fd });
    const d = await resp.json();
    if (d.error) { alert(d.error); return; }
    const tag = $('imgTag');
    tag.className = 'result-tag';
    tag.classList.add(
      d.result_class === 'Healthy' ? 'healthy' :
      d.result_class === 'Unhealthy' ? 'unhealthy' : 'other'
    );
    tag.textContent = d.prediction;
    $('imgConf').textContent = '置信度: ' + d.confidence + '%';
    const cw = $('camWrap');
    if (d.gradcam) { $('camImg').src = d.gradcam; cw.style.display = 'block'; }
    else cw.style.display = 'none';
    $('imgProbs').innerHTML = Object.entries(d.probabilities).map(([name, prob]) =>
      `<div class="bar-item">
        <div class="bar-label"><span>${name}</span><span>${prob}%</span></div>
        <div class="bar-bg"><div class="bar-fg" style="width:${prob}%"></div></div>
      </div>`).join('');
    $('img-tip').style.display = 'none';
    $('img-out').style.display = '';
  } catch (err) { alert('检测失败: ' + err.message); }
  finally { $('imgSpinner').style.display = 'none'; btn.disabled = false; btn.textContent = '开始检测'; }
}

function resetImage() {
  selectedFile = null; imgFile.value = '';
  $('imgPreview').style.display = 'none';
  $('imgBtnRow').style.display = 'none';
  $('img-out').style.display = 'none';
  $('img-tip').style.display = '';
}
