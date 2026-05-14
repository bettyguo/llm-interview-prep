/* llm-interview-prep — site/app.js
 *
 * Single-page app. Five modes:
 *   - browse   : filter + read every Q in the bank
 *   - flash    : flashcards with Leitner-style spaced repetition (localStorage)
 *   - mcq      : auto-generated MCQ from short-answer vs common-mistakes
 *   - mock     : timed mock interview, self-rated, scorecard
 *   - cloze    : key terms blanked in the short answer for active recall
 *
 * Data source: questions.js (sets window.QUESTIONS_DATA). Falls back to
 * fetching questions.json when served over HTTP, so either deployment works.
 */
(function () {
  'use strict';

  // ───────────────────────────────────────────────────────────
  // Bootstrap data
  // ───────────────────────────────────────────────────────────

  const STORE_KEY = 'lip_state_v1';

  function bootstrap() {
    if (window.QUESTIONS_DATA) return Promise.resolve(window.QUESTIONS_DATA);
    return fetch('questions.json').then(r => r.json());
  }

  bootstrap().then(init).catch(err => {
    document.body.innerHTML =
      '<pre style="padding:20px;color:#b23a3a">Failed to load questions data: ' +
      String(err).replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c])) +
      '\n\nIf you opened this file directly (file://) make sure questions.js is in the same folder.</pre>';
  });

  // ───────────────────────────────────────────────────────────
  // Global state
  // ───────────────────────────────────────────────────────────

  let DATA = null;          // {topics, entries, total}
  let ENTRIES = [];         // all entries
  let BY_ID = {};
  let TOPICS = [];          // [{slug, num, title, count}]

  /** Persistent state shape:
   * {
   *   flash: { [id]: { box:1..5, due: epoch_ms, last: epoch_ms } },
   *   mcq:   { [id]: { right: n, wrong: n } },
   *   mock_history: [{ date, topic, difficulty, count, mean, items:[{id, rate}] }]
   * }
   */
  const State = {
    data: { flash: {}, mcq: {}, mock_history: [] },
    load() {
      try {
        const raw = localStorage.getItem(STORE_KEY);
        if (raw) {
          const parsed = JSON.parse(raw);
          this.data = Object.assign({ flash:{}, mcq:{}, mock_history: [] }, parsed);
        }
      } catch (e) { /* ignore */ }
    },
    save() {
      try { localStorage.setItem(STORE_KEY, JSON.stringify(this.data)); }
      catch (e) { /* quota: silently drop */ }
    },
    resetFlash() { this.data.flash = {}; this.save(); },
    resetMcq()   { this.data.mcq   = {}; this.save(); },
  };

  // ───────────────────────────────────────────────────────────
  // Helpers
  // ───────────────────────────────────────────────────────────

  const $  = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  function escapeHtml(s) {
    return String(s).replace(/[<>&"']/g, c => ({
      '<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;',"'":'&#39;'
    }[c]));
  }

  const HAS_MARKED = typeof window.marked === 'object' && window.marked !== null
                     && typeof window.marked.parse === 'function';
  if (HAS_MARKED) {
    try { window.marked.setOptions({ breaks: false, gfm: true, headerIds: false, mangle: false }); }
    catch (e) { /* older versions don't accept all options; safe to ignore */ }
  }
  function md(text) {
    if (!text) return '';
    if (HAS_MARKED) {
      // marked occasionally throws on weird input — guard.
      try { return window.marked.parse(text); }
      catch (e) { return '<p>' + escapeHtml(text) + '</p>'; }
    }
    return '<p>' + escapeHtml(text).replace(/\n\n+/g, '</p><p>').replace(/\n/g, '<br>') + '</p>';
  }

  function shuffleInPlace(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

  function diffOrder(d) {
    return { intro: 1, mid: 2, senior: 3, staff: 4 }[d] || 5;
  }

  function topicShort(slug) {
    const t = TOPICS.find(x => x.slug === slug);
    if (!t) return slug;
    // strip the leading "NN — " from title for compactness
    return t.title.replace(/^\d{2}\s*—\s*/, '');
  }

  function fmtPercent(p) { return Math.round(p * 100) + '%'; }
  function fmtTime(s) {
    s = Math.max(0, Math.floor(s));
    const m = Math.floor(s / 60), ss = s % 60;
    return m + ':' + String(ss).padStart(2, '0');
  }

  function diffPillClass(d) { return 'pill diff-' + (d || 'intro'); }

  function applyMeta(targetIds, e) {
    const t = $(targetIds.topic); if (t) { t.textContent = topicShort(e.topic); t.className = 'pill'; }
    const d = $(targetIds.diff);  if (d) { d.textContent = e.difficulty || ''; d.className = diffPillClass(e.difficulty); }
    if (targetIds.cat) {
      const c = $(targetIds.cat); if (c) { c.textContent = e.category || ''; c.className = 'pill subtle'; }
    }
  }

  // ───────────────────────────────────────────────────────────
  // Init
  // ───────────────────────────────────────────────────────────

  function init(data) {
    DATA = data;
    ENTRIES = data.entries.slice();
    TOPICS = data.topics.slice();
    ENTRIES.forEach(e => { BY_ID[e.id] = e; });
    State.load();

    $('#total-count').textContent = DATA.total + ' Qs';

    setupMode();
    setupBrowse();
    setupFlash();
    setupMcq();
    setupMock();
    setupCloze();
    setupGlobalKeys();
  }

  // ───────────────────────────────────────────────────────────
  // Mode switching
  // ───────────────────────────────────────────────────────────

  function setupMode() {
    const modeFromHash = (location.hash || '').replace(/^#/, '');
    if (modeFromHash && $('#view-' + modeFromHash)) switchMode(modeFromHash);

    $$('.mode-btn').forEach(btn => {
      btn.addEventListener('click', () => switchMode(btn.dataset.mode));
    });
  }

  function switchMode(mode) {
    $$('.mode-btn').forEach(b => {
      const active = b.dataset.mode === mode;
      b.classList.toggle('active', active);
      b.setAttribute('aria-selected', active ? 'true' : 'false');
    });
    $$('.view').forEach(v => v.classList.add('hidden'));
    const view = $('#view-' + mode);
    if (view) view.classList.remove('hidden');
    history.replaceState(null, '', '#' + mode);
    // First-time activation hooks
    if (mode === 'flash')  renderFlash();
    if (mode === 'cloze')  renderCloze();
  }

  // ───────────────────────────────────────────────────────────
  // Browse mode
  // ───────────────────────────────────────────────────────────

  let BROWSE_FILTERS = { search: '', topics: new Set(), diffs: new Set(), cats: new Set() };
  let BROWSE_SORT = 'topic';

  function setupBrowse() {
    // Build filter lists
    const topicBox = $('#topic-filters');
    TOPICS.forEach(t => {
      topicBox.appendChild(makeFilterCheckbox('topic', t.slug, topicShort(t.slug), t.count));
    });
    const diffCounts = countBy(ENTRIES, 'difficulty');
    ['intro', 'mid', 'senior', 'staff'].forEach(d => {
      if (diffCounts[d]) {
        $('#difficulty-filters').appendChild(makeFilterCheckbox('diff', d, d, diffCounts[d]));
      }
    });
    const catCounts = countBy(ENTRIES, 'category');
    Object.keys(catCounts).sort().forEach(c => {
      $('#category-filters').appendChild(makeFilterCheckbox('cat', c, c, catCounts[c]));
    });

    $('#search-input').addEventListener('input', e => {
      BROWSE_FILTERS.search = e.target.value.trim().toLowerCase();
      renderBrowse();
    });

    document.addEventListener('change', e => {
      const t = e.target;
      if (!t.matches('input[type=checkbox][data-filter]')) return;
      const set = BROWSE_FILTERS[t.dataset.filter];
      if (!set) return;
      if (t.checked) set.add(t.value); else set.delete(t.value);
      renderBrowse();
    });

    $('#sort-select').addEventListener('change', e => {
      BROWSE_SORT = e.target.value;
      renderBrowse();
    });

    $('#filter-reset').addEventListener('click', () => {
      BROWSE_FILTERS = { search: '', topics: new Set(), diffs: new Set(), cats: new Set() };
      $('#search-input').value = '';
      $$('.sidebar input[type=checkbox][data-filter]').forEach(c => { c.checked = false; });
      renderBrowse();
    });

    renderBrowse();
  }

  function countBy(arr, key) {
    return arr.reduce((acc, x) => {
      const k = x[key]; if (!k) return acc;
      acc[k] = (acc[k] || 0) + 1;
      return acc;
    }, {});
  }

  function makeFilterCheckbox(filter, value, label, count) {
    // The data-filter attribute is read as a key into BROWSE_FILTERS — map short
    // names to the actual set names.
    const filterKey = { topic: 'topics', diff: 'diffs', cat: 'cats' }[filter];
    const lbl = document.createElement('label');
    lbl.innerHTML =
      '<input type="checkbox" data-filter="' + filterKey + '" value="' + escapeHtml(value) + '" />' +
      '<span>' + escapeHtml(label) + '</span>' +
      '<span class="count">' + count + '</span>';
    return lbl;
  }

  function filterEntries(opts) {
    const f = opts || BROWSE_FILTERS;
    return ENTRIES.filter(e => {
      if (f.topics && f.topics.size && !f.topics.has(e.topic)) return false;
      if (f.diffs  && f.diffs.size  && !f.diffs.has(e.difficulty)) return false;
      if (f.cats   && f.cats.size   && !f.cats.has(e.category)) return false;
      if (f.search) {
        const hay = (e.question + ' ' + e.short + ' ' + (e.tags || []).join(' ')).toLowerCase();
        if (!hay.includes(f.search)) return false;
      }
      return true;
    });
  }

  function renderBrowse() {
    let items = filterEntries();
    if (BROWSE_SORT === 'topic') {
      items.sort((a, b) =>
        a.topic_num - b.topic_num ||
        diffOrder(a.difficulty) - diffOrder(b.difficulty) ||
        a.id.localeCompare(b.id));
    } else if (BROWSE_SORT === 'difficulty') {
      items.sort((a, b) => diffOrder(a.difficulty) - diffOrder(b.difficulty));
    } else {
      shuffleInPlace(items);
    }

    const list = $('#question-list');
    list.innerHTML = '';
    $('#browse-heading').textContent =
      items.length + ' question' + (items.length === 1 ? '' : 's') +
      (items.length < ENTRIES.length ? ' (filtered)' : '');

    items.forEach((e, idx) => {
      const li = document.createElement('li');
      li.dataset.id = e.id;
      li.innerHTML =
        '<div class="q-row">' +
          '<span class="q-num">' + String(idx + 1).padStart(3, '0') + '</span>' +
          '<span class="q-text">' + escapeHtml(e.question) + '</span>' +
          '<span class="q-meta">' +
            '<span class="pill">' + escapeHtml(topicShort(e.topic)) + '</span>' +
            '<span class="' + diffPillClass(e.difficulty) + '">' + escapeHtml(e.difficulty || '') + '</span>' +
          '</span>' +
        '</div>';
      const row = li.querySelector('.q-row');
      row.addEventListener('click', () => toggleBrowseBody(li, e));
      list.appendChild(li);
    });
  }

  function toggleBrowseBody(li, e) {
    const existing = li.querySelector('.q-body');
    if (existing) { existing.remove(); return; }
    const body = document.createElement('div');
    body.className = 'q-body';
    body.innerHTML = renderEntryHtml(e);
    li.appendChild(body);
  }

  function renderEntryHtml(e) {
    const tags = (e.tags && e.tags.length)
      ? '<div class="md" style="margin-bottom:10px"><em>Tags:</em> ' + e.tags.map(t => '<code>' + escapeHtml(t) + '</code>').join(' ') + '</div>'
      : '';
    const tailLabel = e.tail_label || 'References';
    return (
      tags +
      '<div class="section"><h3>Short answer</h3><div class="md">' + md(e.short) + '</div></div>' +
      (e.expansion ? '<div class="section"><h3>Why this is the answer</h3><div class="md">' + md(e.expansion) + '</div></div>' : '') +
      bulletSection('Common follow-ups', e.follow_ups) +
      bulletSection('Common mistakes', e.mistakes) +
      (e.tail ? '<div class="section"><h3>' + escapeHtml(tailLabel) + '</h3><div class="md">' + md(e.tail) + '</div></div>' : '')
    );
  }

  function bulletSection(label, arr) {
    if (!arr || !arr.length) return '';
    return '<div class="section"><h3>' + escapeHtml(label) + '</h3><ul class="md-list">' +
      arr.map(x => '<li>' + md(x).replace(/^<p>|<\/p>$/g, '') + '</li>').join('') +
      '</ul></div>';
  }

  // ───────────────────────────────────────────────────────────
  // Flashcards (Leitner spaced repetition)
  // ───────────────────────────────────────────────────────────

  const BOX_INTERVAL_MS = [
    0,                         // box 0 — never seen
    1 * 24 * 3600 * 1000,      // box 1 — 1 day
    3 * 24 * 3600 * 1000,      // box 2 — 3 days
    7 * 24 * 3600 * 1000,      // box 3 — 7 days
    14 * 24 * 3600 * 1000,     // box 4 — 14 days
    30 * 24 * 3600 * 1000,     // box 5 — 30 days
  ];

  let FLASH_QUEUE = [];
  let FLASH_INDEX = 0;

  function setupFlash() {
    // Topic dropdown
    const tsel = $('#flash-topic');
    TOPICS.forEach(t => {
      const o = document.createElement('option');
      o.value = t.slug; o.textContent = topicShort(t.slug);
      tsel.appendChild(o);
    });

    ['flash-deck', 'flash-topic', 'flash-difficulty'].forEach(id => {
      $('#' + id).addEventListener('change', renderFlash);
    });

    $('#flash-reveal').addEventListener('click', revealFlash);
    $('#flash-skip').addEventListener('click', () => { advanceFlash(); renderFlash(); });
    $$('.rate', $('#flash-card')).forEach(btn => {
      btn.addEventListener('click', () => rateFlash(btn.dataset.rate));
    });
    $('#flash-reset').addEventListener('click', () => {
      if (confirm('Reset all flashcard progress? This clears boxes and due dates.')) {
        State.resetFlash();
        renderFlash();
      }
    });
  }

  function buildFlashQueue() {
    const deck = $('#flash-deck').value;
    const topic = $('#flash-topic').value;
    const diff  = $('#flash-difficulty').value;
    const now = Date.now();

    let candidates = ENTRIES.filter(e => {
      if (topic && e.topic !== topic) return false;
      if (diff  && e.difficulty !== diff) return false;
      return true;
    });

    if (deck === 'new') {
      candidates = candidates.filter(e => !State.data.flash[e.id]);
    } else if (deck === 'weak') {
      candidates.sort((a, b) => {
        const ba = (State.data.flash[a.id] && State.data.flash[a.id].box) || 0;
        const bb = (State.data.flash[b.id] && State.data.flash[b.id].box) || 0;
        return ba - bb;
      });
    } else if (deck === 'due') {
      candidates = candidates.filter(e => {
        const s = State.data.flash[e.id];
        if (!s) return true;             // never seen → due
        return s.due <= now;
      });
      // due first → least-recently-seen first
      candidates.sort((a, b) => {
        const sa = State.data.flash[a.id], sb = State.data.flash[b.id];
        const da = sa ? sa.due : 0, db = sb ? sb.due : 0;
        return da - db;
      });
    }

    if (deck !== 'weak' && deck !== 'due') shuffleInPlace(candidates);
    return candidates;
  }

  function renderFlash() {
    FLASH_QUEUE = buildFlashQueue();
    FLASH_INDEX = 0;
    renderFlashStats();
    showCurrentFlash();
  }

  function renderFlashStats() {
    const flash = State.data.flash;
    const seen = Object.keys(flash).length;
    const now = Date.now();
    const due = ENTRIES.reduce((n, e) => {
      const s = flash[e.id]; return n + (s ? (s.due <= now ? 1 : 0) : 1);
    }, 0);
    const boxes = [0, 0, 0, 0, 0, 0];
    ENTRIES.forEach(e => {
      const s = flash[e.id]; boxes[s ? s.box : 0]++;
    });
    const mastered = boxes[4] + boxes[5];

    $('#flash-stats').innerHTML =
      '<div>Seen: <span class="v">' + seen + ' / ' + ENTRIES.length + '</span></div>' +
      '<div>Due now: <span class="v">' + due + '</span></div>' +
      '<div>Mastered (box 4–5): <span class="v">' + mastered + '</span></div>' +
      '<div>Box distribution: <span class="v">' + boxes.slice(1).join(' · ') + '</span></div>';
  }

  function showCurrentFlash() {
    const card = $('#flash-card');
    const empty = $('#flash-empty');
    if (!FLASH_QUEUE.length) {
      card.classList.add('hidden');
      empty.classList.remove('hidden');
      empty.innerHTML = '<h2>Nothing due.</h2><p>Switch the deck or come back later — your spaced-repetition schedule is empty.</p>';
      return;
    }
    empty.classList.add('hidden');
    card.classList.remove('hidden');

    const e = FLASH_QUEUE[FLASH_INDEX];
    applyMeta({ topic: '#flash-topic-pill', diff: '#flash-diff-pill', cat: '#flash-cat-pill' }, e);
    const s = State.data.flash[e.id];
    const boxLabel = s ? ('box ' + s.box) : 'new';
    $('#flash-box-pill').textContent = boxLabel;
    $('#flash-question').textContent = e.question;
    $('#flash-answer').classList.add('hidden');
    $('#flash-reveal').classList.remove('hidden');
  }

  function revealFlash() {
    const e = FLASH_QUEUE[FLASH_INDEX];
    if (!e) return;
    $('#flash-short').innerHTML = md(e.short);
    $('#flash-expansion').innerHTML = md(e.expansion);
    $('#flash-followups').innerHTML = (e.follow_ups || []).map(x =>
      '<li>' + md(x).replace(/^<p>|<\/p>$/g, '') + '</li>').join('');
    $('#flash-mistakes').innerHTML = (e.mistakes || []).map(x =>
      '<li>' + md(x).replace(/^<p>|<\/p>$/g, '') + '</li>').join('');
    $('#flash-tail-label').textContent = e.tail_label || 'References';
    $('#flash-tail').innerHTML = md(e.tail);
    $('#flash-answer').classList.remove('hidden');
    $('#flash-reveal').classList.add('hidden');
  }

  function rateFlash(rating) {
    const e = FLASH_QUEUE[FLASH_INDEX];
    if (!e) return;
    const now = Date.now();
    let s = State.data.flash[e.id] || { box: 0, due: 0, last: 0 };
    if (rating === 'again')      s.box = 1;
    else if (rating === 'hard')  s.box = Math.max(1, Math.min(5, s.box || 1));
    else if (rating === 'good')  s.box = Math.min(5, (s.box || 0) + 1);
    else if (rating === 'easy')  s.box = Math.min(5, (s.box || 0) + 2);
    s.last = now;
    s.due = now + BOX_INTERVAL_MS[s.box];
    State.data.flash[e.id] = s;
    State.save();
    advanceFlash();
    renderFlashStats();
    showCurrentFlash();
  }

  function advanceFlash() {
    FLASH_INDEX++;
    if (FLASH_INDEX >= FLASH_QUEUE.length) {
      FLASH_QUEUE = buildFlashQueue();
      FLASH_INDEX = 0;
    }
  }

  // ───────────────────────────────────────────────────────────
  // MCQ Quiz
  // ───────────────────────────────────────────────────────────

  let MCQ_SESSION = null;   // {items: [{id, options, correct_idx}], idx, score: [{id, correct: bool}]}

  function setupMcq() {
    const tsel = $('#mcq-topic');
    TOPICS.forEach(t => {
      const o = document.createElement('option');
      o.value = t.slug; o.textContent = topicShort(t.slug);
      tsel.appendChild(o);
    });

    $('#mcq-start').addEventListener('click', startMcq);
    $('#mcq-next').addEventListener('click', nextMcq);
    $('#mcq-restart').addEventListener('click', () => {
      $('#mcq-result').classList.add('hidden');
      $('#mcq-intro').classList.remove('hidden');
      renderMcqStats();
    });
    $('#mcq-reset').addEventListener('click', () => {
      if (confirm('Reset all MCQ lifetime stats?')) {
        State.resetMcq();
        renderMcqStats();
      }
    });

    renderMcqStats();
  }

  function mcqEligible() {
    // Need at least 3 mistakes to build 3 distractors + short_first non-empty.
    return ENTRIES.filter(e => e.short_first && e.mistakes && e.mistakes.length >= 3);
  }

  function startMcq() {
    const topic = $('#mcq-topic').value;
    const diff  = $('#mcq-difficulty').value;
    const length = parseInt($('#mcq-length').value, 10);

    let pool = mcqEligible();
    if (topic) pool = pool.filter(e => e.topic === topic);
    if (diff)  pool = pool.filter(e => e.difficulty === diff);

    if (!pool.length) {
      alert('No eligible questions match those filters (need ≥3 documented common mistakes).');
      return;
    }

    shuffleInPlace(pool);
    const items = (length > 0 ? pool.slice(0, length) : pool).map(buildMcqItem);

    MCQ_SESSION = { items, idx: 0, results: [] };

    $('#mcq-intro').classList.add('hidden');
    $('#mcq-result').classList.add('hidden');
    $('#mcq-quiz').classList.remove('hidden');
    renderMcqQuestion();
  }

  function buildMcqItem(e) {
    // Correct answer = short_first (first sentence). Distractors = 3 random mistakes from this entry.
    const mistakes = shuffleInPlace(e.mistakes.slice()).slice(0, 3);
    const options = mistakes.map(text => ({ text, correct: false }));
    options.push({ text: e.short_first, correct: true });
    shuffleInPlace(options);
    return {
      id: e.id,
      entry: e,
      question: e.question,
      options,
      correct_idx: options.findIndex(o => o.correct),
    };
  }

  function renderMcqQuestion() {
    const s = MCQ_SESSION;
    if (!s) return;
    const item = s.items[s.idx];
    $('#mcq-progress-text').textContent = 'Question ' + (s.idx + 1) + ' / ' + s.items.length;
    const correct_so_far = s.results.filter(r => r.correct).length;
    $('#mcq-progress-score').textContent = s.idx
      ? 'Score so far: ' + correct_so_far + ' / ' + s.idx
      : '';
    applyMeta({ topic: '#mcq-topic-pill', diff: '#mcq-diff-pill' }, item.entry);
    $('#mcq-question').textContent = item.question;
    const ol = $('#mcq-options'); ol.innerHTML = '';
    item.options.forEach((opt, i) => {
      const li = document.createElement('li');
      li.dataset.idx = i;
      li.innerHTML = '<span class="opt-letter">' + 'ABCD'[i] + '</span><span>' + escapeHtml(opt.text) + '</span>';
      li.addEventListener('click', () => answerMcq(i, li));
      ol.appendChild(li);
    });
    $('#mcq-feedback').classList.add('hidden');
  }

  function answerMcq(picked, li) {
    const s = MCQ_SESSION;
    const item = s.items[s.idx];
    if (li.parentElement.querySelector('.locked')) return;  // already answered
    const correct = picked === item.correct_idx;

    // Lock all options + highlight
    $$('#mcq-options li').forEach((el, i) => {
      el.classList.add('locked');
      if (i === item.correct_idx) el.classList.add('correct');
      else if (i === picked && !correct) el.classList.add('wrong');
      else el.classList.add('faded');
    });

    // Persist lifetime stats
    const m = State.data.mcq[item.id] || { right: 0, wrong: 0 };
    if (correct) m.right++; else m.wrong++;
    State.data.mcq[item.id] = m;
    State.save();

    s.results.push({ id: item.id, correct, picked, entry: item.entry, options: item.options, correct_idx: item.correct_idx });

    // Show explanation: full short answer + first expansion bullet if any
    $('#mcq-explanation').innerHTML =
      '<p><strong>' + (correct ? 'Correct.' : 'Not quite.') + '</strong></p>' +
      '<div class="section"><h3>Short answer</h3><div class="md">' + md(item.entry.short) + '</div></div>' +
      (item.entry.expansion ? '<div class="section"><h3>Why this is the answer</h3><div class="md">' + md(item.entry.expansion) + '</div></div>' : '');

    $('#mcq-feedback').classList.remove('hidden');
    renderMcqStats();
  }

  function nextMcq() {
    const s = MCQ_SESSION; if (!s) return;
    s.idx++;
    if (s.idx >= s.items.length) {
      finishMcq();
    } else {
      renderMcqQuestion();
    }
  }

  function finishMcq() {
    const s = MCQ_SESSION; if (!s) return;
    const right = s.results.filter(r => r.correct).length;
    const total = s.results.length;
    const pct = total ? right / total : 0;

    // Group results by topic
    const byTopic = {};
    s.results.forEach(r => {
      const slug = r.entry.topic;
      if (!byTopic[slug]) byTopic[slug] = { right: 0, total: 0 };
      byTopic[slug].total++;
      if (r.correct) byTopic[slug].right++;
    });
    const topicLines = Object.keys(byTopic).sort().map(slug =>
      '<div>' + escapeHtml(topicShort(slug)) + ': <span class="v">' + byTopic[slug].right + ' / ' + byTopic[slug].total + '</span></div>'
    ).join('');

    $('#mcq-result-score').textContent = right + ' / ' + total + '  (' + fmtPercent(pct) + ')';
    $('#mcq-result-detail').innerHTML = topicLines || '<div>No questions answered.</div>';
    $('#mcq-result-review').innerHTML = s.results.map(r =>
      '<li class="' + (r.correct ? 'ok' : 'bad') + '">' +
        '<div class="rq">' + escapeHtml(r.entry.question) + '</div>' +
        '<div class="rmeta">' +
          (r.correct ? 'Correct.' : 'You picked: <em>' + escapeHtml(r.options[r.picked].text) + '</em>') +
          ' · Correct answer: <em>' + escapeHtml(r.options[r.correct_idx].text) + '</em>' +
          ' · ' + escapeHtml(topicShort(r.entry.topic)) +
        '</div>' +
      '</li>'
    ).join('');

    $('#mcq-quiz').classList.add('hidden');
    $('#mcq-result').classList.remove('hidden');
  }

  function renderMcqStats() {
    const ids = Object.keys(State.data.mcq);
    let right = 0, wrong = 0;
    ids.forEach(id => { right += State.data.mcq[id].right; wrong += State.data.mcq[id].wrong; });
    const total = right + wrong;
    const pct = total ? right / total : 0;
    $('#mcq-stats').innerHTML =
      '<div>Questions touched: <span class="v">' + ids.length + ' / ' + ENTRIES.length + '</span></div>' +
      '<div>Answers given: <span class="v">' + total + '</span></div>' +
      '<div>Accuracy: <span class="v">' + (total ? fmtPercent(pct) : '—') + '</span></div>';
  }

  // ───────────────────────────────────────────────────────────
  // Mock Interview
  // ───────────────────────────────────────────────────────────

  let MOCK_SESSION = null; // {queue, idx, timer_id, time_left, time_per, results}

  function setupMock() {
    const tsel = $('#mock-topic');
    TOPICS.forEach(t => {
      const o = document.createElement('option');
      o.value = t.slug; o.textContent = topicShort(t.slug);
      tsel.appendChild(o);
    });

    $('#mock-start').addEventListener('click', startMock);
    $('#mock-quit').addEventListener('click', () => {
      if (confirm('End this round now and see the report so far?')) finishMock();
    });
    $('#mock-reveal').addEventListener('click', revealMock);
    $$('.rate', $('#mock-stage')).forEach(btn => {
      btn.addEventListener('click', () => rateMock(parseInt(btn.dataset.rate, 10)));
    });
    $('#mock-restart').addEventListener('click', () => {
      $('#mock-report').classList.add('hidden');
      $('#mock-intro').classList.remove('hidden');
      renderMockHistory();
    });

    renderMockHistory();
  }

  function startMock() {
    const topic = $('#mock-topic').value;
    const diff  = $('#mock-difficulty').value;
    const count = parseInt($('#mock-count').value, 10);
    const timePer = parseInt($('#mock-time').value, 10);

    let pool = ENTRIES.slice();
    if (topic) pool = pool.filter(e => e.topic === topic);
    if (diff)  pool = pool.filter(e => e.difficulty === diff);
    if (!pool.length) { alert('No questions match those filters.'); return; }

    shuffleInPlace(pool);
    const queue = pool.slice(0, Math.min(count, pool.length));

    MOCK_SESSION = {
      queue, idx: 0, time_per: timePer, time_left: timePer,
      timer_id: null, started_at: Date.now(), results: [],
      topic, difficulty: diff, count: queue.length,
    };

    $('#mock-intro').classList.add('hidden');
    $('#mock-report').classList.add('hidden');
    $('#mock-stage').classList.remove('hidden');
    renderMockQuestion();
  }

  function renderMockQuestion() {
    const s = MOCK_SESSION; if (!s) return;
    const e = s.queue[s.idx];
    $('#mock-progress-text').textContent = (s.idx + 1) + ' / ' + s.queue.length;
    applyMeta({ topic: '#mock-topic-pill', diff: '#mock-diff-pill', cat: '#mock-cat-pill' }, e);
    $('#mock-question').textContent = e.question;
    $('#mock-notes').value = '';
    $('#mock-reveal-block').classList.add('hidden');
    $('#mock-reveal').classList.remove('hidden');
    startMockTimer();
  }

  function startMockTimer() {
    const s = MOCK_SESSION; if (!s) return;
    stopMockTimer();
    if (!s.time_per) {
      $('#mock-timer').textContent = '∞';
      $('#mock-timer').className = 'mock-timer';
      return;
    }
    s.time_left = s.time_per;
    updateMockTimer();
    s.timer_id = setInterval(() => {
      s.time_left--;
      updateMockTimer();
    }, 1000);
  }

  function stopMockTimer() {
    if (MOCK_SESSION && MOCK_SESSION.timer_id) {
      clearInterval(MOCK_SESSION.timer_id);
      MOCK_SESSION.timer_id = null;
    }
  }

  function updateMockTimer() {
    const s = MOCK_SESSION; if (!s) return;
    const el = $('#mock-timer');
    el.textContent = fmtTime(s.time_left);
    el.className = 'mock-timer';
    if (s.time_left <= 0) el.classList.add('over');
    else if (s.time_left <= 15) el.classList.add('warn');
  }

  function revealMock() {
    const s = MOCK_SESSION; if (!s) return;
    const e = s.queue[s.idx];
    stopMockTimer();
    $('#mock-short').innerHTML = md(e.short);
    $('#mock-expansion').innerHTML = md(e.expansion);
    $('#mock-followups').innerHTML = (e.follow_ups || []).map(x =>
      '<li>' + md(x).replace(/^<p>|<\/p>$/g, '') + '</li>').join('');
    $('#mock-mistakes').innerHTML = (e.mistakes || []).map(x =>
      '<li>' + md(x).replace(/^<p>|<\/p>$/g, '') + '</li>').join('');
    $('#mock-tail-label').textContent = e.tail_label || 'References';
    $('#mock-tail').innerHTML = md(e.tail);
    $('#mock-reveal-block').classList.remove('hidden');
    $('#mock-reveal').classList.add('hidden');
  }

  function rateMock(rating) {
    const s = MOCK_SESSION; if (!s) return;
    const e = s.queue[s.idx];
    s.results.push({ id: e.id, entry: e, rate: rating });
    s.idx++;
    if (s.idx >= s.queue.length) finishMock();
    else renderMockQuestion();
  }

  function finishMock() {
    stopMockTimer();
    const s = MOCK_SESSION; if (!s) return;
    const ratings = s.results.map(r => r.rate);
    const total = ratings.length;
    const sum = ratings.reduce((a, b) => a + b, 0);
    const mean = total ? sum / total : 0;

    // Per-topic mean
    const byTopic = {};
    s.results.forEach(r => {
      const slug = r.entry.topic;
      if (!byTopic[slug]) byTopic[slug] = { sum: 0, n: 0 };
      byTopic[slug].sum += r.rate; byTopic[slug].n++;
    });
    const topicLines = Object.keys(byTopic).sort().map(slug => {
      const x = byTopic[slug];
      return '<div>' + escapeHtml(topicShort(slug)) + ': <span class="v">' + (x.sum / x.n).toFixed(2) + ' avg (' + x.n + ')</span></div>';
    }).join('');

    $('#mock-report-score').textContent = mean.toFixed(2) + ' / 5  · ' + total + ' question' + (total === 1 ? '' : 's');
    $('#mock-report-detail').innerHTML = topicLines || '<div>No questions answered.</div>';
    $('#mock-report-list').innerHTML = s.results.map(r => {
      const cls = r.rate >= 4 ? 'ok' : (r.rate <= 2 ? 'bad' : '');
      return '<li class="' + cls + '">' +
        '<div class="rq">' + escapeHtml(r.entry.question) + '</div>' +
        '<div class="rmeta">Rated <strong>' + r.rate + '/5</strong> · ' + escapeHtml(topicShort(r.entry.topic)) +
          ' · ' + escapeHtml(r.entry.difficulty || '') +
        '</div>' +
      '</li>';
    }).join('');

    // Persist history
    State.data.mock_history.unshift({
      date: new Date().toISOString(),
      topic: s.topic || 'all',
      difficulty: s.difficulty || 'all',
      count: total,
      mean: Number(mean.toFixed(2)),
    });
    State.data.mock_history = State.data.mock_history.slice(0, 20); // cap
    State.save();

    $('#mock-stage').classList.add('hidden');
    $('#mock-report').classList.remove('hidden');
  }

  function renderMockHistory() {
    const hist = State.data.mock_history || [];
    const box = $('#mock-history');
    if (!hist.length) { box.innerHTML = ''; return; }
    const rows = hist.slice(0, 10).map(h => {
      const dt = new Date(h.date);
      const when = dt.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
                   ' ' + dt.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
      return '<tr><td>' + escapeHtml(when) + '</td>' +
             '<td>' + escapeHtml(h.topic === 'all' ? 'all' : topicShort(h.topic)) + '</td>' +
             '<td>' + escapeHtml(h.difficulty) + '</td>' +
             '<td>' + h.count + ' Q</td>' +
             '<td>' + h.mean.toFixed(2) + ' / 5</td></tr>';
    }).join('');
    box.innerHTML = '<h3>Recent rounds</h3><table><tbody>' + rows + '</tbody></table>';
  }

  // ───────────────────────────────────────────────────────────
  // Cloze
  // ───────────────────────────────────────────────────────────

  let CLOZE_CURRENT = null;

  function setupCloze() {
    const tsel = $('#cloze-topic');
    TOPICS.forEach(t => {
      const o = document.createElement('option');
      o.value = t.slug; o.textContent = topicShort(t.slug);
      tsel.appendChild(o);
    });
    ['cloze-topic', 'cloze-difficulty'].forEach(id => {
      $('#' + id).addEventListener('change', renderCloze);
    });
    $('#cloze-next').addEventListener('click', renderCloze);
    $('#cloze-next-2').addEventListener('click', renderCloze);
    $('#cloze-reveal-all').addEventListener('click', () => {
      $$('.cloze-blank').forEach(el => el.classList.add('revealed'));
    });
  }

  function clozePool() {
    const topic = $('#cloze-topic').value;
    const diff = $('#cloze-difficulty').value;
    return ENTRIES.filter(e => {
      if (topic && e.topic !== topic) return false;
      if (diff && e.difficulty !== diff) return false;
      return e.short && e.short.length > 30;
    });
  }

  function renderCloze() {
    const pool = clozePool();
    if (!pool.length) {
      $('#cloze-card').innerHTML = '<p class="empty">No matching entries.</p>';
      return;
    }
    let e = pick(pool);
    // Avoid showing the same one twice in a row
    if (CLOZE_CURRENT && CLOZE_CURRENT.id === e.id && pool.length > 1) {
      while (e.id === CLOZE_CURRENT.id) e = pick(pool);
    }
    CLOZE_CURRENT = e;

    // Rebuild card markup in case prior render replaced it
    $('#cloze-card').innerHTML =
      '<div class="flash-meta">' +
        '<span class="pill" id="cloze-topic-pill"></span>' +
        '<span class="pill" id="cloze-diff-pill"></span>' +
      '</div>' +
      '<h2 class="flash-question" id="cloze-question"></h2>' +
      '<div class="section"><h3>Fill in the blanks (click a blank to reveal)</h3>' +
        '<div class="md" id="cloze-body"></div></div>' +
      '<div class="cloze-actions">' +
        '<button id="cloze-reveal-all" class="ghost">Reveal all</button>' +
        '<button id="cloze-next-2" class="primary">Next cloze</button>' +
      '</div>';

    applyMeta({ topic: '#cloze-topic-pill', diff: '#cloze-diff-pill' }, e);
    $('#cloze-question').textContent = e.question;
    $('#cloze-body').innerHTML = blankifyShort(e.short);

    // re-bind buttons (newly rendered)
    $('#cloze-next-2').addEventListener('click', renderCloze);
    $('#cloze-reveal-all').addEventListener('click', () => {
      $$('.cloze-blank').forEach(el => el.classList.add('revealed'));
    });
    $$('.cloze-blank', $('#cloze-body')).forEach(el => {
      el.addEventListener('click', () => el.classList.add('revealed'));
    });
  }

  function blankifyShort(text) {
    // Strategy: render markdown first, then in the resulting HTML, replace
    // a sampled subset of <code>…</code> spans and <strong>…</strong> spans
    // with cloze blanks. This is a pragmatic heuristic; not perfect.
    let html = md(text);
    const tmp = document.createElement('div');
    tmp.innerHTML = html;
    const cands = Array.from(tmp.querySelectorAll('code, strong, em'));
    // Sample up to 5 candidates, weighted toward earlier ones.
    if (!cands.length) {
      // Fall back: blank the longest plain-text word in the first paragraph.
      const p = tmp.querySelector('p');
      if (p) {
        const words = p.textContent.split(/\s+/).filter(w => /^[A-Za-z][A-Za-z\-]{4,}$/.test(w));
        if (words.length) {
          const pick3 = shuffleInPlace(Array.from(new Set(words))).slice(0, 4);
          let txt = p.innerHTML;
          pick3.forEach(w => {
            const re = new RegExp('\\b' + w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\b');
            txt = txt.replace(re, '<span class="cloze-blank" data-answer="' + escapeHtml(w) + '">' + escapeHtml(w) + '</span>');
          });
          p.innerHTML = txt;
        }
      }
      return tmp.innerHTML;
    }
    const sampled = shuffleInPlace(cands.slice()).slice(0, Math.min(5, Math.max(2, Math.ceil(cands.length / 2))));
    sampled.forEach(el => {
      const span = document.createElement('span');
      span.className = 'cloze-blank';
      span.dataset.answer = el.textContent;
      span.textContent = el.textContent;
      el.replaceWith(span);
    });
    return tmp.innerHTML;
  }

  // ───────────────────────────────────────────────────────────
  // Global keyboard shortcuts
  // ───────────────────────────────────────────────────────────

  function setupGlobalKeys() {
    document.addEventListener('keydown', (e) => {
      // Ignore when typing in inputs/textareas
      const t = e.target;
      const editable = t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable);
      if (editable && e.key !== 'Escape') return;

      // '/' focuses the search box (only when in browse mode)
      if (e.key === '/' && !$('#view-browse').classList.contains('hidden')) {
        e.preventDefault(); $('#search-input').focus(); return;
      }

      // Flashcard shortcuts
      if (!$('#view-flash').classList.contains('hidden')) {
        if (e.key === ' ' || e.key === 'Enter') {
          e.preventDefault();
          if ($('#flash-answer').classList.contains('hidden')) revealFlash();
          return;
        }
        if (e.key === 'k' || e.key === 'j') { advanceFlash(); showCurrentFlash(); return; }
        if (!$('#flash-answer').classList.contains('hidden')) {
          if (e.key === '1') return rateFlash('again');
          if (e.key === '2') return rateFlash('hard');
          if (e.key === '3') return rateFlash('good');
          if (e.key === '4') return rateFlash('easy');
        }
      }

      // MCQ shortcuts
      if (!$('#view-mcq').classList.contains('hidden')) {
        if (!$('#mcq-quiz').classList.contains('hidden')) {
          if (['1','2','3','4','a','b','c','d','A','B','C','D'].includes(e.key)) {
            const map = { '1':0,'2':1,'3':2,'4':3,'a':0,'b':1,'c':2,'d':3,'A':0,'B':1,'C':2,'D':3 };
            const idx = map[e.key];
            const li = $$('#mcq-options li')[idx];
            if (li && !li.classList.contains('locked')) answerMcq(idx, li);
            return;
          }
          if ((e.key === 'Enter' || e.key === ' ') && !$('#mcq-feedback').classList.contains('hidden')) {
            e.preventDefault(); nextMcq(); return;
          }
        }
      }
    });
  }

})();
