/* sunny-cx 白话问诊 —— 前端逻辑（零依赖）
 *
 * 只做三件事：把主诉发给后端、把结构化结果排成版面、把「拒绝回答」如实显示。
 * 前端不做任何医学判断，也不缓存结论。
 */
(function () {
  "use strict";

  // 直接用 file:// 打开页面时，后端默认在本机 8000
  var API = location.protocol === "file:" ? "http://127.0.0.1:8000" : "";

  var EXAMPLES = [
    "我最近老觉得累，一动就出汗，舌头发紫，身上老是一个地方刺痛",
    "晚上睡不着，手心脚心发热，夜里出汗，嘴干",
    "小便发黄，阴囊潮湿，两条腿发沉",
    "胃里老烧得慌，反酸，嘴里苦"
  ];

  var VERDICT = {
    ok:           { text: "已辨证",  kind: "pass" },
    differential: { text: "需鉴别",  kind: "hold" },
    weak:         { text: "仅方向",  kind: "hold" },
    refuse:       { text: "已拒答",  kind: "hold" }
  };

  var $ = function (sel, root) { return (root || document).querySelector(sel); };

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function pct(x) { return Math.round((x || 0) * 100) + "%"; }

  /* ---------------------------------------------------------------- 状态 */
  var el = {
    input:    $("#input"),
    counter:  $("#counter"),
    submit:   $("#submit"),
    result:   $("#result"),
    verdict:  $("#verdict"),
    dot:      $("#dot"),
    statusTx: $("#status-text"),
    statLine: $("#stat-line"),
    disclaim: $("#disclaimer"),
    examples: $("#examples")
  };

  var busy = false;

  /* ---------------------------------------------------------------- 启动 */
  function boot() {
    renderExamples();
    bind();

    // ?q=xxx 可直接带主诉进来（便于分享与截图复核）
    var q = new URLSearchParams(location.search).get("q");
    if (q) {
      el.input.value = q.slice(0, 1000);
      onInput();
      go();
    }

    fetchJSON("/api/meta")
      .then(function (m) {
        setStatus("ok", "引擎在线");
        var s = m.stats;
        el.statLine.textContent = s.syndromes + " 证型 · " + s.symptoms + " 症状 · " + s.aliases + " 别名";
        if (m.disclaimer) el.disclaim.textContent = m.disclaimer;
      })
      .catch(function () {
        setStatus("error", "后端未连接");
        el.statLine.textContent = "未取到引擎统计";
      });
  }

  function renderExamples() {
    el.examples.innerHTML = EXAMPLES.map(function (t, i) {
      return '<button class="chip" type="button" data-i="' + i + '">' + esc(t) + "</button>";
    }).join("");
  }

  function bind() {
    el.examples.addEventListener("click", function (e) {
      var b = e.target.closest(".chip");
      if (!b) return;
      el.input.value = EXAMPLES[Number(b.dataset.i)] || "";
      onInput();
      el.input.focus();
    });

    el.input.addEventListener("input", onInput);

    el.input.addEventListener("keydown", function (e) {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); go(); }
    });

    el.submit.addEventListener("click", go);
    onInput();
  }

  function onInput() {
    var n = el.input.value.length;
    el.counter.textContent = n + " / 1000";
    el.counter.style.color = n > 900 ? "var(--cinnabar)" : "";
    el.submit.disabled = busy || n === 0;
  }

  function setStatus(state, text) {
    el.dot.dataset.state = state;
    el.statusTx.textContent = text;
  }

  /* ---------------------------------------------------------------- 请求 */
  function fetchJSON(path, opts) {
    var ctl = new AbortController();
    var timer = setTimeout(function () { ctl.abort(); }, 15000);
    var o = Object.assign({ signal: ctl.signal, headers: {} }, opts || {});
    if (o.body) o.headers["Content-Type"] = "application/json";
    return fetch(API + path, o).then(function (r) {
      clearTimeout(timer);
      if (!r.ok) {
        return r.text().then(function (t) {
          throw new Error("HTTP " + r.status + (t ? " · " + t.slice(0, 120) : ""));
        });
      }
      return r.json();
    }).catch(function (e) {
      clearTimeout(timer);
      throw e;
    });
  }

  function go() {
    var text = el.input.value.trim();
    if (!text || busy) return;

    busy = true;
    el.submit.disabled = true;
    el.submit.textContent = "辨证中";
    el.verdict.hidden = true;
    el.result.innerHTML = '<div class="loading"><i></i><i></i><i></i></div>';

    fetchJSON("/api/consult", {
      method: "POST",
      body: JSON.stringify({ text: text, top_k: 3 })
    }).then(function (d) {
      renderResult(d);
      setStatus("ok", "引擎在线");
    }).catch(function (err) {
      renderError(err);
      setStatus("error", "请求失败");
    }).then(function () {
      busy = false;
      el.submit.textContent = "开始辨证";
      onInput();
    });
  }

  /* ---------------------------------------------------------------- 渲染 */
  function renderError(err) {
    var msg = String(err && err.message || err);
    var hint = /abort/i.test(msg)
      ? "请求超时。引擎首次加载词表较慢，或后端未启动。"
      : (/Failed to fetch|NetworkError/i.test(msg)
        ? "连不上后端。请确认已运行 <code>python -m server.app</code>（默认 127.0.0.1:8000）。"
        : esc(msg));
    el.result.innerHTML =
      '<div class="alert">' +
        '<p class="alert-lead">没有拿到结果</p>' +
        '<p class="alert-sub">' + hint + "</p>" +
      "</div>";
    el.verdict.hidden = true;
  }

  function renderResult(d) {
    var v = VERDICT[d.decision] || { text: d.decision, kind: "hold" };
    el.verdict.hidden = false;
    el.verdict.textContent = v.text;
    el.verdict.dataset.kind = v.kind;

    var html = "";

    /* 识别到的症状 */
    if (d.symptoms && d.symptoms.length) {
      html += '<div class="block">' +
        '<span class="block-label">识别到的症状 ' + d.symptoms.length + " 项</span>" +
        '<div class="sym-list">' +
        d.symptoms.map(function (s) {
          var surface = s.surface && s.surface !== s.term ? "<em>「" + esc(s.surface) + "」</em>" : "";
          return '<span class="sym">' + esc(s.term) + surface + "</span>";
        }).join("") +
        "</div></div>";
    }

    /* 主结论 */
    html += renderMain(d);

    /* 治法 / 方剂 / 药味 */
    if (d.formula) html += renderFormula(d.formula, d.decision);

    /* 候选打分（拒答时不给任何候选：0 分候选等于凭空造结论） */
    if (d.decision !== "refuse" && d.candidates && d.candidates.length) html += renderCandidates(d.candidates);

    /* 内部校验告警 */
    var w = (d.selfcheck && d.selfcheck.warnings) || [];
    if (w.length) {
      html += '<p class="warn">闸门提示：<b>' + w.map(esc).join("；") + "</b></p>";
    }

    /* 原始文本：便于核对模板与槽位 */
    if (d.answer) {
      html += "<details class=\"block\" style=\"border-top:1px solid var(--rule-soft);padding-top:18px\">" +
        '<summary class="block-label" style="cursor:pointer">引擎原始输出</summary>' +
        '<pre style="margin:10px 0 0;white-space:pre-wrap;font-family:var(--serif);font-size:13px;line-height:1.95;color:var(--ink-2)">' +
        esc(d.answer) + "</pre></details>";
    }

    el.result.innerHTML = html;
  }

  function renderMain(d) {
    var top = (d.candidates && d.candidates[0]) || null;
    var ratio = top ? top.ratio : 0;

    if (d.decision === "refuse") {
      return '<div class="block"><div class="alert">' +
        '<p class="alert-lead">这次不判</p>' +
        '<p class="alert-sub">' + esc(d.reason || "信息不足") + "</p>" +
        "</div>" +
        '<p class="ask">可以补这些：哪里不舒服、持续多久、什么情况下加重或缓解、' +
        "饮食睡眠二便、舌苔颜色厚薄、怕冷还是怕热。</p></div>";
    }

    var label = d.decision === "weak" ? "目前只能给方向" : "辨证方向";
    var name = d.syndrome || d.syndrome_hint || "";
    var note = "";

    if (d.decision === "differential" && d.differential_with) {
      note = "与「" + esc(d.differential_with) + "」证据接近，两证治法不同，需医师面诊鉴别后定夺。";
    } else if (d.decision === "weak") {
      var miss = (d.missing || []).join("、");
      note = "还缺 " + (miss ? esc(miss) : "关键信息") + "，补上才可能给出结论。";
    } else {
      note = "";   // 覆盖率与结论理由由下方标尺表达，避免重复
    }

    var h = '<div class="block">' +
      '<span class="block-label">' + label + "</span>" +
      '<p class="diagnosis">' + esc(name) +
      (top && top.base ? ' <span class="bracket">〔' + esc(top.base) + "〕</span>" : "") +
      "</p>" +
      (note ? '<p class="diagnosis-note">' + note + "</p>" : "") +
      '<div class="ruler">' +
        '<div class="ruler-track"><div class="ruler-fill" style="width:' + pct(ratio) + '"></div>' +
        '<div class="ruler-ticks"></div></div>' +
        '<div class="ruler-foot"><span>证据覆盖 ' + pct(ratio) + "</span>" +
        "<span>命中 " + (top ? top.matched.length : 0) + " · 缺 " + (top ? top.missing.length : 0) + "</span></div>" +
      "</div></div>";

    if (d.decision === "weak") {
      h += '<p class="ask">建议补充：起病多久、什么情况下加重、饮食睡眠二便、舌苔颜色厚薄、是否怕冷怕热。</p>';
    }
    return h;
  }

  function renderFormula(f, decision) {
    var rows = "";
    rows += '<div><dt>治法</dt><dd>' + esc(f.zhi_fa) + "</dd></div>";
    rows += '<div><dt>参考方剂</dt><dd>' + esc(f.formula) +
      ' <span class="src">（' + esc(f.source) + "）</span></dd></div>";
    rows += '<div><dt>参考药味</dt><dd class="herbs">' + esc((f.composition || []).join("、")) + "</dd></div>";
    if (f.alt && f.alt.length) {
      rows += '<div><dt>备选方</dt><dd class="herbs">' + esc(f.alt.join("、")) + "</dd></div>";
    }
    if (f.note) {
      rows += '<div><dt>编者注</dt><dd class="src">' + esc(f.note) + "</dd></div>";
    }
    rows += '<div><dt>剂量</dt><dd class="src">不给克数。中药剂量须依体质、年龄、兼证加减，照搬网络剂量有风险。</dd></div>';

    var head = decision === "differential"
      ? '<p class="ask" style="margin:0 0 14px">两证治法不同，请勿自行选方。</p>'
      : "";

    return '<div class="block"><span class="block-label">方证对照</span>' + head +
      '<dl class="kv">' + rows + "</dl></div>";
  }

  function renderCandidates(cands) {
    var max = cands.reduce(function (a, c) { return Math.max(a, c.score || 0); }, 0) || 1;
    return '<div class="block"><span class="block-label">候选证型 top ' + cands.length + "</span>" +
      '<ul class="cands">' +
      cands.map(function (c, i) {
        var rel = Math.max(0.04, (c.score || 0) / max);   // 相对证据强度
        return '<li class="cand' + (i === 0 ? " is-top" : "") + '">' +
          '<span class="cand-name">' + esc(c.syndrome) +
            '<span class="cand-base">' + esc(c.base) + " · 命中 " + c.matched.length +
            " · 覆盖 " + pct(c.ratio) + "</span></span>" +
          '<span class="cand-bar" title="相对证据强度 ' + Math.round(rel * 100) + '%">' +
            '<i style="width:' + Math.round(rel * 100) + '%"></i></span>' +
          '<span class="cand-ratio">' + Math.round(rel * 100) + "%</span>" +
          "</li>";
      }).join("") +
      "</ul>" +
      '<p class="warn" style="margin-top:10px">条形与右侧数字＝相对证据强度（以首位为 100%，含命中项数加权），' +
      "副行的「覆盖」是该证型自身要素的命中比例。单要素证型命中一词即满覆盖，故排序看前者。</p></div>";
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
