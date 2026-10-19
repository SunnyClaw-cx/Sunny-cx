/* ==========================================================================
   sunny-cx 训练看板 —— 轮询本地 API，渲染进度 / 曲线 / 验证点
   无依赖。数据全部来自 /api/train/status（读本机 logs 与 checkpoints）。
   ========================================================================== */
(function () {
  "use strict";

  var POLL_MS = 3000;
  var currentRun = null;
  var lastPoll = 0;
  var lastState = "unknown";

  var $ = function (id) { return document.getElementById(id); };

  function nf(n) {
    return (n === null || n === undefined) ? "—" : Number(n).toLocaleString("en-US");
  }
  function fx(n, d) {
    return (n === null || n === undefined) ? "—" : Number(n).toFixed(d === undefined ? 4 : d);
  }
  function exp(n) {
    if (n === null || n === undefined) return "—";
    return Number(n).toExponential(2).replace("e-", "e−");
  }
  function hours(min) {
    if (min === null || min === undefined) return "—";
    return (min / 60).toFixed(1) + " h";
  }
  function ageText(sec) {
    if (sec === null || sec === undefined) return "—";
    if (sec < 60) return Math.round(sec) + " 秒前";
    if (sec < 3600) return Math.round(sec / 60) + " 分钟前";
    return (sec / 3600).toFixed(1) + " 小时前";
  }

  /* 逐步 loss 抖动很大，画一条 EMA 才能看出趋势；原始值仍以细线保留 */
  function ema(series, alpha) {
    var out = [], prev = null;
    for (var i = 0; i < series.length; i++) {
      prev = prev === null ? series[i] : alpha * series[i] + (1 - alpha) * prev;
      out.push(prev);
    }
    return out;
  }

  /* ------------------------------------------------------------ 曲线 */
  function drawCurve(d) {
    var svg = $("curve");
    var W = 720, H = 250, L = 48, R = 10, T = 14, B = 26;
    var pts = (d.curve || []).filter(function (p) { return p.loss != null; });
    var vals = (d.vals || []).filter(function (v) { return v.val != null; });

    if (pts.length < 2) {
      svg.innerHTML =
        '<text class="empty-msg" x="' + (W / 2) + '" y="' + (H / 2) +
        '" text-anchor="middle">日志里还没有足够的 loss 记录</text>';
      return;
    }

    var ys = pts.map(function (p) { return p.loss; })
      .concat(vals.map(function (v) { return v.val; }));
    var lo = Math.min.apply(null, ys);
    var hi = Math.max.apply(null, ys);
    var pad = (hi - lo) * 0.14 || 0.5;
    lo -= pad; hi += pad;

    var x0 = pts[0].step;
    var x1 = pts[pts.length - 1].step;
    if (x1 === x0) x1 = x0 + 1;

    function sx(s) { return L + (s - x0) / (x1 - x0) * (W - L - R); }
    function sy(v) { return T + (hi - v) / (hi - lo) * (H - T - B); }

    var out = [];

    // 水平网格 + y 轴刻度（4 条）
    var ticks = 4;
    for (var i = 0; i < ticks; i++) {
      var v = lo + (hi - lo) * (i / (ticks - 1));
      var y = sy(v);
      out.push('<line class="grid" x1="' + L + '" y1="' + y.toFixed(1) +
        '" x2="' + (W - R) + '" y2="' + y.toFixed(1) + '"/>');
      out.push('<text class="axis" x="' + (L - 8) + '" y="' + (y + 3.5).toFixed(1) +
        '" text-anchor="end">' + v.toFixed(2) + "</text>");
    }

    // best val 参考线（数值放到图例里，避免压在曲线上）
    if (d.best_val != null && d.best_val >= lo && d.best_val <= hi) {
      var by = sy(d.best_val);
      out.push('<line class="best" x1="' + L + '" y1="' + by.toFixed(1) +
        '" x2="' + (W - R) + '" y2="' + by.toFixed(1) + '"/>');
      $("lg-best").hidden = false;
      $("lg-best-text").textContent = "best " + d.best_val.toFixed(4);
    } else {
      $("lg-best").hidden = true;
    }

    // 原始 loss（细线，保留真实抖动）
    var raw = pts.map(function (p, i) {
      return (i ? "L" : "M") + sx(p.step).toFixed(1) + " " + sy(p.loss).toFixed(1);
    }).join(" ");
    out.push('<path class="raw" d="' + raw + '"/>');

    // EMA 趋势线
    var smooth = ema(pts.map(function (p) { return p.loss; }), 0.18);
    var path = pts.map(function (p, i) {
      return (i ? "L" : "M") + sx(p.step).toFixed(1) + " " + sy(smooth[i]).toFixed(1);
    }).join(" ");
    out.push('<path class="train" d="' + path + '"/>');

    // 验证点
    vals.forEach(function (v) {
      if (v.step < x0 || v.step > x1) return;
      out.push('<circle class="val" cx="' + sx(v.step).toFixed(1) +
        '" cy="' + sy(v.val).toFixed(1) + '" r="3"/>');
    });

    // x 轴两端
    out.push('<text class="axis" x="' + L + '" y="' + (H - 8) + '">' + nf(x0) + "</text>");
    out.push('<text class="axis" x="' + (W - R) + '" y="' + (H - 8) +
      '" text-anchor="end">' + nf(x1) + "</text>");

    svg.innerHTML = out.join("");
  }

  /* ------------------------------------------------------------ 渲染 */
  function render(d) {
    lastState = d.state;
    lastPoll = Date.now();

    var stateMap = {
      training: ["训练中", "training"],
      resting: ["休息中" + (d.rest_timer ? " " + d.rest_timer[0] + "/" + d.rest_timer[1] + " min" : ""), "resting"],
      stopped: ["已停止", "stopped"],
      unknown: ["未知", "unknown"]
    };
    var st = stateMap[d.state] || stateMap.unknown;
    $("dot").setAttribute("data-state", st[1]);
    $("state-text").textContent = st[0];
    $("updated").textContent = "日志 " + ageText(d.log_age_sec) + "更新";

    $("step").textContent = nf(d.step);
    $("total").textContent = nf(d.total_steps);
    var pct = d.progress == null ? 0 : d.progress * 100;
    $("ruler-fill").style.width = pct.toFixed(2) + "%";
    $("pct").textContent = d.progress == null ? "—" : pct.toFixed(2) + "%";
    $("epoch").textContent = (d.epoch == null) ? "—"
      : "epoch " + d.epoch + " / " + d.epochs;

    $("m-loss").textContent = fx(d.loss);
    $("m-lr").textContent = exp(d.lr);
    $("m-gnorm").textContent = fx(d.gnorm, 3);
    $("m-vram").textContent = (d.peak_gb == null) ? "—"
      : fx(d.peak_gb, 2) + " / " + fx(d.peak_total_gb, 2) + " GB";
    $("m-best").textContent = (d.best_val == null) ? "—"
      : fx(d.best_val) + (d.best_step == null ? "" : " @ " + nf(d.best_step));
    $("m-elapsed").textContent = hours(d.elapsed_min);
    $("m-sps").textContent = (d.sec_per_step == null) ? "—" : fx(d.sec_per_step, 2) + " s";
    $("m-eta").textContent = (d.eta_hours == null) ? "—" : fx(d.eta_hours, 1) + " h";
    $("m-rest").textContent = (d.rest_count == null) ? "—"
      : d.rest_count + " 次 / " + fx(d.rest_total_hours, 2) + " h";

    $("m-best").className = d.best_val == null ? "" : "is-accent";

    var hint = "";
    if (d.state === "training") hint = "每 3 秒刷新一次，数据来自本机 logs/ 与 checkpoints/。";
    else if (d.state === "resting") hint = "节律休息中，GPU 空闲属正常，到点自动恢复。";
    else if (d.state === "stopped") hint = "进程未在运行。双击 scripts\\start_training_v2.bat 会从最近检查点续训。";
    else hint = "—";
    $("hint").textContent = hint;

    drawCurve(d);
    renderVals(d.vals || [], d.best_step);
    renderFoot(d);
  }

  function renderVals(vals, bestStep) {
    var body = $("val-body");
    $("val-note").textContent = vals.length ? "共 " + vals.length + " 个" : "";
    if (!vals.length) {
      body.innerHTML = '<tr><td colspan="5" class="empty">暂无验证记录</td></tr>';
      return;
    }
    var tail = vals.slice(-8);
    body.innerHTML = tail.map(function (v) {
      var best = bestStep != null && v.step === bestStep;
      return '<tr' + (best ? ' class="is-best"' : "") + ">" +
        "<td>" + nf(v.step) + "</td>" +
        "<td>" + fx(v.train) + "</td>" +
        "<td>" + fx(v.val) + "</td>" +
        "<td>" + (v.gap > 0 ? "+" : "") + fx(v.gap) + "</td>" +
        "<td>" + fx(v.best) + "</td></tr>";
    }).join("");
  }

  function renderFoot(d) {
    var parts = [];
    if (d.log_file) parts.push("日志 " + d.log_file + " · " + (d.log_bytes / 1024).toFixed(1) + " KB");
    if (d.ckpt_file) parts.push("检查点 " + d.ckpt_file + (d.saved_at ? "（" + d.saved_at + "）" : ""));
    $("foot").textContent = parts.join("　·　") || "—";
  }

  /* ------------------------------------------------------------ 轮询 */
  function tick() {
    var url = "/api/train/status" + (currentRun ? "?run=" + encodeURIComponent(currentRun) : "");
    fetch(url, { cache: "no-store" })
      .then(function (r) {
        return r.json().then(function (j) {
          if (!r.ok) throw new Error(j.detail || ("HTTP " + r.status));
          return j;
        });
      })
      .then(render)
      .catch(function (err) {
        $("dot").setAttribute("data-state", "unknown");
        $("state-text").textContent = "读取失败";
        $("hint").textContent = err.message;
      });
  }

  function loadRuns() {
    return fetch("/api/train/runs", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (j) {
        var sel = $("run");
        sel.innerHTML = "";
        (j.items || []).forEach(function (it) {
          var o = document.createElement("option");
          o.value = it.run;
          o.textContent = it.run + (it.is_latest ? "（最新）" : "");
          sel.appendChild(o);
        });
        currentRun = j.latest || (j.items[0] && j.items[0].run) || null;
        if (currentRun) sel.value = currentRun;
      })
      .catch(function () { /* 下拉失败不影响主视图 */ });
  }

  function boot() {
    $("run").addEventListener("change", function (e) {
      currentRun = e.target.value;
      tick();
    });

    loadRuns().then(function () {
      tick();
      setInterval(tick, POLL_MS);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
