/* ============================================================
   PACT · The Offer Terminal
   "Every offer is a trade. This is the terminal."
   Frontend for the CTC offer intelligence backend.
   ============================================================ */

const { createElement: h, useEffect, useMemo, useRef, useState, useCallback } = React;

/* ------------------------------------------------------------
   API layer (unchanged backend contracts)
------------------------------------------------------------ */

const api = {
  summary: () => fetch("/api/summary").then((res) => res.json()),
  options: () => fetch("/api/options").then((res) => res.json()),
  candidates: () => fetch("/api/candidates?limit=30").then((res) => res.json()),
  chat: (messages, context) =>
    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, context: context || null }),
    }).then((res) => (res.ok ? res.json() : res.json().then((err) => Promise.reject(err)))),
  recommend: (payload) =>
    fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then((res) => (res.ok ? res.json() : res.json().then((err) => Promise.reject(err)))),
  negotiate: (payload) =>
    fetch("/api/negotiate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then((res) => (res.ok ? res.json() : res.json().then((err) => Promise.reject(err)))),
  riskScan: (params) =>
    fetch(`/api/risk-scan?${new URLSearchParams(params)}`).then((res) =>
      res.ok ? res.json() : res.json().then((err) => Promise.reject(err))
    ),
  githubScan: (username, skill) =>
    fetch(`/api/github-scan?${new URLSearchParams({ username, skill })}`).then((res) =>
      res.ok ? res.json() : res.json().then((err) => Promise.reject(err))
    ),
  marketWire: (params) =>
    fetch(`/api/market-wire?${new URLSearchParams(params)}`).then((res) =>
      res.ok ? res.json() : res.json().then((err) => Promise.reject(err))
    ),
  offerLetter: (candidate, quote) =>
    fetch("/api/offer-letter", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ candidate, quote }),
    }).then((res) => (res.ok ? res.json() : res.json().then((err) => Promise.reject(err)))),
};

/* ------------------------------------------------------------
   Voice Desk — Web Speech API (recognition + synthesis)
------------------------------------------------------------ */

const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition || null;

function speakText(text) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const clean = String(text || "")
    .replace(/[#*_`>|]/g, "")
    .replace(/\(https?:\/\/[^)]+\)/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 600);
  if (!clean) return;
  const utterance = new SpeechSynthesisUtterance(clean);
  utterance.rate = 1.05;
  utterance.pitch = 0.95;
  window.speechSynthesis.speak(utterance);
}

/* ------------------------------------------------------------
   Formatting helpers
------------------------------------------------------------ */

function fmtPct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${Math.round(Number(value) * 100)}%`;
}

function fmtLpa(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${Number(value).toFixed(2)}`;
}

function fmtLpaUnit(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${Number(value).toFixed(2)} LPA`;
}

const CHART = {
  green: "#21a06c",
  amber: "#c4862f",
  blue: "#4e8fe0",
  rose: "#d25572",
};

function probTone(p) {
  if (p === null || p === undefined) return "warn";
  if (p >= 0.7) return "good";
  if (p >= 0.45) return "warn";
  return "bad";
}

/* ------------------------------------------------------------
   Icons — lucide replaces <i data-lucide> after render.
   Every component with local state re-runs this on render.
------------------------------------------------------------ */

function useLucide() {
  useEffect(() => {
    if (window.lucide) window.lucide.createIcons();
  });
}

function Icon({ name, size = 16 }) {
  return h("i", { "data-lucide": name, style: { width: size, height: size } });
}

/* ------------------------------------------------------------
   Charts
------------------------------------------------------------ */

/**
 * Multi-series line chart with crosshair + tooltip.
 * series: [{ name, color, data: [{x, y, label?}] }] — x values aligned across series.
 * markers: [{ x, color, label }] vertical reference pins.
 */
function LineChart({ series, yAsPercent = false, xFormat, markers = [], height = 240 }) {
  const width = 680;
  const pad = { top: 18, right: 18, bottom: 30, left: 50 };
  const wrapRef = useRef(null);
  const [hoverIndex, setHoverIndex] = useState(null);

  const base = (series && series[0] && series[0].data) || [];
  if (!base.length) return h("div", { className: "empty" }, "NO DATA IN RANGE");

  const allY = series.flatMap((s) => s.data.map((d) => d.y));
  const xs = base.map((d) => d.x);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...allY, 0);
  const maxY = Math.max(...allY, yAsPercent ? 1 : 0);
  const xScale = (x) => pad.left + ((x - minX) / Math.max(maxX - minX, 1e-9)) * (width - pad.left - pad.right);
  const yScale = (y) => height - pad.bottom - ((y - minY) / Math.max(maxY - minY, 1e-9)) * (height - pad.top - pad.bottom);
  const fmtX = xFormat || ((x, i) => String(base[i] && base[i].label !== undefined ? base[i].label : x));
  const fmtY = (y) => (yAsPercent ? fmtPct(y) : Number(y).toFixed(1));

  function onMove(event) {
    const rect = wrapRef.current ? wrapRef.current.getBoundingClientRect() : null;
    if (!rect) return;
    const fx = ((event.clientX - rect.left) / rect.width) * width;
    let best = 0;
    let bestDist = Infinity;
    base.forEach((d, i) => {
      const dist = Math.abs(xScale(d.x) - fx);
      if (dist < bestDist) {
        bestDist = dist;
        best = i;
      }
    });
    setHoverIndex(best);
  }

  const gridYs = [0.25, 0.5, 0.75, 1].map((f) => minY + f * (maxY - minY));
  const hover = hoverIndex === null ? null : base[hoverIndex];

  const children = [];
  gridYs.forEach((gy, i) =>
    children.push(
      h("line", { key: `g${i}`, className: "grid-line", x1: pad.left, y1: yScale(gy), x2: width - pad.right, y2: yScale(gy) })
    )
  );
  children.push(
    h("line", { key: "ax", className: "axis-line", x1: pad.left, y1: height - pad.bottom, x2: width - pad.right, y2: height - pad.bottom }),
    h("line", { key: "ay", className: "axis-line", x1: pad.left, y1: pad.top, x2: pad.left, y2: height - pad.bottom })
  );

  markers.forEach((m, i) => {
    if (m.x < minX || m.x > maxX) return;
    children.push(
      h("line", {
        key: `m${i}`,
        x1: xScale(m.x),
        y1: pad.top,
        x2: xScale(m.x),
        y2: height - pad.bottom,
        stroke: m.color,
        strokeDasharray: "3 4",
        strokeWidth: 1.5,
        opacity: 0.85,
      }),
      h("text", { key: `mt${i}`, x: xScale(m.x) + 4, y: pad.top + 9, fill: m.color, style: { fill: m.color } }, m.label)
    );
  });

  series.forEach((s, si) => {
    const path = s.data.map((d, i) => `${i === 0 ? "M" : "L"} ${xScale(d.x).toFixed(1)} ${yScale(d.y).toFixed(1)}`).join(" ");
    children.push(
      h("path", { key: `p${si}`, d: path, fill: "none", stroke: s.color, strokeWidth: 2, strokeLinejoin: "round", strokeLinecap: "round" })
    );
    const last = s.data[s.data.length - 1];
    children.push(h("circle", { key: `e${si}`, cx: xScale(last.x), cy: yScale(last.y), r: 3.5, fill: s.color }));
    if (series.length > 1) {
      children.push(
        h(
          "text",
          { key: `dl${si}`, x: Math.min(xScale(last.x) + 6, width - 8), y: yScale(last.y) + 3, style: { fill: s.color }, textAnchor: "start" },
          s.name
        )
      );
    }
  });

  if (hover) {
    children.push(
      h("line", { key: "ch", x1: xScale(hover.x), y1: pad.top, x2: xScale(hover.x), y2: height - pad.bottom, stroke: "#3ddc97", strokeWidth: 1, opacity: 0.5 })
    );
    series.forEach((s, si) => {
      const d = s.data[hoverIndex];
      if (!d) return;
      children.push(
        h("circle", { key: `hc${si}`, cx: xScale(d.x), cy: yScale(d.y), r: 5, fill: s.color, stroke: "#10161c", strokeWidth: 2 })
      );
    });
  }

  children.push(
    h("text", { key: "x0", x: pad.left, y: height - 9 }, fmtX(base[0].x, 0)),
    h("text", { key: "x1", x: width - pad.right, y: height - 9, textAnchor: "end" }, fmtX(base[base.length - 1].x, base.length - 1)),
    h("text", { key: "ymax", x: pad.left - 8, y: pad.top + 4, textAnchor: "end" }, fmtY(maxY)),
    h("text", { key: "ymin", x: pad.left - 8, y: height - pad.bottom, textAnchor: "end" }, fmtY(minY))
  );

  const tipLeft = hover ? Math.min(88, Math.max(12, (xScale(hover.x) / width) * 100)) : 0;
  const tipTop = hover ? (Math.min(...series.map((s) => (s.data[hoverIndex] ? yScale(s.data[hoverIndex].y) : height))) / height) * 100 : 0;

  return h(
    "div",
    { className: "chart-wrap", ref: wrapRef, onMouseMove: onMove, onMouseLeave: () => setHoverIndex(null) },
    [
      h("svg", { key: "svg", className: "chart", viewBox: `0 0 ${width} ${height}`, role: "img" }, children),
      hover &&
        h("div", { key: "tip", className: "chart-tip", style: { left: `${tipLeft}%`, top: `${tipTop}%` } }, [
          h("div", { key: "l", className: "tip-label" }, fmtX(hover.x, hoverIndex)),
          ...series.map((s, si) =>
            h("div", { key: si }, `${series.length > 1 ? s.name + " " : ""}${fmtY(s.data[hoverIndex] ? s.data[hoverIndex].y : null)}`)
          ),
        ]),
    ]
  );
}

function Legend({ items }) {
  return h(
    "div",
    { className: "legend" },
    items.map((it) =>
      h("span", { className: "legend-chip", key: it.name }, [
        h("span", { key: "s", className: "swatch", style: { background: it.color } }),
        it.name,
      ])
    )
  );
}

/** Semicircular acceptance gauge. */
function Gauge({ value, caption }) {
  const v = Math.max(0, Math.min(1, Number(value) || 0));
  const cx = 110;
  const cy = 112;
  const r = 84;
  const angle = (frac) => Math.PI * (1 - frac);
  const px = (frac) => cx + r * Math.cos(angle(frac));
  const py = (frac) => cy - r * Math.sin(angle(frac));
  const arc = (from, to, color, widthPx) =>
    h("path", {
      d: `M ${px(from)} ${py(from)} A ${r} ${r} 0 0 1 ${px(to)} ${py(to)}`,
      fill: "none",
      stroke: color,
      strokeWidth: widthPx,
      strokeLinecap: "round",
      key: `${from}-${to}-${color}`,
    });

  const toneColor = v >= 0.7 ? "#3ddc97" : v >= 0.45 ? "#e8b45a" : "#f0637e";
  const needleA = angle(v);
  const nx = cx + (r - 16) * Math.cos(needleA);
  const ny = cy - (r - 16) * Math.sin(needleA);

  return h("div", { className: "gauge-box" }, [
    h("svg", { key: "svg", viewBox: "0 0 220 130", style: { width: "100%", maxWidth: 260 } }, [
      arc(0.001, 0.999, "#151d24", 12),
      v > 0.01 && arc(0.001, v, toneColor, 12),
      h("line", { key: "needle", x1: cx, y1: cy, x2: nx, y2: ny, stroke: "#e6edf3", strokeWidth: 2 }),
      h("circle", { key: "hub", cx, cy, r: 5, fill: "#e6edf3" }),
      h("text", { key: "t0", x: cx - r, y: cy + 14, textAnchor: "middle", style: { fill: "#4a5866", fontSize: 9 } }, "0%"),
      h("text", { key: "t1", x: cx + r, y: cy + 14, textAnchor: "middle", style: { fill: "#4a5866", fontSize: 9 } }, "100%"),
    ]),
    h("div", { key: "v", className: "gauge-value", style: { color: toneColor } }, fmtPct(v)),
    caption && h("div", { key: "c", className: "gauge-caption" }, caption),
  ]);
}

/** Benchmark band with P20/P50/P80 pins plus offer / suggestion markers. */
function RangeBand({ p20, p50, p80, offered, suggested }) {
  const points = [p20, p50, p80, offered, suggested].filter((x) => x !== null && x !== undefined && Number.isFinite(Number(x)));
  if (points.length < 2) return h("div", { className: "empty" }, "NOT ENOUGH BENCHMARK DATA");
  const lo = Math.min(...points);
  const hi = Math.max(...points);
  const span = Math.max(hi - lo, 1e-9);
  const padFrac = 0.08;
  const pos = (x) => `${((Number(x) - lo) / span) * (1 - 2 * padFrac) * 100 + padFrac * 100}%`;

  const pin = (x, label, cls, showValue = true) =>
    x === null || x === undefined
      ? null
      : h("div", { className: `range-pin ${cls || ""}`, style: { left: pos(x) }, key: `${label}` }, [
          h("span", { key: "l", className: "pin-label" }, label),
          showValue && h("span", { key: "v", className: "pin-value" }, fmtLpa(x)),
        ]);

  return h("div", { className: "range-band" }, [
    h("div", { className: "range-track", key: "track" }, [
      p20 !== null && p80 !== null
        ? h("div", {
            key: "fill",
            className: "range-fill",
            style: { left: pos(p20), width: `calc(${pos(p80)} - ${pos(p20)})` },
          })
        : null,
      pin(p20, "P20", ""),
      pin(p50, "P50", ""),
      pin(p80, "P80", ""),
      pin(offered, "OFFER", "marker-offer", false),
      pin(suggested, "SUGGEST", "marker-suggest", false),
    ]),
  ]);
}

function Histogram({ records }) {
  const values = (records || []).map((row) => Number(row.offered_ctc)).filter((v) => Number.isFinite(v));
  if (!values.length) return h("div", { className: "empty" }, "NO ACCEPTED PROFILES FOR THIS BENCHMARK");

  const min = Math.min(...values);
  const max = Math.max(...values);
  const binCount = Math.min(8, Math.max(3, Math.ceil(Math.sqrt(values.length))));
  const binWidth = Math.max((max - min) / binCount, 1);
  const bins = Array.from({ length: binCount }, (_, index) => {
    const start = min + index * binWidth;
    const end = index === binCount - 1 ? max : start + binWidth;
    return { start, end, count: 0 };
  });
  values.forEach((value) => {
    const index = Math.min(binCount - 1, Math.floor((value - min) / binWidth));
    bins[index].count += 1;
  });
  const maxCount = Math.max(...bins.map((b) => b.count), 1);

  return h(
    "div",
    { className: "histogram" },
    bins.map((bin) =>
      h("div", { className: "histogram-bin", key: `${bin.start}` }, [
        h("div", { key: "w", className: "histogram-bar-wrap" }, [
          h("div", {
            className: "histogram-bar",
            style: { height: `${Math.max(8, (bin.count / maxCount) * 100)}%` },
            title: `${bin.count} accepted/joined profiles`,
          }),
        ]),
        h("strong", { key: "c" }, bin.count),
        h("span", { key: "r" }, `${bin.start.toFixed(1)}–${bin.end.toFixed(1)}`),
      ])
    )
  );
}

function BarRow({ label, value, display, max, tone }) {
  const width = max ? Math.max(3, (value / max) * 100) : 0;
  return h("div", { className: "bar-row" }, [
    h("span", { key: "l", title: label }, label),
    h("div", { className: "bar-track", key: "t" }, [h("div", { className: `bar-fill ${tone || ""}`, style: { width: `${width}%` } })]),
    h("strong", { key: "v" }, display !== undefined ? display : value),
  ]);
}

/* ------------------------------------------------------------
   Live FX line (open.er-api.com via backend)
------------------------------------------------------------ */

function FxLine({ lpa }) {
  const [wire, setWire] = useState(null);
  useEffect(() => {
    let alive = true;
    if (!lpa) {
      setWire(null);
      return undefined;
    }
    api
      .marketWire({ lpa })
      .then((data) => {
        if (alive) setWire(data);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [lpa]);
  const c = wire && wire.conversions;
  if (!c || !c.usd) return null;
  return h(
    "div",
    { className: "fx-line" },
    `≈ $${c.usd.toLocaleString()} · €${c.eur.toLocaleString()} · £${c.gbp.toLocaleString()} / year · live FX`
  );
}

/* ------------------------------------------------------------
   GitHub Talent Scanner card (live GitHub REST API)
------------------------------------------------------------ */

function GithubSignalCard({ signal }) {
  if (!signal) return null;
  if (!signal.available) {
    return h(AlertLine, { tone: "warn", tag: "GITHUB" }, signal.reason || "GitHub scan unavailable.");
  }
  const verdictMap = {
    strong_match: { label: "STRONG MATCH", cls: "good" },
    partial_match: { label: "PARTIAL MATCH", cls: "warn" },
    mismatch: { label: "SKILL MISMATCH", cls: "bad" },
    no_signal: { label: "NO PUBLIC SIGNAL", cls: "" },
    unmapped: { label: "UNMAPPED SKILL", cls: "info" },
  };
  const verdict = verdictMap[signal.verdict] || { label: String(signal.verdict).toUpperCase(), cls: "" };
  const maxRepos = Math.max(...(signal.top_languages || []).map((l) => l.repos), 1);

  return h("div", { className: "panel quiet gh-card" }, [
    h("div", { className: "panel-head", key: "h" }, [
      h("h3", { key: "t" }, "Verified Skill Signal"),
      h("span", { key: "s" }, "live GitHub API · public footprint"),
    ]),
    h("div", { className: "gh-head", key: "id" }, [
      signal.avatar_url && h("img", { key: "av", className: "gh-avatar", src: signal.avatar_url, alt: "" }),
      h("div", { key: "who" }, [
        h("a", { key: "n", className: "gh-name", href: signal.profile_url, target: "_blank", rel: "noreferrer" }, signal.name || signal.username),
        h("div", { key: "u", className: "gh-login" }, `@${signal.username} · ${signal.account_years ?? "?"} yrs on GitHub`),
      ]),
      h("span", { key: "v", className: `tag ${verdict.cls}`, style: { marginLeft: "auto" } }, verdict.label),
    ]),
    h("div", { className: "stat-row", key: "stats" }, [
      h("div", { className: "stat", key: "1" }, [h("span", { key: "l" }, "Source Repos"), h("strong", { key: "v" }, String(signal.source_repos))]),
      h("div", { className: "stat", key: "2" }, [h("span", { key: "l" }, "Total Stars"), h("strong", { key: "v" }, signal.total_stars.toLocaleString())]),
      h("div", { className: "stat", key: "3" }, [h("span", { key: "l" }, "Active (90d)"), h("strong", { key: "v" }, String(signal.recent_active_repos_90d))]),
    ]),
    (signal.top_languages || []).length
      ? h("div", { className: "bars", key: "langs", style: { marginTop: 12 } },
          signal.top_languages.map((row) =>
            h(BarRow, { key: row.language, label: row.language, value: row.repos, max: maxRepos, tone: "good" })
          )
        )
      : null,
    h("p", { className: "explain", key: "note" }, signal.verdict_note),
  ]);
}

/* ------------------------------------------------------------
   Offer Letter Forge modal (LLM-drafted, QR via api.qrserver.com)
------------------------------------------------------------ */

function LetterModal({ letter, candidateName, skill, onClose }) {
  if (!letter) return null;
  const qrData = `PACT OFFER · ${candidateName || "Candidate"} · ${skill} · ${letter.ctc} LPA · ${String(letter.generated_at).slice(0, 10)}`;
  const qrSrc = `https://api.qrserver.com/v1/create-qr-code/?size=110x110&data=${encodeURIComponent(qrData)}`;
  const paragraphs = String(letter.letter || "").split("\n").filter((line) => line.trim().length > 0);
  const usd = letter.conversions && letter.conversions.usd;

  return h("div", { className: "letter-overlay", onClick: onClose }, [
    h("div", { className: "letter-sheet", key: "sheet", onClick: (e) => e.stopPropagation() }, [
      h("div", { className: "letter-head", key: "head" }, [
        h("div", { key: "brand" }, [
          h("div", { className: "letter-brand", key: "b" }, "PACT"),
          h("div", { className: "letter-brand-sub", key: "s" }, "TALENT ACQUISITION DESK"),
        ]),
        h("img", { key: "qr", className: "letter-qr", src: qrSrc, alt: "Offer QR code" }),
      ]),
      h("div", { className: "letter-meta", key: "meta" }, [
        h("span", { key: "d" }, `Date: ${String(letter.generated_at).slice(0, 10)}`),
        h("span", { key: "c" }, `Offer: ${fmtLpaUnit(letter.ctc)}${usd ? ` (≈ $${usd.toLocaleString()}/yr)` : ""}`),
      ]),
      h("div", { className: "letter-body", key: "body" }, paragraphs.map((line, i) => h("p", { key: i }, line))),
      h("div", { className: "letter-actions no-print", key: "act" }, [
        h("button", { className: "btn btn-primary", key: "p", onClick: () => window.print() }, "PRINT / SAVE PDF"),
        h("button", { className: "btn btn-ghost", key: "x", onClick: onClose }, "CLOSE"),
      ]),
    ]),
  ]);
}

/* ------------------------------------------------------------
   Shell: ticker, topbar, rail
------------------------------------------------------------ */

function TickerTape({ candidates, fx }) {
  const items = (candidates || []).slice(0, 24);
  if (!items.length) return null;
  const render = (row, key) => {
    const won = row.status === "Joined" || row.status === "Accepted";
    return h("span", { className: "ticker-item", key }, [
      h("b", { key: "r" }, row.candidate_ref),
      h("span", { key: "s" }, row.primary_skill),
      h("b", { key: "c" }, `${fmtLpa(row.offered_ctc)} LPA`),
      h("span", { key: "st", className: won ? "t-up" : "t-down" }, won ? `▲ ${row.status.toUpperCase()}` : `▼ ${row.status.toUpperCase()}`),
      h("span", { key: "sep", className: "ticker-sep" }, "//"),
    ]);
  };
  const renderFx = (key) => {
    if (!fx || !fx.usd) return null;
    return h("span", { className: "ticker-item", key }, [
      h("span", { key: "l", className: "t-up" }, "◉ LIVE FX"),
      h("b", { key: "u" }, `USD/INR ${(1 / fx.usd).toFixed(2)}`),
      h("b", { key: "e" }, `EUR/INR ${(1 / fx.eur).toFixed(2)}`),
      h("b", { key: "g" }, `GBP/INR ${(1 / fx.gbp).toFixed(2)}`),
      h("span", { key: "sep", className: "ticker-sep" }, "//"),
    ]);
  };
  const half = (prefix) => [renderFx(`${prefix}fx`), ...items.map((row, i) => render(row, `${prefix}${i}`))].filter(Boolean);
  const track = [...half("a"), ...half("b")];
  return h("div", { className: "ticker" }, h("div", { className: "ticker-track" }, track));
}

function Clock() {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  const pad2 = (n) => String(n).padStart(2, "0");
  return h("span", { className: "clock" }, `${pad2(now.getHours())}:${pad2(now.getMinutes())}:${pad2(now.getSeconds())} IST`);
}

function Topbar({ summary, onOpenCmdk, onVoice }) {
  return h("header", { className: "topbar" }, [
    h("div", { className: "brand", key: "brand" }, [
      h("h1", { className: "brand-name", key: "n" }, ["PACT", h("span", { className: "cursor", key: "c" }, "▌")]),
      h("span", { className: "brand-sub", key: "s" }, "The Offer Terminal"),
    ]),
    h("div", { className: "top-meta", key: "meta" }, [
      SpeechRec &&
        h("button", { className: "cmdk-btn mic-top", key: "mic", onClick: onVoice, title: "Talk to the desk" }, [
          h(Icon, { name: "mic", size: 13, key: "i" }),
          "VOICE",
        ]),
      h("span", { className: "chip", key: "live" }, [h("span", { className: "live-dot", key: "d" }), "DESK LIVE"]),
      h("span", { className: "chip", key: "n" }, [h(Icon, { name: "database", size: 13, key: "i" }), h("b", { key: "b" }, summary.kpis.total_offers), "OFFERS"]),
      h("span", { className: "chip", key: "m" }, [
        h(Icon, { name: "activity", size: 13, key: "i" }),
        "MODEL AUC ",
        h("b", { key: "b" }, summary.model_metrics.roc_auc),
      ]),
      h("button", { className: "cmdk-btn", key: "cmdk", onClick: onOpenCmdk }, [
        h(Icon, { name: "command", size: 13, key: "i" }),
        "COMMAND",
        h("kbd", { key: "k" }, "CTRL K"),
      ]),
      h(Clock, { key: "clock" }),
    ]),
  ]);
}

const NAV = [
  { key: "dashboard", label: "Pulse", icon: "activity", num: "1", eyebrow: "Market Overview" },
  { key: "simulator", label: "Offer Studio", icon: "pencil-ruler", num: "2", eyebrow: "Price a Candidate" },
  { key: "negotiation", label: "The Arena", icon: "swords", num: "3", eyebrow: "Agent vs Agent" },
  { key: "risk", label: "Risk Radar", icon: "radar", num: "4", eyebrow: "Autonomous Watch" },
  { key: "table", label: "The Ledger", icon: "scroll-text", num: "5", eyebrow: "Every Trade" },
];

function Rail({ activeTab, onNavigate }) {
  return h("nav", { className: "rail" }, [
    h("div", { className: "rail-caption", key: "cap" }, "DESK"),
    ...NAV.map((item) =>
      h(
        "button",
        {
          key: item.key,
          className: `rail-btn ${activeTab === item.key ? "active" : ""}`,
          onClick: () => onNavigate(item.key),
        },
        [h(Icon, { name: item.icon, size: 16, key: "i" }), item.label, h("span", { className: "rail-key", key: "k" }, item.num)]
      )
    ),
    h("div", { className: "rail-foot", key: "foot" }, [
      h("div", { key: "1" }, ["ENGINE ", h("b", { key: "b" }, "LOGIT-P(ACCEPT)")]),
      h("div", { key: "2" }, ["AGENTS ", h("b", { key: "b" }, "3 ACTIVE")]),
      h("div", { key: "3" }, ["MODE ", h("b", { key: "b" }, "ADVISORY")]),
    ]),
  ]);
}

function ViewHead({ eyebrow, title, blurb }) {
  return h("div", { className: "view-head" }, [
    h("div", { className: "eyebrow", key: "e" }, eyebrow),
    h("h2", { key: "t" }, title),
    blurb && h("p", { key: "b" }, blurb),
  ]);
}

function AlertLine({ tone = "warn", tag, children }) {
  return h("div", { className: `alert ${tone}` }, [
    h("span", { className: "alert-tag", key: "t" }, tag || tone.toUpperCase()),
    h("span", { key: "c" }, children),
  ]);
}

/* ------------------------------------------------------------
   View 1 — Pulse (dashboard)
------------------------------------------------------------ */

function PulseView({ summary, candidates, onNavigate }) {
  useLucide();
  const kpis = summary.kpis;
  const maxStatus = Math.max(...Object.values(summary.status_counts));
  const maxBand = Math.max(...summary.by_band.map((r) => r.offers));
  const maxSource = Math.max(...summary.by_source.map((r) => r.offers));
  const trendData = (summary.trend || []).map((row, index) => ({ x: index, y: row.acceptance_rate, label: row.period }));
  const pct = summary.accepted_ctc_percentiles;

  return h("div", null, [
    h(ViewHead, {
      key: "head",
      eyebrow: "01 · PULSE",
      title: "The talent market, on one screen.",
      blurb: "Live read of offer outcomes, price levels, and model health across the hiring desk.",
    }),
    summary.insight &&
      h("div", { className: "signal", key: "signal" }, [
        h("span", { className: "signal-tag", key: "t" }, "◈ SIGNAL"),
        h("span", { key: "c" }, summary.insight),
      ]),
    h("div", { className: "hero-grid", key: "kpis" }, [
      h("div", { className: "hero-stat", key: "a" }, [
        h("div", { className: "hero-label", key: "l" }, "Acceptance Rate"),
        h("div", { className: "hero-value", key: "v" }, fmtPct(kpis.acceptance_rate)),
        h("div", { className: "hero-note", key: "n" }, `${kpis.accepted_or_joined} of ${kpis.total_offers} offers won`),
      ]),
      h("div", { className: "hero-stat blue", key: "b" }, [
        h("div", { className: "hero-label", key: "l" }, "Median Offer"),
        h("div", { className: "hero-value", key: "v" }, [fmtLpa(kpis.median_offered_ctc), h("small", { key: "u" }, "LPA")]),
        h("div", { className: "hero-note", key: "n" }, "Across all historical offers"),
      ]),
      h("div", { className: "hero-stat amber", key: "c" }, [
        h("div", { className: "hero-label", key: "l" }, "Avg Hike Paid"),
        h("div", { className: "hero-value", key: "v" }, [String(kpis.avg_offered_hike_pct), h("small", { key: "u" }, "%")]),
        h("div", { className: "hero-note", key: "n" }, "Premium over current CTC"),
      ]),
      h("div", { className: "hero-stat rose", key: "d" }, [
        h("div", { className: "hero-label", key: "l" }, "Lost Offers"),
        h("div", { className: "hero-value", key: "v" }, String(kpis.declined_or_no_show)),
        h("div", { className: "hero-note", key: "n" }, `${kpis.no_show} silent no-shows included`),
      ]),
    ]),
    h("section", { className: "section grid-2", key: "row1" }, [
      h("div", { className: "panel", key: "trend" }, [
        h("div", { className: "panel-head", key: "h" }, [h("h3", { key: "t" }, "Acceptance Trend"), h("span", { key: "s" }, "monthly win rate")]),
        h(LineChart, {
          key: "c",
          series: [{ name: "Acceptance", color: CHART.green, data: trendData }],
          yAsPercent: true,
          xFormat: (x, i) => (summary.trend[i] ? summary.trend[i].period : String(x)),
        }),
      ]),
      h("div", { className: "panel", key: "funnel" }, [
        h("div", { className: "panel-head", key: "h" }, [h("h3", { key: "t" }, "Outcome Book"), h("span", { key: "s" }, "count by status")]),
        h(
          "div",
          { className: "bars", key: "bars" },
          Object.entries(summary.status_counts).map(([label, value]) =>
            h(BarRow, {
              key: label,
              label,
              value,
              max: maxStatus,
              tone: label === "Joined" || label === "Accepted" ? "good" : label === "No Show" ? "bad" : "warn",
            })
          )
        ),
        h("div", { style: { marginTop: 14 }, key: "range" }, [
          h("div", { className: "panel-head", key: "h" }, [h("h3", { key: "t" }, "Winning Price Band"), h("span", { key: "s" }, "accepted CTC · LPA")]),
          h(RangeBand, { key: "b", p20: pct.p20, p50: pct.p50, p80: pct.p80 }),
        ]),
      ]),
    ]),
    h("section", { className: "section grid-3", key: "row2" }, [
      h("div", { className: "panel quiet", key: "band" }, [
        h("div", { className: "panel-head", key: "h" }, [h("h3", { key: "t" }, "By Band"), h("span", { key: "s" }, "volume · win rate")]),
        h(
          "div",
          { className: "bars", key: "b" },
          summary.by_band.map((row) =>
            h(BarRow, { key: row.band, label: `${row.band} · ${fmtPct(row.acceptance_rate)}`, value: row.offers, max: maxBand, tone: "good" })
          )
        ),
      ]),
      h("div", { className: "panel quiet", key: "src" }, [
        h("div", { className: "panel-head", key: "h" }, [h("h3", { key: "t" }, "By Source"), h("span", { key: "s" }, "volume · win rate")]),
        h(
          "div",
          { className: "bars", key: "b" },
          summary.by_source.map((row) =>
            h(BarRow, { key: row.source, label: `${row.source} · ${fmtPct(row.acceptance_rate)}`, value: row.offers, max: maxSource, tone: "" })
          )
        ),
      ]),
      h("div", { className: "panel quiet", key: "model" }, [
        h("div", { className: "panel-head", key: "h" }, [h("h3", { key: "t" }, "Engine Health"), h("span", { key: "s" }, "logistic acceptance model")]),
        h("div", { className: "stat-row", key: "s", style: { gridTemplateColumns: "1fr" } }, [
          h("div", { className: "stat", key: "1" }, [
            h("span", { key: "l" }, "ROC AUC"),
            h("strong", { key: "v" }, String(summary.model_metrics.roc_auc)),
            h("em", { key: "e" }, "Discrimination — how well the model ranks winners"),
          ]),
          h("div", { className: "stat", key: "2" }, [
            h("span", { key: "l" }, "Brier Score"),
            h("strong", { key: "v" }, String(summary.model_metrics.brier_score)),
            h("em", { key: "e" }, "Calibration — lower is better"),
          ]),
          h("div", { className: "stat", key: "3" }, [
            h("span", { key: "l" }, "Test Records"),
            h("strong", { key: "v" }, String(summary.model_metrics.test_records)),
            h("em", { key: "e" }, "Held-out validation set"),
          ]),
        ]),
      ]),
    ]),
    h("section", { className: "section", key: "recent" }, [
      h("div", { className: "panel", key: "p" }, [
        h("div", { className: "panel-head", key: "h" }, [
          h("h3", { key: "t" }, "Desk Activity"),
          h(
            "button",
            { key: "b", className: "btn btn-ghost", style: { height: 28, padding: "0 12px", fontSize: 10 }, onClick: () => onNavigate("table") },
            "OPEN LEDGER →"
          ),
        ]),
        h(LedgerTable, { key: "tbl", candidates: candidates.slice(0, 8) }),
      ]),
    ]),
  ]);
}

/* ------------------------------------------------------------
   Shared candidate form
------------------------------------------------------------ */

function pickOption(values, preferred) {
  if (!values || !values.length) return preferred;
  return values.includes(preferred) ? preferred : values[0];
}

function useCandidateForm(options, extraDefaults = {}) {
  const defaults = useMemo(
    () => ({
      current_ctc: 12,
      expected_ctc: 17,
      offered_ctc: 16,
      relevant_experience_years: 6,
      notice_period_days: 30,
      offered_band: pickOption(options.offered_band, "E2"),
      candidate_source: pickOption(options.candidate_source, "Direct"),
      lob: pickOption(options.lob, "Digital"),
      primary_skill: pickOption(options.primary_skill, "Java Spring"),
      previous_company_type: pickOption(options.previous_company_type, "Service"),
      location: pickOption(options.location, "Bangalore"),
      joining_bonus: 0,
      relocation: 0,
      ...extraDefaults,
    }),
    [options]
  );
  const [form, setForm] = useState(defaults);
  useEffect(() => setForm(defaults), [defaults]);
  const setField = (key, value) => setForm((prev) => ({ ...prev, [key]: value }));
  return [form, setField, setForm];
}

function FormField({ kind, form, setField, k, label, values, step }) {
  if (kind === "select") {
    return h("div", { className: "field" }, [
      h("label", { key: "l" }, label),
      h(
        "select",
        { key: "i", value: form[k], onChange: (e) => setField(k, e.target.value) },
        (values || []).map((v) => h("option", { key: String(v), value: v }, String(v)))
      ),
    ]);
  }
  return h("div", { className: "field" }, [
    h("label", { key: "l" }, label),
    h("input", { key: "i", type: "number", step: step || "0.1", value: form[k], onChange: (e) => setField(k, e.target.value) }),
  ]);
}

function numericPayload(form) {
  return {
    ...form,
    current_ctc: Number(form.current_ctc),
    expected_ctc: Number(form.expected_ctc),
    offered_ctc: Number(form.offered_ctc),
    relevant_experience_years: Number(form.relevant_experience_years),
    notice_period_days: Number(form.notice_period_days),
    joining_bonus: Number(form.joining_bonus),
    relocation: Number(form.relocation),
  };
}

/* ------------------------------------------------------------
   View 2 — Offer Studio (simulator)
------------------------------------------------------------ */

function StudioView({ options, prefill, onQuote }) {
  useLucide();
  const [form, setField, setForm] = useCandidateForm(options, { flexibility: "balanced", candidate_name: "", github_username: "" });
  const [result, setResult] = useState(null);
  const [ghSignal, setGhSignal] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const runQuote = useCallback(
    (candidateForm) => {
      setError(null);
      setBusy(true);
      const { candidate_name, github_username, ...candidate } = candidateForm;
      api
        .recommend(numericPayload(candidate))
        .then((data) => {
          setResult(data);
          if (onQuote) onQuote(data);
        })
        .catch((err) => setError(err.detail || "Unable to price this candidate"))
        .finally(() => setBusy(false));
      if (github_username && github_username.trim()) {
        setGhSignal({ pending: true });
        api
          .githubScan(github_username.trim(), candidateForm.primary_skill)
          .then(setGhSignal)
          .catch(() => setGhSignal({ available: false, reason: "GitHub scan failed." }));
      } else {
        setGhSignal(null);
      }
    },
    [onQuote]
  );

  // Copilot-driven prefill: the agent fills the console and (optionally) runs it.
  useEffect(() => {
    if (!prefill || !prefill.nonce) return;
    const merged = { ...form, ...(prefill.fields || {}) };
    setForm(merged);
    if (prefill.run) runQuote(merged);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prefill && prefill.nonce]);

  function submit(event) {
    event.preventDefault();
    runQuote(form);
  }

  const f = (kind, k, label, values, step) => h(FormField, { kind, form, setField, k, label, values, step, key: k });
  const textField = (k, label, placeholder) =>
    h("div", { className: "field", key: k }, [
      h("label", { key: "l" }, label),
      h("input", { key: "i", type: "text", placeholder, value: form[k], onChange: (e) => setField(k, e.target.value) }),
    ]);

  return h("div", null, [
    h(ViewHead, {
      key: "head",
      eyebrow: "02 · OFFER STUDIO",
      title: "Price the candidate like an asset.",
      blurb: "Enter a profile and the desk quotes a CTC — with the acceptance probability, the benchmark band, and every reason behind it.",
    }),
    h("div", { className: "workbench", key: "bench" }, [
      h("form", { className: "console", key: "console", onSubmit: submit }, [
        h("div", { className: "panel-head", key: "h" }, [h("h3", { key: "t" }, "Candidate Console"), h("span", { key: "s" }, "all CTC in LPA")]),
        h("div", { className: "form-grid", key: "grid" }, [
          f("input", "current_ctc", "Current CTC"),
          f("input", "expected_ctc", "Expected CTC"),
          f("input", "offered_ctc", "Offer on Table"),
          f("input", "relevant_experience_years", "Experience (yrs)"),
          f("input", "notice_period_days", "Notice (days)", null, "1"),
          f("select", "offered_band", "Band", options.offered_band),
          f("select", "lob", "Line of Business", options.lob),
          f("select", "primary_skill", "Primary Skill", options.primary_skill),
          f("select", "location", "Location", options.location),
          f("select", "previous_company_type", "Company Type", options.previous_company_type),
          f("select", "candidate_source", "Source", options.candidate_source),
          f("select", "flexibility", "Benchmark Mode", ["strict", "balanced", "broad"]),
          f("select", "joining_bonus", "Joining Bonus", [0, 1]),
          f("select", "relocation", "Relocation", [0, 1]),
          textField("candidate_name", "Candidate Name (optional)", "for the offer letter"),
          textField("github_username", "GitHub (optional)", "username to verify skill"),
        ]),
        h("div", { className: "actions", key: "act" }, [
          h("button", { className: "btn btn-primary", type: "submit", disabled: busy, key: "b" }, [
            h(Icon, { name: "zap", size: 14, key: "i" }),
            busy ? "PRICING…" : "QUOTE THIS OFFER",
          ]),
        ]),
        error && h(AlertLine, { tone: "bad", tag: "ERROR", key: "err" }, error),
      ]),
      h(
        "div",
        { key: "result" },
        result
          ? h(StudioResult, { result, ghSignal, candidateName: form.candidate_name })
          : h("div", { className: "empty", style: { padding: "80px 20px" } }, "RUN A CANDIDATE THROUGH THE CONSOLE TO GET A QUOTE ▌")
      ),
    ]),
  ]);
}

function StudioResult({ result, ghSignal, candidateName }) {
  const p = result.percentile_recommendation || {};
  const benchmarkRecords = result.accepted_benchmark_records || [];
  const status = result.recommendation_status;
  const isOk = status === "ok";
  const isEscalation = String(status).startsWith("escalate") || status === "no_target_in_range" || status === "insufficient_data";
  const tone = isOk ? "" : isEscalation ? "bad" : "warn";
  const stampText = isOk ? "CLEARED" : isEscalation ? "ESCALATE" : "REVIEW";
  const filters = Object.entries(p.filters_used || {});
  const acceptedCount = p.accepted_similar_records !== undefined ? p.accepted_similar_records : benchmarkRecords.length;
  const warnings = result.warnings || [];
  const probSug = result.probability_at_suggested_ctc;
  const [showFallback, setShowFallback] = useState(false);
  const [letter, setLetter] = useState(null);
  const [letterError, setLetterError] = useState(null);
  const [forging, setForging] = useState(false);
  const quotedCtc = result.suggested_ctc !== null && result.suggested_ctc !== undefined ? result.suggested_ctc : result.candidate.offered_ctc;

  function forgeLetter() {
    setForging(true);
    setLetterError(null);
    const candidate = { ...result.candidate, candidate_name: candidateName || "" };
    api
      .offerLetter(candidate, { suggested_ctc: quotedCtc })
      .then(setLetter)
      .catch((err) => setLetterError(err.detail || "The letter forge is unavailable right now."))
      .finally(() => setForging(false));
  }

  return h("div", null, [
    /* --- deal ticket --- */
    h("div", { className: `ticket ${tone}`, key: "ticket" }, [
      h("div", { className: "ticket-head", key: "h" }, [
        h("div", { key: "l" }, [
          h("div", { className: "ticket-eyebrow", key: "e" }, "DESK QUOTE · SUGGESTED CTC"),
          h("div", { className: "ticket-ctc", key: "v" }, [
            result.suggested_ctc === null ? "—" : fmtLpa(result.suggested_ctc),
            h("small", { key: "u" }, " LPA"),
          ]),
          h(FxLine, { key: "fx", lpa: result.suggested_ctc }),
        ]),
        h("div", { key: "r", style: { display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 10 } }, [
          h("span", { className: `stamp ${tone}`, key: "stamp" }, stampText),
          h(
            "button",
            { className: "btn btn-ghost", key: "forge", style: { height: 30, fontSize: 10 }, onClick: forgeLetter, disabled: forging },
            forging ? "FORGING…" : "⚒ FORGE OFFER LETTER"
          ),
        ]),
      ]),
      h("div", { className: "ticket-row", key: "cells" }, [
        h("div", { className: "ticket-cell", key: "1" }, [
          h("span", { key: "l" }, "Offer on Table"),
          h("strong", { key: "v" }, fmtLpaUnit(result.candidate.offered_ctc)),
        ]),
        h("div", { className: "ticket-cell", key: "2" }, [
          h("span", { key: "l" }, "P(Accept) Now"),
          h("strong", { key: "v", className: probTone(result.acceptance_probability) === "good" ? "up" : probTone(result.acceptance_probability) === "bad" ? "down" : "" }, fmtPct(result.acceptance_probability)),
        ]),
        h("div", { className: "ticket-cell", key: "3" }, [
          h("span", { key: "l" }, "P(Accept) at Quote"),
          h("strong", { key: "v", className: probTone(probSug) === "good" ? "up" : "" }, fmtPct(probSug)),
        ]),
        h("div", { className: "ticket-cell", key: "4" }, [
          h("span", { key: "l" }, "Market P50"),
          h("strong", { key: "v" }, fmtLpaUnit(p.p50_offered_ctc)),
        ]),
        h("div", { className: "ticket-cell", key: "5" }, [
          h("span", { key: "l" }, "Confidence"),
          h("strong", { key: "v" }, `${p.specificity || "—"}`),
        ]),
      ]),
    ]),
    !isOk && h(AlertLine, { tone: isEscalation ? "bad" : "warn", tag: stampText, key: "status" }, result.recommendation_message),
    letterError && h(AlertLine, { tone: "bad", tag: "FORGE", key: "letter-err" }, letterError),

    /* --- GitHub talent scanner --- */
    ghSignal &&
      h("section", { className: "section", key: "github", style: { marginTop: 14 } },
        ghSignal.pending
          ? h("div", { className: "empty" }, "SCANNING GITHUB FOOTPRINT ▌")
          : h(GithubSignalCard, { signal: ghSignal })
      ),

    /* --- gauge + market band --- */
    h("section", { className: "section grid-2", key: "row1", style: { marginTop: 14 } }, [
      h("div", { className: "panel", key: "gauge" }, [
        h("div", { className: "panel-head", key: "h" }, [h("h3", { key: "t" }, "Acceptance Meter"), h("span", { key: "s" }, "at the suggested quote")]),
        h(Gauge, { key: "g", value: probSug !== null && probSug !== undefined ? probSug : result.acceptance_probability, caption: "predicted P(accept)" }),
      ]),
      h("div", { className: "panel", key: "band" }, [
        h("div", { className: "panel-head", key: "h" }, [
          h("h3", { key: "t" }, "Market Band"),
          h("span", { key: "s" }, `${acceptedCount} winning offers · ${p.specificity || ""}`),
        ]),
        h(RangeBand, {
          key: "b",
          p20: p.p20_offered_ctc,
          p50: p.p50_offered_ctc,
          p80: p.p80_offered_ctc,
          offered: result.candidate.offered_ctc,
          suggested: result.suggested_ctc,
        }),
        h("p", { className: "explain", key: "e" }, `Band built from accepted/joined offers matching: ${
          filters.length ? filters.map(([k, v]) => `${k}=${v}`).join(", ") : "no filters"
        }.`),
      ]),
    ]),

    /* --- probability curve --- */
    h("section", { className: "section", key: "curve" }, [
      h("div", { className: "panel", key: "p" }, [
        h("div", { className: "panel-head", key: "h" }, [h("h3", { key: "t" }, "Price ↔ Probability Curve"), h("span", { key: "s" }, "drag the slider to test any price")]),
        h(CurveExplorer, {
          key: "c",
          curve: result.acceptance_curve,
          offered: result.candidate.offered_ctc,
          suggested: result.suggested_ctc,
        }),
      ]),
    ]),

    /* --- warnings --- */
    warnings.length
      ? h("section", { className: "section", key: "warnings" }, [
          h("div", { className: "panel quiet", key: "p" }, [
            h("div", { className: "panel-head", key: "h" }, [h("h3", { key: "t" }, "Desk Notes"), h("span", { key: "s" }, `${warnings.length} caution(s) from the engine`)]),
            ...warnings.map((w, i) => h(AlertLine, { tone: "warn", tag: "NOTE", key: i }, w)),
          ]),
        ])
      : null,

    /* --- evidence --- */
    h("section", { className: "section grid-2", key: "evidence" }, [
      h("div", { className: "panel quiet", key: "hist" }, [
        h("div", { className: "panel-head", key: "h" }, [h("h3", { key: "t" }, "Winning Price Distribution"), h("span", { key: "s" }, "accepted profiles per CTC bin")]),
        h(Histogram, { key: "c", records: benchmarkRecords }),
      ]),
      h("div", { className: "panel quiet", key: "coverage" }, [
        h("div", { className: "panel-head", key: "h" }, [h("h3", { key: "t" }, "Evidence Coverage"), h("span", { key: "s" }, "why trust this quote")]),
        h("div", { className: "stat-row", key: "s1" }, [
          h("div", { className: "stat", key: "1" }, [
            h("span", { key: "l" }, "Skill + LOB"),
            h("strong", { key: "v" }, String(result.profile_match.skill_lob_records)),
            h("em", { key: "e" }, "records with this skill in this business"),
          ]),
          h("div", { className: "stat", key: "2" }, [
            h("span", { key: "l" }, "Exact Profile"),
            h("strong", { key: "v" }, String(result.profile_match.exact_profile_records)),
            h("em", { key: "e" }, "skill + LOB + location + band"),
          ]),
          h("div", { className: "stat", key: "3" }, [
            h("span", { key: "l" }, `± Experience`),
            h("strong", { key: "v" }, String(result.profile_match.experience_band_records)),
            h("em", { key: "e" }, `same profile at ${result.profile_match.experience_band}`),
          ]),
        ]),
        h("div", { className: "stat-row", key: "s2", style: { marginTop: 10 } }, [
          h("div", { className: "stat", key: "1" }, [
            h("span", { key: "l" }, "70% Threshold"),
            h("strong", { key: "v" }, fmtLpaUnit(result.target_offer_ctc)),
            h("em", { key: "e" }, "model price for target probability"),
          ]),
          h("div", { className: "stat", key: "2" }, [
            h("span", { key: "l" }, "Max Searched"),
            h("strong", { key: "v" }, fmtLpaUnit(result.curve_max_offer_ctc)),
            h("em", { key: "e" }, `P(accept) ${fmtPct(result.probability_at_curve_max)} at ceiling`),
          ]),
          h("div", { className: "stat", key: "3" }, [
            h("span", { key: "l" }, "Similar Offers"),
            h("strong", { key: "v" }, String(p.similar_records !== undefined ? p.similar_records : "—")),
            h("em", { key: "e" }, "before the accepted-only filter"),
          ]),
        ]),
        h("div", { style: { marginTop: 12 }, key: "tags" }, [
          h("div", { className: "filter-tags", key: "f" }, filters.map(([k, v]) => h("span", { className: "tag", key: k }, `${k}: ${v}`))),
        ]),
      ]),
    ]),

    /* --- benchmark table --- */
    h("section", { className: "section", key: "bench" }, [
      h("div", { className: "panel quiet", key: "p" }, [
        h("div", { className: "panel-head", key: "h" }, [
          h("h3", { key: "t" }, "The Comparables"),
          h("span", { key: "s" }, `${benchmarkRecords.length} winning offers behind the band`),
        ]),
        benchmarkRecords.length
          ? h("div", { className: "table-wrap", key: "t" }, [
              h("table", null, [
                h(
                  "thead",
                  { key: "th" },
                  h(
                    "tr",
                    null,
                    ["Ref", "Date", "Skill", "Location", "Band", "Exp", "Offered", "Hike", "Status"].map((head) => h("th", { key: head }, head))
                  )
                ),
                h(
                  "tbody",
                  { key: "tb" },
                  benchmarkRecords.slice(0, 10).map((row) =>
                    h("tr", { key: `${row.candidate_ref}-${row.offer_date}` }, [
                      h("td", { className: "num", key: "1" }, row.candidate_ref),
                      h("td", { className: "num", key: "2" }, row.offer_date),
                      h("td", { key: "3" }, row.primary_skill),
                      h("td", { key: "4" }, row.location),
                      h("td", { key: "5" }, h("span", { className: "tag" }, row.offered_band)),
                      h("td", { className: "num", key: "6" }, `${Number(row.relevant_experience_years).toFixed(1)}y`),
                      h("td", { className: "num", key: "7" }, fmtLpaUnit(row.offered_ctc)),
                      h("td", { className: "num", key: "8" }, `${Number(row.offered_hike_pct).toFixed(1)}%`),
                      h("td", { key: "9" }, h("span", { className: `tag ${row.status === "Joined" ? "good" : "info"}` }, row.status)),
                    ])
                  )
                ),
              ]),
            ])
          : h("div", { className: "empty", key: "e" }, "NO COMPARABLE WINNING OFFERS FOUND"),
        h(
          "button",
          { key: "fb", className: "btn btn-ghost", style: { marginTop: 12, height: 30, fontSize: 10 }, onClick: () => setShowFallback(!showFallback) },
          showFallback ? "HIDE BENCHMARK SEARCH TRAIL" : "SHOW BENCHMARK SEARCH TRAIL"
        ),
        showFallback && h(FallbackTable, { key: "fbt", attempts: p.fallback_attempts || [] }),
      ]),
    ]),

    /* --- offer letter modal --- */
    letter &&
      h(LetterModal, {
        key: "letter",
        letter,
        candidateName,
        skill: result.candidate.primary_skill,
        onClose: () => setLetter(null),
      }),
  ]);
}

function CurveExplorer({ curve, offered, suggested }) {
  const values = curve || [];
  const target = suggested !== null && suggested !== undefined ? suggested : offered;
  const initialIndex = values.length
    ? values.reduce((best, row, index) => (Math.abs(row.offered_ctc - target) < Math.abs(values[best].offered_ctc - target) ? index : best), 0)
    : 0;
  const [index, setIndex] = useState(initialIndex);
  useEffect(() => {
    if (values.length) setIndex(initialIndex);
  }, [initialIndex, values.length]);

  if (!values.length) return h("div", { className: "empty" }, "NO CURVE DATA");
  const selected = values[Math.max(0, Math.min(values.length - 1, Number(index)))];
  const data = values.map((d) => ({ x: d.offered_ctc, y: d.acceptance_probability }));
  const markers = [
    { x: offered, color: "#e8b45a", label: "OFFER" },
    suggested !== null && suggested !== undefined ? { x: suggested, color: "#3ddc97", label: "QUOTE" } : null,
    { x: selected.offered_ctc, color: "#6cb0ff", label: "" },
  ].filter(Boolean);

  return h("div", null, [
    h(LineChart, {
      key: "chart",
      series: [{ name: "P(accept)", color: CHART.green, data }],
      yAsPercent: true,
      xFormat: (x) => `${Number(x).toFixed(1)} LPA`,
      markers,
    }),
    h("div", { className: "slider-panel", key: "slider" }, [
      h("input", {
        key: "r",
        type: "range",
        min: 0,
        max: values.length - 1,
        value: index,
        onChange: (event) => setIndex(Number(event.target.value)),
        "aria-label": "Inspect offer CTC on probability curve",
      }),
      h("div", { className: "slider-readout", key: "read" }, [
        h("div", { key: "1" }, [h("span", { key: "l" }, "Test Price"), h("strong", { key: "v" }, fmtLpaUnit(selected.offered_ctc))]),
        h("div", { key: "2" }, [h("span", { key: "l" }, "Predicted Acceptance"), h("strong", { key: "v", style: { color: selected.acceptance_probability >= 0.7 ? "#3ddc97" : selected.acceptance_probability >= 0.45 ? "#e8b45a" : "#f0637e" } }, fmtPct(selected.acceptance_probability))]),
      ]),
    ]),
  ]);
}

function FallbackTable({ attempts }) {
  if (!attempts.length) return h("div", { className: "empty", style: { marginTop: 10 } }, "NO FALLBACK ATTEMPTS RECORDED");
  return h("div", { className: "table-wrap", style: { marginTop: 10 } }, [
    h("table", null, [
      h(
        "thead",
        { key: "h" },
        h("tr", null, ["#", "Similarity Rule", "Filters", "Similar", "Won", "Win Rate", "P20", "P50", "P80"].map((head) => h("th", { key: head }, head)))
      ),
      h(
        "tbody",
        { key: "b" },
        attempts.map((attempt, index) =>
          h("tr", { key: index }, [
            h("td", { className: "num", key: "0" }, index + 1),
            h("td", { key: "1" }, attempt.similarity_rule || "—"),
            h(
              "td",
              { key: "2" },
              Object.entries(attempt.filters_used || {}).map(([k, v]) => h("span", { className: "tag", style: { marginRight: 4 }, key: k }, `${k}: ${v}`))
            ),
            h("td", { className: "num", key: "3" }, attempt.similar_records),
            h("td", { className: "num", key: "4" }, attempt.accepted_similar_records),
            h("td", { className: "num", key: "5" }, fmtPct(attempt.acceptance_rate)),
            h("td", { className: "num", key: "6" }, fmtLpa(attempt.p20_offered_ctc)),
            h("td", { className: "num", key: "7" }, fmtLpa(attempt.p50_offered_ctc)),
            h("td", { className: "num", key: "8" }, fmtLpa(attempt.p80_offered_ctc)),
          ])
        )
      ),
    ]),
  ]);
}

/* ------------------------------------------------------------
   View 3 — The Arena (negotiation twin)
------------------------------------------------------------ */

function ArenaView({ options, onDeal }) {
  useLucide();
  const [form, setField] = useCandidateForm(options, { offered_ctc: 14, target_probability: 0.75, max_rounds: 6 });
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [reveal, setReveal] = useState(0);
  const timerRef = useRef(null);

  const rounds = (result && result.rounds) || [];
  const done = reveal >= rounds.length;

  useEffect(() => {
    if (!result) return undefined;
    setReveal(0);
    timerRef.current = setInterval(() => {
      setReveal((r) => {
        if (r + 1 >= (result.rounds || []).length) clearInterval(timerRef.current);
        return r + 1;
      });
    }, 1100);
    return () => clearInterval(timerRef.current);
  }, [result]);

  function submit(event) {
    event.preventDefault();
    setError(null);
    setBusy(true);
    setResult(null);
    api
      .negotiate({ ...numericPayload(form), target_probability: Number(form.target_probability), max_rounds: Number(form.max_rounds) })
      .then((data) => {
        setResult(data);
        if (onDeal) onDeal(data);
      })
      .catch((err) => setError(err.detail || "Unable to run the negotiation"))
      .finally(() => setBusy(false));
  }

  function skip() {
    clearInterval(timerRef.current);
    setReveal(rounds.length);
  }

  const f = (kind, k, label, values, step) => h(FormField, { kind, form, setField, k, label, values, step, key: k });

  return h("div", null, [
    h(ViewHead, {
      key: "head",
      eyebrow: "03 · THE ARENA",
      title: "Watch two agents fight over the price.",
      blurb: "A Recruiter Agent guarding budget faces a Candidate Agent chasing value — both grounded in the same acceptance model. Rounds play out live.",
    }),
    h("div", { className: "workbench", key: "bench" }, [
      h("form", { className: "console", key: "console", onSubmit: submit }, [
        h("div", { className: "panel-head", key: "h" }, [h("h3", { key: "t" }, "Match Setup"), h("span", { key: "s" }, "all CTC in LPA")]),
        h("div", { className: "form-grid", key: "grid" }, [
          f("input", "current_ctc", "Current CTC"),
          f("input", "expected_ctc", "Expected CTC"),
          f("input", "offered_ctc", "Opening Offer"),
          f("input", "relevant_experience_years", "Experience (yrs)"),
          f("input", "notice_period_days", "Notice (days)", null, "1"),
          f("select", "offered_band", "Band", options.offered_band),
          f("select", "lob", "Line of Business", options.lob),
          f("select", "primary_skill", "Primary Skill", options.primary_skill),
          f("select", "location", "Location", options.location),
          f("select", "previous_company_type", "Company Type", options.previous_company_type),
          f("select", "candidate_source", "Source", options.candidate_source),
          f("select", "joining_bonus", "Joining Bonus", [0, 1]),
          f("select", "relocation", "Relocation", [0, 1]),
          f("select", "target_probability", "Target P(Accept)", [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]),
          f("select", "max_rounds", "Max Rounds", [3, 4, 5, 6, 7, 8]),
        ]),
        h("div", { className: "actions", key: "act" }, [
          h("button", { className: "btn btn-primary", type: "submit", disabled: busy, key: "b" }, [
            h(Icon, { name: "swords", size: 14, key: "i" }),
            busy ? "AGENTS NEGOTIATING…" : "OPEN THE ARENA",
          ]),
        ]),
        error && h(AlertLine, { tone: "bad", tag: "ERROR", key: "err" }, error),
      ]),
      h(
        "div",
        { key: "result" },
        result
          ? h(ArenaResult, { result, reveal, done, onSkip: skip })
          : h(
              "div",
              { className: "empty", style: { padding: "80px 20px" } },
              busy ? "AGENTS ARE AT THE TABLE ▌" : "SET UP THE MATCH AND OPEN THE ARENA ▌"
            )
      ),
    ]),
  ]);
}

function ArenaResult({ result, reveal, done, onSkip }) {
  const rounds = result.rounds || [];
  const shown = rounds.slice(0, reveal);
  const lastShown = shown[shown.length - 1];
  const liveProb = lastShown ? lastShown.acceptance_probability : null;
  const statusMap = {
    agreed: { label: "DEAL CLOSED", tone: "" },
    agreement_with_risk: { label: "DEAL AT RISK", tone: "warn" },
    impasse: { label: "IMPASSE", tone: "bad" },
    max_rounds_reached: { label: "NO DEAL IN LIMIT", tone: "warn" },
  };
  const status = statusMap[result.status] || { label: String(result.status).toUpperCase(), tone: "warn" };

  const offerSeries = rounds.map((r, i) => ({ x: r.round, y: r.recruiter_offer }));
  const askSeries = rounds.map((r) => ({ x: r.round, y: r.candidate_ask }));
  const probSeries = rounds.map((r) => ({ x: r.round, y: r.acceptance_probability }));

  return h("div", null, [
    h("div", { className: "arena-progress", key: "prog" }, [
      h("span", { key: "r" }, ["ROUND ", h("span", { className: "round-count", key: "c" }, `${Math.min(reveal, rounds.length)}/${rounds.length}`)]),
      liveProb !== null && h("span", { className: `tag ${probTone(liveProb)}`, key: "p" }, `P(ACCEPT) ${fmtPct(liveProb)}`),
      !done &&
        h("button", { className: "btn btn-ghost", style: { height: 26, padding: "0 10px", fontSize: 10, marginLeft: "auto" }, onClick: onSkip, key: "s" }, "SKIP ⏭"),
    ]),
    h(
      "div",
      { className: "transcript", key: "transcript" },
      shown.flatMap((r) => [
        h("div", { className: "turn recruiter", key: `${r.round}-r` }, [
          h("div", { className: "turn-meta", key: "m" }, [`◆ RECRUITER AGENT · ROUND ${r.round}`, h("span", { className: "tag", key: "o", style: { marginLeft: 6 } }, fmtLpaUnit(r.recruiter_offer))]),
          h("p", { key: "p" }, r.recruiter_message),
          r.recruiter_reason && h("div", { className: "turn-reason", key: "why" }, `why: ${r.recruiter_reason}`),
        ]),
        h("div", { className: "turn candidate", key: `${r.round}-c` }, [
          h("div", { className: "turn-meta", key: "m" }, [`◇ CANDIDATE AGENT`, h("span", { className: "tag", key: "o", style: { marginLeft: 6 } }, fmtLpaUnit(r.candidate_ask))]),
          h("p", { key: "p" }, r.candidate_message),
          r.candidate_reason && h("div", { className: "turn-reason", key: "why" }, `why: ${r.candidate_reason}`),
        ]),
      ])
    ),
    !done && h("div", { className: "chat-typing", key: "typing", style: { marginTop: 10 } }, "AGENTS EXCHANGING TERMS…"),

    done &&
      h("div", { key: "outcome", style: { marginTop: 16 } }, [
        h("div", { className: `ticket ${status.tone}`, key: "ticket" }, [
          h("div", { className: "ticket-head", key: "h" }, [
            h("div", { key: "l" }, [
              h("div", { className: "ticket-eyebrow", key: "e" }, "FINAL TERMS"),
              h("div", { className: "ticket-ctc", key: "v" }, [fmtLpa(result.final_offer), h("small", { key: "u" }, " LPA")]),
              h(FxLine, { key: "fx", lpa: result.final_offer }),
            ]),
            h("span", { className: `stamp ${status.tone}`, key: "stamp" }, status.label),
          ]),
          h("div", { className: "ticket-row", key: "cells" }, [
            h("div", { className: "ticket-cell", key: "1" }, [h("span", { key: "l" }, "P(Accept) Final"), h("strong", { key: "v", className: probTone(result.final_probability) === "good" ? "up" : "" }, fmtPct(result.final_probability))]),
            h("div", { className: "ticket-cell", key: "2" }, [h("span", { key: "l" }, "Budget Ceiling"), h("strong", { key: "v" }, fmtLpaUnit(result.budget_cap))]),
            h("div", { className: "ticket-cell", key: "3" }, [h("span", { key: "l" }, "Final Ask"), h("strong", { key: "v" }, fmtLpaUnit(result.final_candidate_ask))]),
            h("div", { className: "ticket-cell", key: "4" }, [h("span", { key: "l" }, "Model Target"), h("strong", { key: "v" }, fmtLpaUnit(result.minimum_offer_for_target))]),
            h("div", { className: "ticket-cell", key: "5" }, [h("span", { key: "l" }, "Levers Used"), h("strong", { key: "v" }, (result.active_levers || []).join(", ") || "none")]),
          ]),
        ]),
        h("p", { className: "explain", key: "summary" }, result.summary),
        result.next_action && h(AlertLine, { tone: "warn", tag: "NEXT", key: "next" }, result.next_action),

        h("section", { className: "section grid-2", key: "charts", style: { marginTop: 14 } }, [
          h("div", { className: "panel quiet", key: "conv" }, [
            h("div", { className: "panel-head", key: "h" }, [h("h3", { key: "t" }, "Convergence"), h("span", { key: "s" }, "offer vs ask by round · LPA")]),
            h(LineChart, {
              key: "c",
              series: [
                { name: "OFFER", color: CHART.green, data: offerSeries },
                { name: "ASK", color: CHART.amber, data: askSeries },
              ],
              xFormat: (x) => `R${x}`,
            }),
            h(Legend, { key: "l", items: [{ name: "Recruiter offer", color: CHART.green }, { name: "Candidate ask", color: CHART.amber }] }),
          ]),
          h("div", { className: "panel quiet", key: "prob" }, [
            h("div", { className: "panel-head", key: "h" }, [h("h3", { key: "t" }, "Win Probability"), h("span", { key: "s" }, "P(accept) by round")]),
            h(LineChart, {
              key: "c",
              series: [{ name: "P(accept)", color: CHART.blue, data: probSeries }],
              yAsPercent: true,
              xFormat: (x) => `R${x}`,
            }),
          ]),
        ]),

        h("div", { className: "table-wrap", key: "tbl" }, [
          h("table", null, [
            h(
              "thead",
              { key: "h" },
              h("tr", null, ["Round", "Offer", "Ask", "P(Accept)", "Commercial", "Target Met"].map((head) => h("th", { key: head }, head)))
            ),
            h(
              "tbody",
              { key: "b" },
              rounds.map((r) =>
                h("tr", { key: r.round }, [
                  h("td", { className: "num", key: "0" }, r.round),
                  h("td", { className: "num", key: "1" }, fmtLpaUnit(r.recruiter_offer)),
                  h("td", { className: "num", key: "2" }, fmtLpaUnit(r.candidate_ask)),
                  h("td", { className: "num", key: "3" }, fmtPct(r.acceptance_probability)),
                  h("td", { key: "4" }, h("span", { className: `tag ${r.commercial_agreement ? "good" : ""}` }, r.commercial_agreement ? "YES" : "NO")),
                  h("td", { key: "5" }, h("span", { className: `tag ${r.target_probability_met ? "good" : ""}` }, r.target_probability_met ? "YES" : "NO")),
                ])
              )
            ),
          ]),
        ]),
        (result.method_notes || []).length
          ? h("div", { className: "explain", key: "notes", style: { marginTop: 10 } }, (result.method_notes || []).map((note, i) => h("p", { key: i }, note)))
          : null,
      ]),
  ]);
}

/* ------------------------------------------------------------
   View 4 — Risk Radar
------------------------------------------------------------ */

function RadarScope({ flagged, selected, onSelect }) {
  const size = 320;
  const c = size / 2;
  const maxR = c - 18;
  const rings = [0.33, 0.66, 1];
  const golden = 137.508;

  const blips = (flagged || []).map((row, i) => {
    const angleDeg = (i * golden + 20) % 360;
    const a = (angleDeg * Math.PI) / 180;
    // low acceptance probability → close to the center (danger closing in)
    const rr = 24 + Math.max(0, Math.min(1, row.acceptance_probability)) * (maxR - 34);
    return { row, x: c + rr * Math.cos(a), y: c + rr * Math.sin(a) };
  });

  return h("div", { className: "radar-scope" }, [
    h("svg", { key: "svg", viewBox: `0 0 ${size} ${size}` }, [
      ...rings.map((f, i) =>
        h("circle", { key: `ring${i}`, cx: c, cy: c, r: maxR * f, fill: "none", stroke: "#1d2731", strokeWidth: 1 })
      ),
      h("line", { key: "h", x1: 18, y1: c, x2: size - 18, y2: c, stroke: "#16202a" }),
      h("line", { key: "v", x1: c, y1: 18, x2: c, y2: size - 18, stroke: "#16202a" }),
      h("g", { key: "sweep", className: "radar-sweep" }, [
        h("path", {
          key: "wedge",
          d: `M ${c} ${c} L ${c + maxR} ${c} A ${maxR} ${maxR} 0 0 0 ${c + maxR * Math.cos(-0.6)} ${c + maxR * Math.sin(-0.6)} Z`,
          fill: "rgba(61, 220, 151, 0.08)",
        }),
        h("line", { key: "beam", x1: c, y1: c, x2: c + maxR, y2: c, stroke: "rgba(61, 220, 151, 0.5)", strokeWidth: 1.5 }),
      ]),
      h("circle", { key: "hub", cx: c, cy: c, r: 3, fill: "#3ddc97" }),
      ...blips.map(({ row, x, y }) => {
        const hot = row.urgency === "High";
        const color = hot ? "#f0637e" : "#e8b45a";
        const active = selected === row.candidate_ref;
        return h(
          "g",
          { key: row.candidate_ref, className: "blip", onClick: () => onSelect(row.candidate_ref) },
          [
            h("circle", { key: "pulse", className: "pulse", cx: x, cy: y, r: 8, fill: "none", stroke: color, strokeWidth: 1 }),
            h("circle", { key: "dot", cx: x, cy: y, r: active ? 6 : 4.5, fill: color, stroke: active ? "#e6edf3" : "none", strokeWidth: 1.5 }),
            h("title", { key: "t" }, `${row.candidate_ref} · ${row.primary_skill} · P(accept) ${fmtPct(row.acceptance_probability)}`),
          ]
        );
      }),
    ]),
    h("div", { key: "cap", className: "radar-caption" }, "CENTER = LOWEST P(ACCEPT) · CLICK A BLIP TO INSPECT"),
  ]);
}

function RadarView() {
  useLucide();
  const [scan, setScan] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [auto, setAuto] = useState(false);
  const [expanded, setExpanded] = useState(null);

  const runScan = useCallback(() => {
    setBusy(true);
    setError(null);
    api
      .riskScan({ queue_size: 40, risk_threshold: 0.55, top_n: 10 })
      .then(setScan)
      .catch((err) => setError(err.detail || "Unable to run risk scan"))
      .finally(() => setBusy(false));
  }, []);

  useEffect(() => {
    runScan();
  }, [runScan]);

  useEffect(() => {
    if (!auto) return undefined;
    const id = setInterval(runScan, 20000);
    return () => clearInterval(id);
  }, [auto, runScan]);

  const flagged = (scan && scan.flagged_offers) || [];

  return h("div", null, [
    h(ViewHead, {
      key: "head",
      eyebrow: "04 · RISK RADAR",
      title: "The agent that never sleeps.",
      blurb: "An autonomous watcher re-scores the live offer queue, spots deals about to fall through, and drafts the escalation before anyone asks.",
    }),
    error && h(AlertLine, { tone: "bad", tag: "ERROR", key: "err" }, error),
    h("div", { className: "radar-layout", key: "layout" }, [
      h("div", { className: "panel", key: "scope" }, [
        h("div", { className: "panel-head", key: "h" }, [h("h3", { key: "t" }, "Scope"), h("span", { key: "s" }, busy ? "sweeping…" : "live")]),
        h(RadarScope, { key: "radar", flagged, selected: expanded, onSelect: (ref) => setExpanded(expanded === ref ? null : ref) }),
        h("div", { className: "actions", key: "act" }, [
          h("button", { className: "btn btn-primary", onClick: runScan, disabled: busy, key: "b" }, [
            h(Icon, { name: "radar", size: 14, key: "i" }),
            busy ? "SWEEPING…" : "SWEEP NOW",
          ]),
          h(
            "button",
            { className: `btn btn-ghost`, onClick: () => setAuto(!auto), key: "a", style: auto ? { borderColor: "#3ddc97", color: "#3ddc97" } : {} },
            auto ? "AUTO · ON" : "AUTO · OFF"
          ),
        ]),
        scan &&
          h("div", { className: "stat-row", key: "stats", style: { marginTop: 14 } }, [
            h("div", { className: "stat", key: "1" }, [h("span", { key: "l" }, "Scanned"), h("strong", { key: "v" }, String(scan.queue_scanned))]),
            h("div", { className: "stat", key: "2" }, [h("span", { key: "l" }, "Flagged"), h("strong", { key: "v", style: { color: "#f0637e" } }, String(scan.flagged_count))]),
            h("div", { className: "stat", key: "3" }, [h("span", { key: "l" }, "Threshold"), h("strong", { key: "v" }, fmtPct(scan.risk_threshold))]),
          ]),
      ]),
      h("div", { className: "panel quiet", key: "list" }, [
        h("div", { className: "panel-head", key: "h" }, [h("h3", { key: "t" }, "Flagged Deals"), h("span", { key: "s" }, "lowest P(accept) first")]),
        flagged.length
          ? h("div", { className: "table-wrap", key: "t" }, [
              h("table", null, [
                h(
                  "thead",
                  { key: "h" },
                  h("tr", null, ["", "Ref", "Skill", "Band", "Offered", "P(Accept)", "Fix Quote", "Outcome", ""].map((head, i) => h("th", { key: i }, head)))
                ),
                h(
                  "tbody",
                  { key: "b" },
                  flagged.flatMap((row) => {
                    const open = expanded === row.candidate_ref;
                    return [
                      h("tr", { key: row.candidate_ref }, [
                        h("td", { key: "u" }, h("span", { className: `tag ${row.urgency === "High" ? "bad" : "warn"}` }, row.urgency.toUpperCase())),
                        h("td", { className: "num", key: "r" }, row.candidate_ref),
                        h("td", { key: "s" }, row.primary_skill),
                        h("td", { key: "bd" }, h("span", { className: "tag" }, row.offered_band)),
                        h("td", { className: "num", key: "o" }, fmtLpaUnit(row.offered_ctc)),
                        h("td", { className: "num", key: "p", style: { color: row.acceptance_probability < 0.35 ? "#f0637e" : "#e8b45a" } }, fmtPct(row.acceptance_probability)),
                        h("td", { className: "num", key: "f" }, row.suggested_ctc ? fmtLpaUnit(row.suggested_ctc) : "REVIEW"),
                        h("td", { key: "a" }, h("span", { className: `tag ${row.actual_outcome === "Joined" || row.actual_outcome === "Accepted" ? "good" : "bad"}` }, row.actual_outcome)),
                        h(
                          "td",
                          { key: "x" },
                          h(
                            "button",
                            { className: "btn btn-ghost", style: { height: 24, padding: "0 10px", fontSize: 9 }, onClick: () => setExpanded(open ? null : row.candidate_ref) },
                            open ? "HIDE" : "ALERT"
                          )
                        ),
                      ]),
                      open &&
                        h("tr", { key: `${row.candidate_ref}-alert` }, [
                          h("td", { colSpan: 9, key: "c" }, h(AlertLine, { tone: "bad", tag: "DRAFTED ALERT" }, row.alert_message)),
                        ]),
                    ];
                  })
                ),
              ]),
            ])
          : h("div", { className: "empty", key: "e" }, busy ? "SWEEPING THE QUEUE ▌" : "NO AT-RISK DEALS IN THE CURRENT QUEUE"),
      ]),
    ]),
  ]);
}

/* ------------------------------------------------------------
   View 5 — The Ledger
------------------------------------------------------------ */

function LedgerTable({ candidates }) {
  return h("div", { className: "table-wrap" }, [
    h("table", null, [
      h(
        "thead",
        { key: "h" },
        h(
          "tr",
          null,
          ["Ref", "Date", "Skill", "LOB", "Location", "Band", "Current", "Expected", "Offered", "Hike", "Status"].map((head) => h("th", { key: head }, head))
        )
      ),
      h(
        "tbody",
        { key: "b" },
        candidates.map((row) =>
          h("tr", { key: row.candidate_ref }, [
            h("td", { className: "num", key: "1" }, row.candidate_ref),
            h("td", { className: "num", key: "2" }, row.offer_date),
            h("td", { key: "3" }, row.primary_skill),
            h("td", { key: "4" }, row.lob),
            h("td", { key: "5" }, row.location),
            h("td", { key: "6" }, h("span", { className: "tag" }, row.offered_band)),
            h("td", { className: "num", key: "7" }, fmtLpaUnit(row.current_ctc)),
            h("td", { className: "num", key: "8" }, fmtLpaUnit(row.expected_ctc)),
            h("td", { className: "num", key: "9" }, fmtLpaUnit(row.offered_ctc)),
            h("td", { className: "num", key: "10" }, `${Number(row.offered_hike_pct).toFixed(1)}%`),
            h("td", { key: "11" }, h("span", { className: `tag ${row.status === "Joined" || row.status === "Accepted" ? "good" : "bad"}` }, row.status)),
          ])
        )
      ),
    ]),
  ]);
}

function LedgerView({ candidates }) {
  useLucide();
  return h("div", null, [
    h(ViewHead, {
      key: "head",
      eyebrow: "05 · THE LEDGER",
      title: "Every trade on the book.",
      blurb: "The most recent offers, their pricing, and how each one settled.",
    }),
    h("div", { className: "panel", key: "p" }, [h(LedgerTable, { key: "t", candidates })]),
  ]);
}

/* ------------------------------------------------------------
   Rich answer cards inside chat bubbles
------------------------------------------------------------ */

function ChatCard({ card }) {
  if (!card) return null;

  const stat = (label, value, tone) =>
    h("div", { className: "chat-stat", key: label }, [
      h("span", { key: "l" }, label),
      h("strong", { key: "v", className: tone || "" }, value),
    ]);

  if (card.type === "kpis") {
    return h("div", { className: "chat-card" }, [
      h("div", { className: "chat-card-title", key: "t" }, "DESK KPIS"),
      h("div", { className: "chat-stats", key: "s" }, [
        stat("Offers", String(card.total_offers)),
        stat("Won", String(card.accepted_or_joined)),
        stat("Win Rate", fmtPct(card.acceptance_rate), card.acceptance_rate >= 0.5 ? "up" : "down"),
        stat("No-shows", String(card.no_show)),
      ]),
    ]);
  }

  if (card.type === "brief") {
    return h("div", { className: "chat-card" }, [
      h("div", { className: "chat-card-title", key: "t" }, "DESK BRIEF"),
      h("div", { className: "chat-stats", key: "s" }, [
        stat("Win Rate", fmtPct(card.acceptance_rate)),
        stat("Trend", String(card.trend_direction || "—").toUpperCase(), card.trend_direction === "improving" ? "up" : card.trend_direction === "declining" ? "down" : ""),
        stat("At Risk", String(card.at_risk), card.at_risk > 0 ? "down" : "up"),
      ]),
      (card.best_skill || card.worst_skill) &&
        h("div", { className: "chat-card-note", key: "n" }, [
          card.best_skill ? `▲ ${card.best_skill}` : null,
          card.best_skill && card.worst_skill ? "  ·  " : null,
          card.worst_skill ? `▼ ${card.worst_skill}` : null,
        ]),
    ]);
  }

  if (card.type === "quote") {
    const hasBand = card.p20 !== null && card.p20 !== undefined && card.p80 !== null && card.p80 !== undefined;
    return h("div", { className: "chat-card" }, [
      h("div", { className: "chat-card-title", key: "t" }, `QUOTE · ${card.label || ""}`),
      h("div", { className: "chat-stats", key: "s" }, [
        stat("Offer", fmtLpaUnit(card.offered_ctc)),
        stat("P(Accept)", fmtPct(card.probability), probTone(card.probability) === "good" ? "up" : probTone(card.probability) === "bad" ? "down" : ""),
        card.suggested_ctc !== null && card.suggested_ctc !== undefined ? stat("Suggest", fmtLpaUnit(card.suggested_ctc), "up") : null,
        card.probability_at_suggestion !== null && card.probability_at_suggestion !== undefined ? stat("P at Suggest", fmtPct(card.probability_at_suggestion), "up") : null,
      ]),
      hasBand &&
        h("div", { className: "chat-band", key: "b" }, [
          h(RangeBand, {
            p20: card.p20,
            p50: card.p50,
            p80: card.p80,
            offered: card.offered_ctc,
            suggested: card.suggested_ctc,
          }),
        ]),
    ]);
  }

  return null;
}

/* ------------------------------------------------------------
   Copilot drawer (chat + dataset upload)
------------------------------------------------------------ */

const CHAT_CHIPS = [
  { label: "☀ BRIEF ME", prompt: "Give me the daily desk brief." },
  { label: "⚡ PRICE ONE", prompt: "Price a 6-year Java Spring developer in Bangalore, current 12 LPA, expecting 17, offer on table 16." },
  { label: "◎ AT RISK?", prompt: "How many offers are currently at risk?" },
];

function Copilot({ open, onClose, onUiAction, voicePing, deskContext }) {
  useLucide();
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([
    { role: "model", content: "Desk copilot online. Ask about benchmarks, run simulations, or drop a CSV/Excel dataset to retrain the engine." },
  ]);
  const [busy, setBusy] = useState(false);
  const [listening, setListening] = useState(false);
  const [speakReplies, setSpeakReplies] = useState(false);
  const endRef = useRef(null);
  const fileRef = useRef(null);
  const recognitionRef = useRef(null);
  const speakNextRef = useRef(false);

  useEffect(() => {
    if (endRef.current) endRef.current.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy, open]);

  const sendMessage = useCallback(
    (content, viaVoice = false) => {
      const text = String(content || "").trim();
      if (!text) return;
      speakNextRef.current = viaVoice || speakReplies;
      setMessages((cur) => {
        const next = [...cur, { role: "user", content: text }];
        setBusy(true);
        api
          .chat(next.map(({ role, content }) => ({ role, content })), deskContext)
          .then((data) => {
            const reply = data.response || "No response generated.";
            setMessages((current) => [...current, { role: "model", content: reply, cards: data.cards || [] }]);
            (data.ui_actions || []).forEach((action) => onUiAction && onUiAction(action));
            if (speakNextRef.current) speakText(reply);
          })
          .catch((err) => {
            setMessages((current) => [...current, { role: "model", content: err.detail || "Copilot could not connect. Check backend/.env and server logs." }]);
          })
          .finally(() => setBusy(false));
        return next;
      });
    },
    [onUiAction, speakReplies, deskContext]
  );

  const startListening = useCallback(() => {
    if (!SpeechRec || listening) return;
    window.speechSynthesis && window.speechSynthesis.cancel();
    const recognition = new SpeechRec();
    recognitionRef.current = recognition;
    recognition.lang = "en-IN";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setListening(false);
      setSpeakReplies(true);
      sendMessage(transcript, true);
    };
    recognition.onerror = () => setListening(false);
    recognition.onend = () => setListening(false);
    setListening(true);
    recognition.start();
  }, [listening, sendMessage]);

  useEffect(() => {
    if (voicePing && open) startListening();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voicePing, open]);

  useEffect(() => () => {
    if (recognitionRef.current) recognitionRef.current.abort();
    if (window.speechSynthesis) window.speechSynthesis.cancel();
  }, []);

  function handleFile(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    event.target.value = "";
    setBusy(true);
    setMessages((cur) => [...cur, { role: "user", content: `Uploading dataset: ${file.name}` }]);
    const formData = new FormData();
    formData.append("file", file);
    fetch("/api/upload", { method: "POST", body: formData })
      .then((res) => (res.ok ? res.json() : res.json().then((err) => Promise.reject(err))))
      .then((data) => {
        setMessages((cur) => [
          ...cur,
          {
            role: "model",
            content: `Loaded ${data.record_count} records from ${file.name}. Engine retrained — ROC AUC ${data.metrics.roc_auc}. All desk views refreshed.`,
          },
        ]);
        if (onUiAction) onUiAction({ type: "DATA_UPDATED" });
      })
      .catch((err) => {
        setMessages((cur) => [...cur, { role: "model", content: `Upload failed: ${err.detail || err.message || "Unknown error"}` }]);
      })
      .finally(() => setBusy(false));
  }

  function submit(event) {
    event.preventDefault();
    if (busy) return;
    const content = input.trim();
    if (!content) return;
    setInput("");
    sendMessage(content, false);
  }

  if (!open) return null;

  return h("div", { className: "copilot" }, [
    h("div", { className: "copilot-head", key: "head" }, [
      h("h3", { key: "t" }, [h(Icon, { name: "bot", size: 15, key: "i" }), listening ? "Listening…" : "Desk Copilot"]),
      h("div", { key: "btns", style: { display: "flex", gap: 6 } }, [
        h(
          "button",
          {
            className: "copilot-close",
            key: "spk",
            title: speakReplies ? "Spoken replies ON" : "Spoken replies OFF",
            "aria-label": "Toggle spoken replies",
            style: speakReplies ? { color: "#3ddc97", borderColor: "#3ddc97" } : {},
            onClick: () => {
              if (speakReplies && window.speechSynthesis) window.speechSynthesis.cancel();
              setSpeakReplies(!speakReplies);
            },
          },
          h(Icon, { name: speakReplies ? "volume-2" : "volume-x", size: 14 })
        ),
        h("button", { className: "copilot-close", onClick: onClose, key: "x", "aria-label": "Close copilot" }, h(Icon, { name: "x", size: 14 })),
      ]),
    ]),
    h("div", { className: "chat-messages", key: "msgs" }, [
      ...messages.map((message, index) =>
        h(
          "div",
          { key: `${index}-${message.role}`, className: `chat-message ${message.role === "user" ? "user" : "model"}` },
          [
            ...String(message.content || "")
              .split("\n")
              .filter((line) => line.trim().length > 0)
              .map((line, li) => h("p", { key: li }, line)),
            ...(message.cards || []).map((card, ci) => h(ChatCard, { key: `card${ci}`, card })),
          ]
        )
      ),
      busy && h("div", { className: "chat-typing", key: "typing" }, "COPILOT THINKING ▌"),
      h("div", { ref: endRef, key: "end" }),
    ]),
    h("div", { className: "chat-chips", key: "chips" },
      CHAT_CHIPS.map((chip) =>
        h(
          "button",
          { key: chip.label, className: "chat-chip", disabled: busy, onClick: () => sendMessage(chip.prompt, false) },
          chip.label
        )
      )
    ),
    h("form", { className: "chat-input-form", key: "input", onSubmit: submit }, [
      h("input", { key: "file", type: "file", ref: fileRef, onChange: handleFile, accept: ".xlsx,.xls,.csv", style: { display: "none" } }),
      h(
        "button",
        { key: "up", type: "button", disabled: busy, onClick: () => fileRef.current && fileRef.current.click(), "aria-label": "Upload dataset" },
        h(Icon, { name: "paperclip", size: 15 })
      ),
      SpeechRec &&
        h(
          "button",
          {
            key: "mic",
            type: "button",
            className: listening ? "mic-live" : "",
            disabled: busy,
            onClick: startListening,
            "aria-label": "Voice input",
            title: "Talk to the desk",
          },
          h(Icon, { name: "mic", size: 15 })
        ),
      h("input", {
        key: "text",
        type: "text",
        value: input,
        onChange: (event) => setInput(event.target.value),
        placeholder: listening ? "Listening… speak now" : "Ask the desk…",
        disabled: busy,
      }),
      h("button", { key: "send", type: "submit", disabled: busy || !input.trim(), "aria-label": "Send" }, h(Icon, { name: "send", size: 15 })),
    ]),
  ]);
}

/* ------------------------------------------------------------
   Command palette
------------------------------------------------------------ */

function CommandPalette({ open, onClose, onNavigate, onToggleCopilot, onVoice }) {
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef(null);

  const commands = useMemo(
    () => [
      ...NAV.map((n) => ({ id: n.key, label: `${n.label} — ${n.eyebrow}`, icon: n.icon, kind: "GO TO", run: () => onNavigate(n.key) })),
      { id: "copilot", label: "Toggle Desk Copilot", icon: "bot", kind: "AGENT", run: onToggleCopilot },
      ...(SpeechRec && onVoice ? [{ id: "voice", label: "Talk to the Desk (voice)", icon: "mic", kind: "VOICE", run: onVoice }] : []),
    ],
    [onNavigate, onToggleCopilot, onVoice]
  );

  const filtered = commands.filter((c) => c.label.toLowerCase().includes(query.toLowerCase()));

  useEffect(() => {
    if (open) {
      setQuery("");
      setActive(0);
      setTimeout(() => inputRef.current && inputRef.current.focus(), 20);
    }
  }, [open]);

  useEffect(() => {
    if (window.lucide) window.lucide.createIcons();
  });

  if (!open) return null;

  function onKeyDown(event) {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActive((a) => Math.min(a + 1, filtered.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive((a) => Math.max(a - 1, 0));
    } else if (event.key === "Enter") {
      event.preventDefault();
      const cmd = filtered[active];
      if (cmd) {
        cmd.run();
        onClose();
      }
    } else if (event.key === "Escape") {
      onClose();
    }
  }

  return h("div", { className: "cmdk-overlay", onClick: onClose }, [
    h("div", { className: "cmdk", key: "box", onClick: (e) => e.stopPropagation() }, [
      h("input", {
        key: "input",
        ref: inputRef,
        className: "cmdk-input",
        placeholder: "TYPE A COMMAND…",
        value: query,
        onChange: (e) => {
          setQuery(e.target.value);
          setActive(0);
        },
        onKeyDown,
      }),
      h(
        "div",
        { className: "cmdk-list", key: "list" },
        filtered.length
          ? filtered.map((cmd, i) =>
              h(
                "button",
                {
                  key: cmd.id,
                  className: `cmdk-item ${i === active ? "active" : ""}`,
                  onMouseEnter: () => setActive(i),
                  onClick: () => {
                    cmd.run();
                    onClose();
                  },
                },
                [h(Icon, { name: cmd.icon, size: 15, key: "i" }), cmd.label, h("span", { className: "cmdk-kind", key: "k" }, cmd.kind)]
              )
            )
          : h("div", { className: "empty", key: "none", style: { border: "none" } }, "NO MATCHING COMMAND")
      ),
      h("div", { className: "cmdk-hint", key: "hint" }, [
        h("span", { key: "1" }, "↑↓ NAVIGATE"),
        h("span", { key: "2" }, "↵ RUN"),
        h("span", { key: "3" }, "ESC CLOSE"),
      ]),
    ]),
  ]);
}

/* ------------------------------------------------------------
   App shell
------------------------------------------------------------ */

function initialTab() {
  const fromHash = window.location.hash.replace("#", "");
  return NAV.some((n) => n.key === fromHash) ? fromHash : "dashboard";
}

function App() {
  const [summary, setSummary] = useState(null);
  const [options, setOptions] = useState(null);
  const [candidates, setCandidates] = useState([]);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTabState] = useState(initialTab);

  const setActiveTab = useCallback((key) => {
    if (NAV.some((n) => n.key === key)) window.location.hash = key;
    setActiveTabState(key);
  }, []);
  const [copilotOpen, setCopilotOpen] = useState(false);
  const [cmdkOpen, setCmdkOpen] = useState(false);
  const [fx, setFx] = useState(null);
  const [voicePing, setVoicePing] = useState(0);
  const [studioPrefill, setStudioPrefill] = useState(null);
  const [lastQuote, setLastQuote] = useState(null);
  const [lastDeal, setLastDeal] = useState(null);

  const handleQuote = useCallback((result) => {
    setLastQuote({
      candidate: result.candidate,
      recommendation_status: result.recommendation_status,
      acceptance_probability: result.acceptance_probability,
      suggested_ctc: result.suggested_ctc,
      probability_at_suggested_ctc: result.probability_at_suggested_ctc,
      benchmark_p50: (result.percentile_recommendation || {}).p50_offered_ctc,
      warnings: (result.warnings || []).slice(0, 3),
    });
  }, []);

  const handleDeal = useCallback((result) => {
    setLastDeal({
      status: result.status,
      final_offer: result.final_offer,
      final_probability: result.final_probability,
      rounds: (result.rounds || []).length,
      summary: result.summary,
    });
  }, []);

  useEffect(() => {
    api
      .marketWire({})
      .then((data) => setFx(data.rates || null))
      .catch(() => {});
  }, []);

  const triggerVoice = useCallback(() => {
    setCopilotOpen(true);
    setVoicePing((v) => v + 1);
  }, []);

  const loadAll = useCallback(() => {
    Promise.all([api.summary(), api.options(), api.candidates()])
      .then(([summaryData, optionsData, candidateData]) => {
        setSummary(summaryData);
        setOptions(optionsData);
        setCandidates(candidateData);
      })
      .catch((err) => setError(String(err)));
  }, []);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  useEffect(() => {
    if (window.lucide) window.lucide.createIcons();
  });

  useEffect(() => {
    function onKey(event) {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setCmdkOpen((v) => !v);
        return;
      }
      const target = event.target;
      const typing = target && (target.tagName === "INPUT" || target.tagName === "SELECT" || target.tagName === "TEXTAREA");
      if (typing) return;
      const nav = NAV.find((n) => n.num === event.key);
      if (nav) setActiveTab(nav.key);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  function handleUiAction(action) {
    if (action.type === "FILTER_UI" && action.tab) {
      setActiveTab(action.tab);
    } else if (action.type === "PREFILL_SIMULATOR") {
      setStudioPrefill({ fields: action.fields || {}, run: action.run !== false, nonce: Date.now() });
      setActiveTab("simulator");
    } else if (action.type === "DATA_UPDATED") {
      loadAll();
    }
  }

  if (error) {
    return h("div", { className: "boot" }, [
      h("div", { className: "boot-logo", key: "l" }, ["PACT", h("span", { className: "cursor", key: "c" }, "▌")]),
      h("div", { key: "e", style: { color: "#f0637e" } }, `TERMINAL FAULT · ${error}`),
    ]);
  }

  if (!summary || !options) {
    return h("div", { className: "boot" }, [
      h("div", { className: "boot-logo", key: "l" }, ["PACT", h("span", { className: "cursor", key: "c" }, "▌")]),
      h("div", { key: "m" }, "BOOTING THE OFFER TERMINAL…"),
    ]);
  }

  const deskContext = {
    view: activeTab,
    last_quote: lastQuote,
    last_deal: lastDeal,
  };

  const view =
    activeTab === "simulator"
      ? h(StudioView, { options, prefill: studioPrefill, onQuote: handleQuote })
      : activeTab === "negotiation"
      ? h(ArenaView, { options, onDeal: handleDeal })
      : activeTab === "risk"
      ? h(RadarView, null)
      : activeTab === "table"
      ? h(LedgerView, { candidates })
      : h(PulseView, { summary, candidates, onNavigate: setActiveTab });

  return h("div", { className: "app" }, [
    h(TickerTape, { key: "ticker", candidates, fx }),
    h(Topbar, { key: "topbar", summary, onOpenCmdk: () => setCmdkOpen(true), onVoice: triggerVoice }),
    h("div", { className: "body", key: "body" }, [
      h(Rail, { key: "rail", activeTab, onNavigate: setActiveTab }),
      h("main", { className: "stage", key: "stage" }, [h("div", { className: "stage-inner", key: activeTab }, view)]),
    ]),
    !copilotOpen &&
      h("button", { key: "fab", className: "copilot-fab", onClick: () => setCopilotOpen(true) }, [
        h(Icon, { name: "bot", size: 16, key: "i" }),
        "COPILOT",
      ]),
    h(Copilot, { key: "copilot", open: copilotOpen, onClose: () => setCopilotOpen(false), onUiAction: handleUiAction, voicePing, deskContext }),
    h(CommandPalette, {
      key: "cmdk",
      open: cmdkOpen,
      onClose: () => setCmdkOpen(false),
      onNavigate: setActiveTab,
      onToggleCopilot: () => setCopilotOpen((v) => !v),
      onVoice: triggerVoice,
    }),
  ]);
}

ReactDOM.createRoot(document.getElementById("root")).render(h(App));
