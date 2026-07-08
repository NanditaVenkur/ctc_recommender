const { createElement: h, useEffect, useMemo, useState, useRef } = React;

const api = {
  summary: () => fetch("/api/summary").then((res) => res.json()),
  options: () => fetch("/api/options").then((res) => res.json()),
  candidates: () => fetch("/api/candidates?limit=24").then((res) => res.json()),
  chat: (messages) =>
    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages }),
    }).then((res) => {
      if (!res.ok) return res.json().then((err) => Promise.reject(err));
      return res.json();
    }),
  recommend: (payload) =>
    fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then((res) => {
      if (!res.ok) return res.json().then((err) => Promise.reject(err));
      return res.json();
    }),
  benchmarkRecords: (filters) =>
    fetch("/api/benchmark-records", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filters }),
    }).then((res) => {
      if (!res.ok) return res.json().then((err) => Promise.reject(err));
      return res.json();
    }),
};

function fmtPct(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${Math.round(value * 100)}%`;
}

function fmtLpa(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${Number(value).toFixed(2)} LPA`;
}

function Icon({ name, size = 18 }) {
  return h("i", { "data-lucide": name, style: { width: size, height: size } });
}

function Metric({ label, value, note, icon, trend }) {
  return h("div", { className: "metric" }, [
    h("div", { className: "metric-top" }, [
      h("div", { className: "metric-label", key: "label" }, label),
      icon && h("div", { className: "metric-icon" }, h(Icon, { name: icon, size: 20 }))
    ]),
    h("div", { className: "metric-value", key: "value" }, value),
    h("div", { className: "metric-bottom", key: "note" }, [
      trend && h("span", { className: `trend ${trend > 0 ? 'up' : 'down'}` }, [
        h(Icon, { name: trend > 0 ? "trending-up" : "trending-down", size: 14 }),
        `${Math.abs(trend)}%`
      ]),
      h("span", { className: "metric-note" }, note)
    ]),
  ]);
}

function BarRow({ label, value, max, tone }) {
  const width = max ? Math.max(3, (value / max) * 100) : 0;
  return h("div", { className: "bar-row" }, [
    h("span", { key: "label" }, label),
    h("div", { className: "bar-track", key: "track" }, [
      h("div", { className: `bar-fill ${tone || ""}`, style: { width: `${width}%` } }),
    ]),
    h("strong", { key: "value" }, value),
  ]);
}

function LineChart({ data, xKey, yKey, yAsPercent = false }) {
  const width = 680;
  const height = 220;
  const pad = { top: 16, right: 18, bottom: 28, left: 42 };
  const values = data || [];
  if (!values.length) return h("div", { className: "empty" }, "No chart data yet");

  const xs = values.map((d) => d[xKey]);
  const ys = values.map((d) => d[yKey]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys, 0);
  const maxY = Math.max(...ys, yAsPercent ? 1 : 0);
  const xScale = (x) => pad.left + ((x - minX) / Math.max(maxX - minX, 1)) * (width - pad.left - pad.right);
  const yScale = (y) => height - pad.bottom - ((y - minY) / Math.max(maxY - minY, 0.01)) * (height - pad.top - pad.bottom);
  const path = values.map((d, i) => `${i === 0 ? "M" : "L"} ${xScale(d[xKey])} ${yScale(d[yKey])}`).join(" ");
  const areaPath = `${path} L ${xScale(values[values.length - 1][xKey])} ${yScale(minY)} L ${xScale(values[0][xKey])} ${yScale(minY)} Z`;
  const last = values[values.length - 1];

  return h("svg", { className: "chart", viewBox: `0 0 ${width} ${height}`, role: "img" }, [
    h("defs", null, [
      h("linearGradient", { id: "areaGrad", x1: "0%", y1: "0%", x2: "0%", y2: "100%" }, [
        h("stop", { offset: "0%", stopColor: "var(--accent)", stopOpacity: 0.3 }),
        h("stop", { offset: "100%", stopColor: "var(--accent)", stopOpacity: 0.0 })
      ])
    ]),
    h("line", { x1: pad.left, y1: height - pad.bottom, x2: width - pad.right, y2: height - pad.bottom, stroke: "#2e2e36" }),
    h("line", { x1: pad.left, y1: pad.top, x2: pad.left, y2: height - pad.bottom, stroke: "#2e2e36" }),
    h("path", { d: areaPath, className: "area-path", fill: "url(#areaGrad)" }),
    h("path", { d: path, className: "line-path" }),
    h("circle", { cx: xScale(last[xKey]), cy: yScale(last[yKey]), r: 5, fill: "var(--accent)", stroke: "#222228", strokeWidth: 2.5 }),
    h("text", { x: pad.left, y: height - 8 }, String(xs[0])),
    h("text", { x: width - pad.right - 60, y: height - 8 }, String(xs[xs.length - 1])),
    h("text", { x: 8, y: pad.top + 4 }, yAsPercent ? fmtPct(maxY) : maxY.toFixed(1)),
    h("text", { x: 8, y: height - pad.bottom }, yAsPercent ? fmtPct(minY) : minY.toFixed(1)),
  ]);
}

function TrendChart({ data }) {
  const chartData = (data || []).map((row, index) => ({
    index,
    label: row.period,
    acceptance_rate: row.acceptance_rate,
  }));
  return h(LineChart, { data: chartData, xKey: "index", yKey: "acceptance_rate", yAsPercent: true });
}

function ProbabilityChart({ data }) {
  return h(LineChart, { data, xKey: "offered_ctc", yKey: "acceptance_probability", yAsPercent: true });
}

function DonutChart({ rate }) {
  const pct = Math.round(rate * 100);
  const r = 54, cx = 64, cy = 64, stroke = 10;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - rate);
  return h("div", { className: "donut-wrap" }, [
    h("svg", { width: 128, height: 128, viewBox: "0 0 128 128" }, [
      h("circle", { cx, cy, r, fill: "none", stroke: "rgba(255,255,255,0.06)", strokeWidth: stroke }),
      h("circle", { cx, cy, r, fill: "none", stroke: "var(--accent)", strokeWidth: stroke,
        strokeDasharray: circ, strokeDashoffset: offset,
        strokeLinecap: "round", transform: "rotate(-90 64 64)",
        style: { transition: "stroke-dashoffset 1s ease" }
      }),
    ]),
    h("div", { className: "donut-label" }, [
      h("strong", null, `${pct}%`),
      h("span", null, "Accepted")
    ])
  ]);
}

function Dashboard({ summary, onNavigate }) {
  const kpis = summary.kpis;
  const maxStatus = Math.max(...Object.values(summary.status_counts));
  const maxBand = Math.max(...summary.by_band.map((row) => row.offers));
  const maxSource = Math.max(...summary.by_source.map((row) => row.offers));
  const maxSkill = Math.max(...(summary.by_skill || []).map((r) => r.offers), 1);

  return h("div", null, [
    // ─── HEADER ───
    h("div", { className: "dashboard-header" }, [
      h("h2", null, "Dashboard"),
      h("p", null, "Recruitment analytics and AI-powered insights at a glance.")
    ]),

    // ─── AI INSIGHT BANNER ───
    summary.insight && h("div", { className: "insight-banner", key: "insight" }, [
      h(Icon, { name: "sparkles", size: 18 }),
      h("span", null, summary.insight)
    ]),

    // ─── QUICK ACTIONS ───
    h("div", { className: "quick-actions", key: "actions" }, [
      h("button", { className: "action-card", onClick: () => onNavigate && onNavigate("simulator") }, [
        h("div", { className: "action-icon sim" }, h(Icon, { name: "calculator", size: 22 })),
        h("div", null, [h("strong", null, "Simulate Offer"), h("span", null, "Predict acceptance probability")])
      ]),
      h("button", { className: "action-card", onClick: () => onNavigate && onNavigate("table") }, [
        h("div", { className: "action-icon tbl" }, h(Icon, { name: "table", size: 22 })),
        h("div", null, [h("strong", null, "Browse Offers"), h("span", null, "View recent offer history")])
      ]),
    ]),

    // ─── KPI CARDS + DONUT ───
    h("section", { className: "section kpi-section", key: "kpis" }, [
      h("div", { className: "kpi-grid" }, [
        h("div", { className: "kpi-cards" }, [
          h(Metric, { label: "Total offers", value: kpis.total_offers, note: "Historical records", icon: "briefcase", trend: 12 }),
          h(Metric, { label: "Accepted / joined", value: kpis.accepted_or_joined, note: `${fmtPct(kpis.acceptance_rate)} acceptance`, icon: "user-check", trend: 5 }),
          h(Metric, { label: "Median offered CTC", value: fmtLpa(kpis.median_offered_ctc), note: "Across all offers", icon: "banknote" }),
          h(Metric, { label: "Avg offered hike", value: `${kpis.avg_offered_hike_pct}%`, note: "Over current CTC", icon: "trending-up", trend: -2 }),
        ]),
        h("div", { className: "panel donut-panel" }, [
          h("div", { className: "panel-title" }, [h("h3", null, "Acceptance Rate"), h("span", null, "Overall")]),
          h(DonutChart, { rate: kpis.acceptance_rate }),
          h("div", { className: "donut-stats" }, [
            h("div", null, [h("span", null, "Accepted"), h("strong", null, kpis.accepted_or_joined)]),
            h("div", null, [h("span", null, "Declined"), h("strong", null, kpis.declined_or_no_show)]),
            h("div", null, [h("span", null, "No Show"), h("strong", null, kpis.no_show)]),
          ])
        ])
      ])
    ]),

    // ─── CHARTS ROW ───
    h("section", { className: "section grid-2", key: "charts" }, [
      h("div", { className: "panel" }, [
        h("div", { className: "panel-title" }, [h("h3", null, "Offer outcome funnel"), h("span", null, "Count by status")]),
        h("div", { className: "funnel" }, Object.entries(summary.status_counts).map(([label, value]) =>
          h(BarRow, { key: label, label, value, max: maxStatus, tone: label === "Joined" || label === "Accepted" ? "good" : "warn" })
        )),
      ]),
      h("div", { className: "panel" }, [
        h("div", { className: "panel-title" }, [h("h3", null, "Acceptance trend"), h("span", null, "Monthly rate")]),
        h(TrendChart, { data: summary.trend }),
      ]),
    ]),

    // ─── SKILLS LEADERBOARD + LOCATION ───
    h("section", { className: "section grid-2", key: "skills" }, [
      h("div", { className: "panel" }, [
        h("div", { className: "panel-title" }, [h("h3", null, "Top Skills"), h("span", null, "By acceptance rate")]),
        h("div", { className: "leaderboard" }, (summary.by_skill || []).slice(0, 8).map((row, i) =>
          h("div", { key: row.skill, className: "lb-row" }, [
            h("span", { className: `lb-rank ${i < 3 ? "top" : ""}` }, `#${i + 1}`),
            h("span", { className: "lb-name" }, row.skill),
            h("span", { className: `tag ${row.acceptance_rate >= 0.5 ? "good" : "bad"}` }, fmtPct(row.acceptance_rate)),
            h("span", { className: "lb-meta" }, `${row.offers} offers`),
          ])
        ))
      ]),
      h("div", { className: "panel" }, [
        h("div", { className: "panel-title" }, [h("h3", null, "Locations"), h("span", null, "Volume & acceptance")]),
        h("div", { className: "leaderboard" }, (summary.by_location || []).slice(0, 8).map((row) =>
          h("div", { key: row.location, className: "lb-row" }, [
            h(Icon, { name: "map-pin", size: 14 }),
            h("span", { className: "lb-name" }, row.location),
            h("span", { className: `tag ${row.acceptance_rate >= 0.5 ? "good" : "bad"}` }, fmtPct(row.acceptance_rate)),
            h("span", { className: "lb-meta" }, `${row.offers} offers · ${fmtLpa(row.median_ctc)} median`),
          ])
        ))
      ]),
    ]),

    // ─── BENCHMARKS ROW ───
    h("section", { className: "section grid-2", key: "benchmarks" }, [
      h("div", { className: "panel" }, [
        h("div", { className: "panel-title" }, [h("h3", null, "Accepted CTC range"), h("span", null, "P20 / P50 / P80")]),
        h("div", { className: "result-band" }, [
          h("div", { className: "result-stat" }, [h("span", null, "P20"), h("strong", null, fmtLpa(summary.accepted_ctc_percentiles.p20))]),
          h("div", { className: "result-stat" }, [h("span", null, "P50"), h("strong", null, fmtLpa(summary.accepted_ctc_percentiles.p50))]),
          h("div", { className: "result-stat" }, [h("span", null, "P80"), h("strong", null, fmtLpa(summary.accepted_ctc_percentiles.p80))]),
        ]),
      ]),
      h("div", { className: "panel" }, [
        h("div", { className: "panel-title" }, [h("h3", null, "Offers by band"), h("span", null, "Volume and acceptance")]),
        h("div", { className: "funnel" }, summary.by_band.map((row) =>
          h(BarRow, { key: row.band, label: `${row.band} (${fmtPct(row.acceptance_rate)})`, value: row.offers, max: maxBand, tone: "good" })
        )),
      ]),
    ]),

    // ─── MODEL + SOURCES ───
    h("section", { className: "section grid-2", key: "model" }, [
      h("div", { className: "panel" }, [
        h("div", { className: "panel-title" }, [h("h3", null, "Model health"), h("span", null, "ML prediction quality")]),
        h("div", { className: "result-band" }, [
          h("div", { className: "result-stat" }, [h("span", null, "ROC AUC"), h("strong", null, summary.model_metrics.roc_auc)]),
          h("div", { className: "result-stat" }, [h("span", null, "Brier score"), h("strong", null, summary.model_metrics.brier_score)]),
          h("div", { className: "result-stat" }, [h("span", null, "Test records"), h("strong", null, summary.model_metrics.test_records)]),
        ]),
      ]),
      h("div", { className: "panel" }, [
        h("div", { className: "panel-title" }, [h("h3", null, "Candidate sources"), h("span", null, "Volume and acceptance")]),
        h("div", { className: "funnel" }, summary.by_source.map((row) =>
          h(BarRow, { key: row.source, label: `${row.source} (${fmtPct(row.acceptance_rate)})`, value: row.offers, max: maxSource, tone: "good" })
        )),
      ]),
    ]),
  ]);
}

function RecentOffersTable({ candidates }) {
  return h("section", { className: "section", key: "table" }, [
    h("div", { className: "panel table-wrap" }, [
      h("table", null, [
        h("thead", null, h("tr", null, ["Ref", "Date", "Skill", "LOB", "Location", "Band", "Current", "Expected", "Offered", "Hike", "Status"].map((head) => h("th", { key: head }, head)))),
        h("tbody", null, candidates.map((row) =>
          h("tr", { key: row.candidate_ref }, [
            h("td", null, row.candidate_ref),
            h("td", null, row.offer_date),
            h("td", null, row.primary_skill),
            h("td", null, row.lob),
            h("td", null, row.location),
            h("td", null, h("span", { className: "tag" }, row.offered_band)),
            h("td", null, fmtLpa(row.current_ctc)),
            h("td", null, fmtLpa(row.expected_ctc)),
            h("td", null, fmtLpa(row.offered_ctc)),
            h("td", null, `${Number(row.offered_hike_pct).toFixed(1)}%`),
            h("td", null, h("span", { className: `tag ${row.status === "Joined" || row.status === "Accepted" ? "good" : "bad"}` }, row.status)),
          ])
        )),
      ]),
    ]),
  ]);
}

function OfferSimulator({ options }) {
  const defaults = useMemo(() => ({
    current_ctc: 15,
    expected_ctc: 22,
    offered_ctc: 20,
    relevant_experience_years: 6,
    notice_period_days: 60,
    offered_band: options.offered_band?.[0] || "E2",
    candidate_source: options.candidate_source?.[0] || "Direct",
    lob: options.lob?.[0] || "Digital",
    primary_skill: options.primary_skill?.[0] || "Java Spring",
    previous_company_type: options.previous_company_type?.[0] || "Service",
    location: options.location?.[0] || "Bangalore",
    joining_bonus: 0,
    relocation: 0,
    flexibility: "balanced",
  }), [options]);

  const [form, setForm] = useState(defaults);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => setForm(defaults), [defaults]);

  function setField(key, value) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  function submit(event) {
    event.preventDefault();
    setError(null);
    api.recommend({
      ...form,
      current_ctc: Number(form.current_ctc),
      expected_ctc: Number(form.expected_ctc),
      offered_ctc: Number(form.offered_ctc),
      relevant_experience_years: Number(form.relevant_experience_years),
      notice_period_days: Number(form.notice_period_days),
      joining_bonus: Number(form.joining_bonus),
      relocation: Number(form.relocation),
    })
      .then(setResult)
      .catch((err) => setError(err.detail || "Unable to generate recommendation"));
  }

  const select = (key, label, values) =>
    h("div", { className: "field" }, [
      h("label", null, label),
      h("select", { value: form[key], onChange: (e) => setField(key, e.target.value) },
        (values || []).map((value) => h("option", { key: value, value }, value))
      ),
    ]);

  const input = (key, label, step = "0.1") =>
    h("div", { className: "field" }, [
      h("label", null, label),
      h("input", { type: "number", step, value: form[key], onChange: (e) => setField(key, e.target.value) }),
    ]);

  return h("div", { className: "simulator-container" }, [
    h("section", { className: "section" }, [
      h("form", { className: "form-panel", onSubmit: submit }, [
        h("div", { className: "form-grid" }, [
          input("current_ctc", "Current CTC"),
          input("expected_ctc", "Expected CTC"),
          input("offered_ctc", "Offered CTC"),
          input("relevant_experience_years", "Experience"),
          input("notice_period_days", "Notice period", "1"),
          select("offered_band", "Band", options.offered_band),
          select("lob", "LOB", options.lob),
          select("primary_skill", "Primary skill", options.primary_skill),
          select("location", "Location", options.location),
          select("previous_company_type", "Company type", options.previous_company_type),
          select("candidate_source", "Source", options.candidate_source),
          select("flexibility", "Benchmark flexibility", ["strict", "balanced", "broad"]),
          select("joining_bonus", "Joining bonus", [0, 1]),
          select("relocation", "Relocation", [0, 1]),
        ]),
        h("div", { className: "actions" }, [
          h("button", { className: "primary-button", type: "submit" }, [h(Icon, { name: "calculator" }), "Calculate"]),
        ]),
        error && h("div", { className: "warning" }, error),
      ]),
    ]),
    h("section", { className: "section" }, result ? h(ResultPanel, { result }) : h("div", { className: "empty" }, "Run a candidate through the simulator to see the suggested CTC, acceptance probability, and benchmark range.")),
  ]);
}

function ResultPanel({ result }) {
  const p = result.percentile_recommendation;
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [showModal, setShowModal] = useState(false);

  function openBenchmark() {
    if (benchmarkData) { setShowModal(true); return; }
    api.benchmarkRecords(p.filters_used)
      .then((data) => { setBenchmarkData(data); setShowModal(true); })
      .catch(() => {});
  }

  return h("div", { className: "panel" }, [
    h("div", { className: "panel-title" }, [
      h("h3", null, "Recommendation"),
      h("span", null, `${p.specificity} / ${p.confidence} confidence`),
    ]),
    h("div", { className: "result-band" }, [
      h("div", { className: "result-stat" }, [h("span", null, "Suggested CTC"), h("strong", null, fmtLpa(result.suggested_ctc))]),
      h("div", { className: "result-stat" }, [h("span", null, "Current offer probability"), h("strong", null, fmtPct(result.acceptance_probability))]),
      h("div", { className: "result-stat clickable", onClick: openBenchmark }, [
        h("span", null, ["Benchmark records ", h(Icon, { name: "external-link", size: 12 })]),
        h("strong", null, p.accepted_similar_records)
      ]),
    ]),
    h("div", { className: "result-band" }, [
      h("div", { className: "result-stat" }, [h("span", null, "P20 accepted CTC"), h("strong", null, fmtLpa(p.p20_offered_ctc))]),
      h("div", { className: "result-stat" }, [h("span", null, "P50 accepted CTC"), h("strong", null, fmtLpa(p.p50_offered_ctc))]),
      h("div", { className: "result-stat" }, [h("span", null, "P80 accepted CTC"), h("strong", null, fmtLpa(p.p80_offered_ctc))]),
    ]),
    result.warnings.map((warning) => h("div", { className: "warning", key: warning }, warning)),
    h("div", { className: "panel-title", style: { marginTop: 16 } }, [h("h3", null, "Acceptance probability curve"), h("span", null, "Offer CTC vs probability")]),
    h(ProbabilityChart, { data: result.acceptance_curve }),
    h("div", { className: "panel-title", style: { marginTop: 12 } }, [h("h3", null, "Benchmark filters used"), h("span", null, `${p.similar_records} similar offers`) ]),
    h("div", null, Object.entries(p.filters_used).map(([key, value]) => h("span", { className: "tag", style: { marginRight: 6, marginBottom: 6 }, key }, `${key}: ${value}`))),
    showModal && h(BenchmarkModal, { data: benchmarkData, filters: p.filters_used, onClose: () => setShowModal(false) }),
  ]);
}

function BenchmarkModal({ data, filters, onClose }) {
  const rows = data || [];
  return h("div", { className: "modal-overlay", onClick: (e) => { if (e.target === e.currentTarget) onClose(); } }, [
    h("div", { className: "modal-content" }, [
      h("div", { className: "modal-header" }, [
        h("div", null, [
          h("h2", null, `Benchmark Records (${rows.length})`),
          h("p", null, Object.entries(filters).map(([k, v]) => `${k}: ${v}`).join(" · "))
        ]),
        h("button", { className: "modal-close", onClick: onClose }, h(Icon, { name: "x", size: 20 })),
      ]),
      h("div", { className: "modal-body" }, [
        rows.length === 0
          ? h("div", { className: "empty" }, "No records found.")
          : h("div", { className: "table-wrap" }, [
              h("table", null, [
                h("thead", null, h("tr", null, ["Ref", "Date", "Skill", "LOB", "Location", "Band", "Current", "Expected", "Offered", "Hike", "Status"].map((hd) => h("th", { key: hd }, hd)))),
                h("tbody", null, rows.map((row, i) =>
                  h("tr", { key: row.candidate_ref || i }, [
                    h("td", null, row.candidate_ref),
                    h("td", null, row.offer_date),
                    h("td", null, row.primary_skill),
                    h("td", null, row.lob),
                    h("td", null, row.location),
                    h("td", null, h("span", { className: "tag" }, row.offered_band)),
                    h("td", null, fmtLpa(row.current_ctc)),
                    h("td", null, fmtLpa(row.expected_ctc)),
                    h("td", null, fmtLpa(row.offered_ctc)),
                    h("td", null, row.offered_hike_pct != null ? `${Number(row.offered_hike_pct).toFixed(1)}%` : "-"),
                    h("td", null, h("span", { className: `tag ${row.status === "Joined" || row.status === "Accepted" ? "good" : "bad"}` }, row.status)),
                  ])
                )),
              ])
            ])
      ])
    ])
  ]);
}

function ChatPane({ onUiAction }) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([
    { role: "model", content: "Hi! I'm your AI Recruiter Assistant. I have access to our entire historical dataset and the ML offer simulator. What would you like to know?" }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    if (endRef.current) {
      endRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  function submit(e) {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    const newMessages = [...messages, { role: "user", content: input.trim() }];
    setMessages(newMessages);
    setInput("");
    setIsLoading(true);

    api.chat(newMessages)
      .then((data) => {
        setMessages((prev) => [...prev, { role: "model", content: data.response }]);
        if (data.ui_actions && data.ui_actions.length > 0 && onUiAction) {
          data.ui_actions.forEach(action => onUiAction(action));
        }
      })
      .catch((err) => {
        setMessages((prev) => [...prev, { role: "model", content: "Error: " + (err.detail || err.message || "Request failed") }]);
      })
      .finally(() => {
        setIsLoading(false);
      });
  }

  return h("div", { className: "chat-pane" }, [
    h("div", { className: "chat-header" }, [
      h("h3", null, [h(Icon, { name: "bot", size: 18 }), " AI Assistant"]),
    ]),
    h("div", { className: "chat-messages" }, [
      messages.map((msg, i) =>
        h("div", { key: i, className: `chat-message ${msg.role === "user" ? "user" : "model"}` }, [
          h("span", { dangerouslySetInnerHTML: { __html: msg.content.replace(/\n/g, '<br/>') } })
        ])
      ),
      isLoading && h("div", { className: "chat-typing" }, "Assistant is typing..."),
      h("div", { ref: endRef })
    ]),
    h("form", { className: "chat-input-area chat-input-form", onSubmit: submit }, [
      h("input", {
        value: input,
        onChange: (e) => setInput(e.target.value),
        placeholder: "Ask about CTC or trends...",
        disabled: isLoading
      }),
      h("button", { type: "submit", disabled: isLoading }, h(Icon, { name: "send", size: 16 })),
    ])
  ]);
}

function RightCanvas({ summary, options, candidates, activeTab, setActiveTab }) {
  const renderTab = () => {
    if (activeTab === "dashboard") return h(Dashboard, { summary, onNavigate: setActiveTab });
    if (activeTab === "simulator") return h(OfferSimulator, { options });
    if (activeTab === "table") return h(RecentOffersTable, { candidates });
    return null;
  };

  return h("div", { className: "canvas-pane" }, [
    h("div", { className: "tabs-header" }, [
      h("button", { className: `tab-btn ${activeTab === 'dashboard' ? 'active' : ''}`, onClick: () => setActiveTab('dashboard') }, [h(Icon, { name: "layout-dashboard", size: 16 }), "Dashboard"]),
      h("button", { className: `tab-btn ${activeTab === 'simulator' ? 'active' : ''}`, onClick: () => setActiveTab('simulator') }, [h(Icon, { name: "calculator", size: 16 }), "Simulator"]),
      h("button", { className: `tab-btn ${activeTab === 'table' ? 'active' : ''}`, onClick: () => setActiveTab('table') }, [h(Icon, { name: "table", size: 16 }), "Recent Offers"]),
    ]),
    h("div", { className: "canvas-content" }, renderTab())
  ]);
}

function App() {
  const [summary, setSummary] = useState(null);
  const [options, setOptions] = useState(null);
  const [candidates, setCandidates] = useState([]);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("dashboard");

  useEffect(() => {
    Promise.all([api.summary(), api.options(), api.candidates()])
      .then(([summaryData, optionsData, candidateData]) => {
        setSummary(summaryData);
        setOptions(optionsData);
        setCandidates(candidateData);
      })
      .catch((err) => setError(String(err)));
  }, []);

  useEffect(() => {
    if (window.lucide) window.lucide.createIcons();
  });

  if (error) return h("div", { className: "empty" }, error);
  if (!summary || !options) return h("div", { className: "empty" }, "Loading HR offer intelligence...");

  function handleUiAction(action) {
    if (action.type === "FILTER_UI") {
      if (action.tab) {
        setActiveTab(action.tab);
      }
    }
  }

  return h("div", { className: "app" }, [
    h("header", { className: "topbar" }, [
      h("div", { className: "brand" }, [
        h("div", { className: "brand-mark" }, h(Icon, { name: "badge-indian-rupee" })),
        h("div", null, [h("h1", null, "CTC Offer Intelligence"), h("p", null, "AI-Driven Offer Recommendations")]),
      ]),
      h("div", { className: "status-pill" }, [h(Icon, { name: "database", size: 16 }), `${summary.kpis.total_offers} offers loaded`]),
    ]),
    h("main", { className: "main-layout" }, [
      h(ChatPane, { onUiAction: handleUiAction }),
      h(RightCanvas, { summary, options, candidates, activeTab, setActiveTab }),
    ])
  ]);
}

ReactDOM.createRoot(document.getElementById("root")).render(h(App));
