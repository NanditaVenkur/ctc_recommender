const { createElement: h, useEffect, useMemo, useState } = React;

const api = {
  summary: () => fetch("/api/summary").then((res) => res.json()),
  options: () => fetch("/api/options").then((res) => res.json()),
  candidates: () => fetch("/api/candidates?limit=24").then((res) => res.json()),
  recommend: (payload) =>
    fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
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

function Metric({ label, value, note }) {
  return h("div", { className: "metric" }, [
    h("div", { className: "metric-label", key: "label" }, label),
    h("div", { className: "metric-value", key: "value" }, value),
    h("div", { className: "metric-note", key: "note" }, note),
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
  const last = values[values.length - 1];

  return h("svg", { className: "chart", viewBox: `0 0 ${width} ${height}`, role: "img" }, [
    h("line", { x1: pad.left, y1: height - pad.bottom, x2: width - pad.right, y2: height - pad.bottom, stroke: "#dbe1e7" }),
    h("line", { x1: pad.left, y1: pad.top, x2: pad.left, y2: height - pad.bottom, stroke: "#dbe1e7" }),
    h("path", { d: path, className: "line-path" }),
    h("circle", { cx: xScale(last[xKey]), cy: yScale(last[yKey]), r: 4, fill: "#0f766e" }),
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

function Dashboard({ summary, candidates }) {
  const kpis = summary.kpis;
  const maxStatus = Math.max(...Object.values(summary.status_counts));
  const maxBand = Math.max(...summary.by_band.map((row) => row.offers));
  const maxSource = Math.max(...summary.by_source.map((row) => row.offers));

  return h("div", null, [
    h("section", { className: "section", key: "kpis" }, [
      h("div", { className: "grid-4" }, [
        h(Metric, { label: "Total offers", value: kpis.total_offers, note: "Historical offer records" }),
        h(Metric, { label: "Accepted / joined", value: kpis.accepted_or_joined, note: `${fmtPct(kpis.acceptance_rate)} acceptance rate` }),
        h(Metric, { label: "Median offered CTC", value: fmtLpa(kpis.median_offered_ctc), note: "Across all offers" }),
        h(Metric, { label: "Avg offered hike", value: `${kpis.avg_offered_hike_pct}%`, note: "Over current CTC" }),
      ]),
    ]),
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
    h("section", { className: "section grid-2", key: "quality" }, [
      h("div", { className: "panel" }, [
        h("div", { className: "panel-title" }, [h("h3", null, "Model health"), h("span", null, "Logistic regression baseline")]),
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
    h("section", { className: "section", key: "table" }, [
      h("div", { className: "section-header" }, [
        h("div", null, [h("h2", null, "Recent Offers"), h("p", null, "A quick view of offer amounts, hikes, and outcomes.")]),
      ]),
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

  return h("aside", null, [
    h("section", { className: "section" }, [
      h("div", { className: "section-header" }, [
        h("div", null, [h("h2", null, "Offer Simulator"), h("p", null, "Estimate acceptance probability and CTC range for a candidate.")]),
      ]),
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
  const benchmarkRecords = result.accepted_benchmark_records || [];
  const isOk = result.recommendation_status === "ok";
  const panelTitle = result.recommendation_status === "review_low_support" ? "Recommendation Review" : isOk ? "Recommendation" : "Escalation Review";
  return h("div", { className: "panel" }, [
    h("div", { className: "panel-title" }, [
      h("h3", null, panelTitle),
      h("span", null, `${p.specificity} / ${p.confidence} match confidence`),
    ]),
    h("div", { className: isOk ? "notice good" : "notice warn" }, result.recommendation_message),
    h("div", { className: "result-band" }, [
      h("div", { className: "result-stat" }, [h("span", null, "Suggested CTC"), h("strong", null, fmtLpa(result.suggested_ctc))]),
      h("div", { className: "result-stat" }, [h("span", null, "Current offer probability"), h("strong", null, fmtPct(result.acceptance_probability))]),
      h("div", { className: "result-stat" }, [h("span", null, "Probability at suggested"), h("strong", null, fmtPct(result.probability_at_suggested_ctc))]),
    ]),
    h("div", { className: "result-band" }, [
      h("div", { className: "result-stat" }, [h("span", null, "P20 accepted CTC"), h("strong", null, fmtLpa(p.p20_offered_ctc))]),
      h("div", { className: "result-stat" }, [h("span", null, "P50 accepted CTC"), h("strong", null, fmtLpa(p.p50_offered_ctc))]),
      h("div", { className: "result-stat" }, [h("span", null, "P80 accepted CTC"), h("strong", null, fmtLpa(p.p80_offered_ctc))]),
    ]),
    h("div", { className: "result-band" }, [
      h("div", { className: "result-stat" }, [h("span", null, "Accepted benchmark records"), h("strong", null, p.accepted_similar_records)]),
      h("div", { className: "result-stat" }, [h("span", null, "Skill + LOB records"), h("strong", null, result.profile_match.skill_lob_records)]),
      h("div", { className: "result-stat" }, [h("span", null, "Target 70% offer"), h("strong", null, fmtLpa(result.target_offer_ctc))]),
    ]),
    h("div", { className: "result-band" }, [
      h("div", { className: "result-stat" }, [h("span", null, "Max searched offer"), h("strong", null, fmtLpa(result.curve_max_offer_ctc))]),
      h("div", { className: "result-stat" }, [h("span", null, "Probability at max"), h("strong", null, fmtPct(result.probability_at_curve_max))]),
      h("div", { className: "result-stat" }, [h("span", null, "Primary skill support"), h("strong", null, result.category_support?.primary_skill ?? "-")]),
    ]),
    result.warnings.map((warning) => h("div", { className: "warning", key: warning }, warning)),
    h("div", { className: "panel-title", style: { marginTop: 16 } }, [h("h3", null, "Acceptance probability curve"), h("span", null, "Offer CTC vs probability")]),
    h(ProbabilityChart, { data: result.acceptance_curve }),
    h("div", { className: "panel-title", style: { marginTop: 12 } }, [h("h3", null, "Benchmark filters used"), h("span", null, `${p.similar_records} similar offers`) ]),
    h("div", null, Object.entries(p.filters_used).map(([key, value]) => h("span", { className: "tag", style: { marginRight: 6, marginBottom: 6 }, key }, `${key}: ${value}`))),
    h("div", { className: "panel-title", style: { marginTop: 16 } }, [
      h("h3", null, "Accepted benchmark records"),
      h("span", null, `${benchmarkRecords.length} records shown`),
    ]),
    benchmarkRecords.length
      ? h("div", { className: "table-wrap benchmark-table" }, [
          h("table", null, [
            h("thead", null, h("tr", null, ["Ref", "Date", "Skill", "LOB", "Location", "Band", "Current", "Expected", "Offered", "Hike", "Gap", "Source", "Status"].map((head) => h("th", { key: head }, head)))),
            h("tbody", null, benchmarkRecords.map((row) =>
              h("tr", { key: `${row.candidate_ref}-${row.offer_date}` }, [
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
                h("td", null, `${Number(row.offer_gap_pct).toFixed(1)}%`),
                h("td", null, row.candidate_source),
                h("td", null, h("span", { className: "tag good" }, row.status)),
              ])
            )),
          ]),
        ])
      : h("div", { className: "empty" }, "No accepted benchmark records found for the selected filters."),
  ]);
}

function App() {
  const [summary, setSummary] = useState(null);
  const [options, setOptions] = useState(null);
  const [candidates, setCandidates] = useState([]);
  const [error, setError] = useState(null);

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

  return h("div", { className: "app" }, [
    h("header", { className: "topbar" }, [
      h("div", { className: "brand" }, [
        h("div", { className: "brand-mark" }, h(Icon, { name: "badge-indian-rupee" })),
        h("div", null, [h("h1", null, "CTC Offer Intelligence"), h("p", null, "Historical offer analytics and acceptance probability support")]),
      ]),
      h("div", { className: "status-pill" }, [h(Icon, { name: "database", size: 16 }), `${summary.kpis.total_offers} offers loaded`]),
    ]),
    h("main", { className: "content" }, [
      h("div", null, h(Dashboard, { summary, candidates })),
      h(OfferSimulator, { options }),
    ]),
  ]);
}

ReactDOM.createRoot(document.getElementById("root")).render(h(App));
