const { createElement: h, useEffect, useMemo, useRef, useState } = React;

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
  negotiate: (payload) =>
    fetch("/api/negotiate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then((res) => {
      if (!res.ok) return res.json().then((err) => Promise.reject(err));
      return res.json();
    }),
  riskScan: (params) =>
    fetch(`/api/risk-scan?${new URLSearchParams(params)}`).then((res) => {
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

function LineChart({ data, xKey, yKey, yAsPercent = false, selectedIndex = null }) {
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
  const selected =
    selectedIndex === null
      ? last
      : values[Math.max(0, Math.min(values.length - 1, Number(selectedIndex)))];

  return h("svg", { className: "chart", viewBox: `0 0 ${width} ${height}`, role: "img" }, [
    h("line", { x1: pad.left, y1: height - pad.bottom, x2: width - pad.right, y2: height - pad.bottom, stroke: "#dbe1e7" }),
    h("line", { x1: pad.left, y1: pad.top, x2: pad.left, y2: height - pad.bottom, stroke: "#dbe1e7" }),
    h("path", { d: path, className: "line-path" }),
    selected && h("line", {
      x1: xScale(selected[xKey]),
      y1: pad.top,
      x2: xScale(selected[xKey]),
      y2: height - pad.bottom,
      className: "selected-line",
    }),
    selected && h("circle", {
      cx: xScale(selected[xKey]),
      cy: yScale(selected[yKey]),
      r: 6,
      className: "selected-point",
    }),
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

function ProbabilityChart({ data, currentOffer, suggestedOffer }) {
  const values = data || [];
  const targetOffer = suggestedOffer ?? currentOffer;
  const initialIndex = values.length
    ? values.reduce((best, row, index) => (
        Math.abs(row.offered_ctc - targetOffer) < Math.abs(values[best].offered_ctc - targetOffer) ? index : best
      ), 0)
    : 0;
  const [index, setIndex] = useState(initialIndex);

  useEffect(() => {
    if (!values.length) return;
    setIndex(initialIndex);
  }, [initialIndex, values.length]);

  if (!values.length) return h(LineChart, { data: values, xKey: "offered_ctc", yKey: "acceptance_probability", yAsPercent: true });

  const selected = values[Math.max(0, Math.min(values.length - 1, Number(index)))];
  return h("div", { className: "probability-tool" }, [
    h(LineChart, {
      data: values,
      xKey: "offered_ctc",
      yKey: "acceptance_probability",
      yAsPercent: true,
      selectedIndex: index,
    }),
    h("div", { className: "slider-panel" }, [
      h("input", {
        type: "range",
        min: 0,
        max: values.length - 1,
        value: index,
        onChange: (event) => setIndex(Number(event.target.value)),
        "aria-label": "Inspect offer CTC on probability curve",
      }),
      h("div", { className: "slider-readout" }, [
        h("div", null, [h("span", null, "Inspected offer"), h("strong", null, fmtLpa(selected.offered_ctc))]),
        h("div", null, [h("span", null, "Predicted acceptance"), h("strong", null, fmtPct(selected.acceptance_probability))]),
      ]),
    ]),
  ]);
}

function Dashboard({ summary, candidates, onNavigate }) {
  const kpis = summary.kpis;
  const maxStatus = Math.max(...Object.values(summary.status_counts));
  const maxBand = Math.max(...summary.by_band.map((row) => row.offers));
  const maxSource = Math.max(...summary.by_source.map((row) => row.offers));

  return h("div", null, [
    h("div", { className: "dashboard-header" }, [
      h("h2", null, "Offer Intelligence"),
      h("p", null, "Track offer outcomes, benchmark CTC ranges, and model health."),
    ]),
    summary.insight && h("div", { className: "insight-banner" }, [
      h(Icon, { name: "sparkles", size: 16 }),
      h("span", null, summary.insight),
    ]),
    h("div", { className: "quick-actions" }, [
      h("button", { className: "action-card", onClick: () => onNavigate && onNavigate("simulator") }, [
        h("div", { className: "action-icon sim" }, h(Icon, { name: "calculator", size: 22 })),
        h("div", null, [h("strong", null, "Run offer simulator"), h("span", null, "Check a candidate CTC and acceptance probability")]),
      ]),
      h("button", { className: "action-card", onClick: () => onNavigate && onNavigate("table") }, [
        h("div", { className: "action-icon tbl" }, h(Icon, { name: "table", size: 22 })),
        h("div", null, [h("strong", null, "Review recent offers"), h("span", null, "Scan latest outcomes and CTCs")]),
      ]),
    ]),
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
        h("div", { className: "panel-title" }, [h("h3", null, "Accepted / joined CTC range"), h("span", null, "P20 / P50 / P80")]),
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

function RecentOffersTable({ candidates }) {
  return h("section", { className: "section", key: "table" }, [
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
  const filters = Object.entries(p.filters_used || {});
  const filterSummary = filters.length
    ? filters.map(([key, value]) => `${key}: ${value}`).join(", ")
    : "no benchmark filters";
  const acceptedCount = p.accepted_similar_records ?? benchmarkRecords.length;
  const warnings = result.warnings || [];
  return h("div", { className: "panel" }, [
    h("div", { className: "panel-title" }, [
      h("h3", null, panelTitle),
      h("span", null, `${p.specificity} / ${p.confidence} match confidence`),
    ]),
    h("div", { className: isOk ? "notice good" : "notice warn" }, result.recommendation_message),
    h("div", { className: "decision-grid" }, [
      h("div", { className: "decision-card current" }, [
        h("span", null, "Current offer being evaluated"),
        h("strong", null, fmtLpa(result.candidate.offered_ctc)),
        h("div", { className: "paired-metric" }, [
          h("span", null, "Acceptance probability at this offer"),
          h("b", null, fmtPct(result.acceptance_probability)),
        ]),
      ]),
      h("div", { className: "decision-card suggested" }, [
        h("span", null, "Suggested CTC"),
        h("strong", null, fmtLpa(result.suggested_ctc)),
        h("div", { className: "paired-metric" }, [
          h("span", null, "Acceptance probability at suggested CTC"),
          h("b", null, fmtPct(result.probability_at_suggested_ctc)),
        ]),
      ]),
    ]),
    h("div", { className: "panel-title", style: { marginTop: 16 } }, [
      h("h3", null, "Successful Offer Benchmark Range"),
      h("span", null, `${acceptedCount} accepted/joined records used for P20 / P50 / P80`),
    ]),
    h("p", { className: "explain" }, `Calculated from historical offers with status Accepted or Joined matching ${filterSummary}. Similarity rule: ${p.similarity_rule || "rule-based match"}.`),
    h("div", { className: "result-band" }, [
      h("div", { className: "result-stat" }, [h("span", null, "P20 successful CTC"), h("strong", null, fmtLpa(p.p20_offered_ctc))]),
      h("div", { className: "result-stat" }, [h("span", null, "P50 successful CTC"), h("strong", null, fmtLpa(p.p50_offered_ctc))]),
      h("div", { className: "result-stat" }, [h("span", null, "P80 successful CTC"), h("strong", null, fmtLpa(p.p80_offered_ctc))]),
    ]),
    h("div", { className: "panel-title", style: { marginTop: 16 } }, [
      h("h3", null, "Successful Profiles Behind This Range"),
      h("span", null, `${benchmarkRecords.length} records shown`),
    ]),
    benchmarkRecords.length
      ? h("div", { className: "table-wrap benchmark-table" }, [
          h("table", null, [
            h("thead", null, h("tr", null, ["Ref", "Date", "Skill", "LOB", "Location", "Band", "Exp", "Current", "Expected", "Offered", "Hike", "Gap", "Source", "Status"].map((head) => h("th", { key: head }, head)))),
            h("tbody", null, benchmarkRecords.map((row) =>
              h("tr", { key: `${row.candidate_ref}-${row.offer_date}`, className: row.status === "Joined" ? "row-joined" : "row-accepted" }, [
                h("td", null, row.candidate_ref),
                h("td", null, row.offer_date),
                h("td", null, row.primary_skill),
                h("td", null, row.lob),
                h("td", null, row.location),
                h("td", null, h("span", { className: "tag" }, row.offered_band)),
                h("td", null, `${Number(row.relevant_experience_years).toFixed(1)} yrs`),
                h("td", null, fmtLpa(row.current_ctc)),
                h("td", null, fmtLpa(row.expected_ctc)),
                h("td", null, fmtLpa(row.offered_ctc)),
                h("td", null, `${Number(row.offered_hike_pct).toFixed(1)}%`),
                h("td", null, `${Number(row.offer_gap_pct).toFixed(1)}%`),
                h("td", null, row.candidate_source),
                h("td", null, h("span", { className: row.status === "Joined" ? "tag good" : "tag info" }, row.status)),
              ])
            )),
          ]),
        ])
      : h("div", { className: "empty" }, "No accepted/joined benchmark records found for the selected filters."),
    warnings.map((warning) => h("div", { className: "warning", key: warning }, warning)),
    h("div", { className: "panel-title", style: { marginTop: 16 } }, [
      h("h3", null, "Acceptance Probability Curve"),
      h("span", null, "Move the slider to inspect offer values"),
    ]),
    h(ProbabilityChart, {
      data: result.acceptance_curve,
      currentOffer: result.candidate.offered_ctc,
      suggestedOffer: result.suggested_ctc,
    }),
    h("div", { className: "panel-title", style: { marginTop: 16 } }, [
      h("h3", null, "Benchmark Transparency"),
      h("span", null, "Why this recommendation has this confidence"),
    ]),
    h("div", { className: "result-band compact-band" }, [
      h("div", { className: "result-stat" }, [h("span", null, "Similar offers before acceptance filter"), h("strong", null, p.similar_records)]),
      h("div", { className: "result-stat" }, [h("span", null, "Accepted/joined offers used for range"), h("strong", null, acceptedCount)]),
      h("div", { className: "result-stat" }, [h("span", null, "Model 70% threshold"), h("strong", null, fmtLpa(result.target_offer_ctc)), h("em", null, "Reference only; not a separate recommendation")]),
    ]),
    h("div", { className: "result-band compact-band" }, [
      h("div", { className: "result-stat" }, [h("span", null, "Skill + LOB coverage"), h("strong", null, result.profile_match.skill_lob_records), h("em", null, "All historical records with this skill and business unit")]),
      h("div", { className: "result-stat" }, [h("span", null, "Exact profile records"), h("strong", null, result.profile_match.exact_profile_records), h("em", null, "Skill + LOB + location + band")]),
      h("div", { className: "result-stat" }, [h("span", null, `Exact profile + ${result.profile_match.experience_band}`), h("strong", null, result.profile_match.experience_band_records), h("em", null, "Same profile near this experience level")]),
    ]),
    h("div", { className: "result-band compact-band" }, [
      h("div", { className: "result-stat" }, [h("span", null, "Max searched offer"), h("strong", null, fmtLpa(result.curve_max_offer_ctc))]),
      h("div", { className: "result-stat" }, [h("span", null, "Probability at max searched offer"), h("strong", null, fmtPct(result.probability_at_curve_max))]),
      h("div", { className: "result-stat" }, [h("span", null, "Primary skill support"), h("strong", null, result.category_support?.primary_skill ?? "-")]),
    ]),
    h("div", { className: "panel-title", style: { marginTop: 12 } }, [h("h3", null, "Benchmark filters used"), h("span", null, `${p.similar_records} similar offers`) ]),
    h("div", { className: "filter-tags" }, filters.map(([key, value]) => h("span", { className: "tag", key }, `${key}: ${value}`))),
    h("div", { className: "panel-title", style: { marginTop: 16 } }, [
      h("h3", null, "Fallback attempts"),
      h("span", null, "Tried from most similar to broader"),
    ]),
    h(FallbackAttemptsTable, { attempts: p.fallback_attempts || [] }),
  ]);
  }

  function FallbackAttemptsTable({ attempts }) {
    if (!attempts.length) {
      return h("div", { className: "empty" }, "No fallback attempts available.");
    }

    return h("div", { className: "table-wrap fallback-table" }, [
      h("table", null, [
        h("thead", null, h("tr", null, ["#", "Similarity rule", "Filters", "Similar", "Accepted/joined", "Success rate", "P20", "P50", "P80"].map((head) => h("th", { key: head }, head)))),
        h("tbody", null, attempts.map((attempt, index) =>
          h("tr", { key: `${index}-${JSON.stringify(attempt.filters_used)}` }, [
            h("td", null, index + 1),
            h("td", null, attempt.similarity_rule || "-"),
            h("td", null, Object.entries(attempt.filters_used || {}).map(([key, value]) =>
              h("span", { className: "tag", style: { marginRight: 4, marginBottom: 4 }, key }, `${key}: ${value}`)
            )),
            h("td", null, attempt.similar_records),
            h("td", null, attempt.accepted_similar_records),
            h("td", null, fmtPct(attempt.acceptance_rate)),
            h("td", null, fmtLpa(attempt.p20_offered_ctc)),
            h("td", null, fmtLpa(attempt.p50_offered_ctc)),
            h("td", null, fmtLpa(attempt.p80_offered_ctc)),
          ])
        )),
      ]),
    ]);
  }

function NegotiationSimulator({ options }) {
  const defaults = useMemo(() => ({
    current_ctc: 15,
    expected_ctc: 22,
    offered_ctc: 17,
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
    target_probability: 0.75,
    max_rounds: 6,
  }), [options]);

  const [form, setForm] = useState(defaults);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => setForm(defaults), [defaults]);

  function setField(key, value) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  function submit(event) {
    event.preventDefault();
    setError(null);
    setIsRunning(true);
    api.negotiate({
      ...form,
      current_ctc: Number(form.current_ctc),
      expected_ctc: Number(form.expected_ctc),
      offered_ctc: Number(form.offered_ctc),
      relevant_experience_years: Number(form.relevant_experience_years),
      notice_period_days: Number(form.notice_period_days),
      joining_bonus: Number(form.joining_bonus),
      relocation: Number(form.relocation),
      target_probability: Number(form.target_probability),
      max_rounds: Number(form.max_rounds),
    })
      .then(setResult)
      .catch((err) => setError(err.detail || "Unable to run negotiation"))
      .finally(() => setIsRunning(false));
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
        h("div", null, [
          h("h2", null, "Negotiation Twin"),
          h("p", null, "A Recruiter Agent and a Candidate Agent negotiate CTC round by round, grounded by the same acceptance model."),
        ]),
      ]),
      h("form", { className: "form-panel", onSubmit: submit }, [
        h("div", { className: "form-grid" }, [
          input("current_ctc", "Current CTC"),
          input("expected_ctc", "Expected CTC"),
          input("offered_ctc", "Opening offer"),
          input("relevant_experience_years", "Experience"),
          input("notice_period_days", "Notice period", "1"),
          select("offered_band", "Band", options.offered_band),
          select("lob", "LOB", options.lob),
          select("primary_skill", "Primary skill", options.primary_skill),
          select("location", "Location", options.location),
          select("previous_company_type", "Company type", options.previous_company_type),
          select("candidate_source", "Source", options.candidate_source),
          select("joining_bonus", "Joining bonus", [0, 1]),
          select("relocation", "Relocation", [0, 1]),
          select("target_probability", "Target acceptance", [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]),
          select("max_rounds", "Max rounds", [3, 4, 5, 6, 7, 8]),
        ]),
        h("div", { className: "actions" }, [
          h("button", { className: "primary-button", type: "submit", disabled: isRunning }, [
            h(Icon, { name: "handshake" }),
            isRunning ? "Negotiating..." : "Run negotiation",
          ]),
        ]),
        error && h("div", { className: "warning" }, error),
      ]),
    ]),
    h("section", { className: "section" }, result ? h(NegotiationResult, { result }) : h("div", { className: "empty" }, "Run the negotiation to see the Recruiter Agent and Candidate Agent converge on a CTC.")),
  ]);
}

function NegotiationResult({ result }) {
  const statusTone = result.status === "agreed" ? "good" : result.status === "impasse" ? "bad" : "warn";
  const statusLabel = { agreed: "Agreement reached", impasse: "Impasse", max_rounds_reached: "No agreement within round limit" }[result.status] || result.status;
  const chartData = (result.rounds || []).map((r) => ({ round: r.round, acceptance_probability: r.acceptance_probability }));

  return h("div", { className: "panel" }, [
    h("div", { className: "panel-title" }, [
      h("h3", null, "Negotiation Outcome"),
      h("span", null, `${result.rounds.length} round(s)`),
    ]),
    h("div", { className: `notice ${statusTone === "good" ? "good" : "warn"}` }, `${statusLabel}. ${result.summary}`),
    h("div", { className: "decision-grid" }, [
      h("div", { className: "decision-card current" }, [
        h("span", null, "Final offer"),
        h("strong", null, fmtLpa(result.final_offer)),
        h("div", { className: "paired-metric" }, [
          h("span", null, "Predicted acceptance"),
          h("b", null, fmtPct(result.final_probability)),
        ]),
      ]),
      h("div", { className: "decision-card suggested" }, [
        h("span", null, "Authorized budget ceiling"),
        h("strong", null, fmtLpa(result.budget_cap)),
        h("div", { className: "paired-metric" }, [
          h("span", null, "Target acceptance"),
          h("b", null, fmtPct(result.target_probability)),
        ]),
      ]),
    ]),
    h("div", { className: "panel-title", style: { marginTop: 16 } }, [
      h("h3", null, "Negotiation Transcript"),
      h("span", null, "Recruiter Agent vs. Candidate Agent"),
    ]),
    h("div", { className: "negotiation-transcript" }, (result.rounds || []).flatMap((r) => [
      h("div", { className: "negotiation-turn recruiter", key: `${r.round}-r` }, [
        h("span", { className: "negotiation-round-tag" }, `Round ${r.round}`),
        h("p", null, r.recruiter_message),
      ]),
      h("div", { className: "negotiation-turn candidate", key: `${r.round}-c` }, [
        h("p", null, r.candidate_message),
      ]),
    ])),
    h("div", { className: "panel-title", style: { marginTop: 16 } }, [
      h("h3", null, "Acceptance Probability by Round"),
    ]),
    h(LineChart, { data: chartData, xKey: "round", yKey: "acceptance_probability", yAsPercent: true }),
    h("div", { className: "panel-title", style: { marginTop: 16 } }, [
      h("h3", null, "Round-by-Round Detail"),
    ]),
    h("div", { className: "table-wrap" }, [
      h("table", null, [
        h("thead", null, h("tr", null, ["Round", "Recruiter offer", "Candidate ask", "Acceptance probability"].map((head) => h("th", { key: head }, head)))),
        h("tbody", null, (result.rounds || []).map((r) =>
          h("tr", { key: r.round }, [
            h("td", null, r.round),
            h("td", null, fmtLpa(r.recruiter_offer)),
            h("td", null, fmtLpa(r.candidate_ask)),
            h("td", null, fmtPct(r.acceptance_probability)),
          ])
        )),
      ]),
    ]),
  ]);
}

function RiskRadar() {
  const [scan, setScan] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [expanded, setExpanded] = useState(null);

  function runScan() {
    setIsLoading(true);
    setError(null);
    api.riskScan({ queue_size: 40, risk_threshold: 0.55, top_n: 10 })
      .then(setScan)
      .catch((err) => setError(err.detail || "Unable to run risk scan"))
      .finally(() => setIsLoading(false));
  }

  useEffect(() => {
    runScan();
  }, []);

  useEffect(() => {
    if (!autoRefresh) return undefined;
    const id = setInterval(runScan, 20000);
    return () => clearInterval(id);
  }, [autoRefresh]);

  const flagged = scan?.flagged_offers || [];

  return h("div", null, [
    h("div", { className: "dashboard-header" }, [
      h("h2", null, "Risk Radar"),
      h("p", null, "An agent proactively scans the most recent offers and drafts escalation alerts for the ones at risk of being declined - no one has to ask it."),
    ]),
    h("div", { className: "quick-actions", style: { gridTemplateColumns: "auto auto 1fr" } }, [
      h("button", { className: "action-card", onClick: runScan, disabled: isLoading, style: { flex: "0 0 auto" } }, [
        h("div", { className: "action-icon sim" }, h(Icon, { name: "radar", size: 22 })),
        h("div", null, [h("strong", null, isLoading ? "Scanning..." : "Run scan now"), h("span", null, "Re-score the most recent offers")]),
      ]),
      h("label", { className: "action-card", style: { cursor: "pointer" } }, [
        h("input", {
          type: "checkbox",
          checked: autoRefresh,
          onChange: (e) => setAutoRefresh(e.target.checked),
          style: { width: 18, height: 18 },
        }),
        h("div", null, [h("strong", null, "Auto-refresh"), h("span", null, "Re-scan every 20 seconds")]),
      ]),
    ]),
    error && h("div", { className: "warning" }, error),
    scan && h("div", { className: "result-band compact-band" }, [
      h("div", { className: "result-stat" }, [h("span", null, "Offers scanned"), h("strong", null, scan.queue_scanned)]),
      h("div", { className: "result-stat" }, [h("span", null, "Flagged as at-risk"), h("strong", null, scan.flagged_count)]),
      h("div", { className: "result-stat" }, [h("span", null, "Risk threshold"), h("strong", null, fmtPct(scan.risk_threshold))]),
    ]),
    h("section", { className: "section", style: { marginTop: 16 } }, [
      h("div", { className: "panel-title" }, [
        h("h3", null, "Flagged Offers"),
        h("span", null, "Sorted by lowest predicted acceptance first"),
      ]),
      flagged.length
        ? h("div", { className: "table-wrap" }, [
            h("table", null, [
              h("thead", null, h("tr", null, ["Urgency", "Ref", "Skill", "Band", "Offered", "Acceptance", "Suggested CTC", "Actual outcome", ""].map((head) => h("th", { key: head }, head)))),
              h("tbody", null, flagged.flatMap((row) => ([
                h("tr", { key: row.candidate_ref }, [
                  h("td", null, h("span", { className: `tag ${row.urgency === "High" ? "bad" : "info"}` }, row.urgency)),
                  h("td", null, row.candidate_ref),
                  h("td", null, row.primary_skill),
                  h("td", null, h("span", { className: "tag" }, row.offered_band)),
                  h("td", null, fmtLpa(row.offered_ctc)),
                  h("td", null, fmtPct(row.acceptance_probability)),
                  h("td", null, row.suggested_ctc ? fmtLpa(row.suggested_ctc) : "Needs review"),
                  h("td", null, h("span", { className: `tag ${row.actual_outcome === "Joined" || row.actual_outcome === "Accepted" ? "good" : "bad"}` }, row.actual_outcome)),
                  h("td", null, h("button", {
                    className: "icon-button",
                    onClick: () => setExpanded(expanded === row.candidate_ref ? null : row.candidate_ref),
                  }, expanded === row.candidate_ref ? "Hide" : "Alert")),
                ]),
                expanded === row.candidate_ref && h("tr", { key: `${row.candidate_ref}-alert` }, [
                  h("td", { colSpan: 9 }, h("div", { className: "notice warn", style: { margin: 0 } }, row.alert_message)),
                ]),
              ]))),
            ]),
          ])
        : h("div", { className: "empty" }, isLoading ? "Scanning..." : "No at-risk offers found in the current queue."),
    ]),
  ]);
}

function ChatPane({ onUiAction }) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([
    {
      role: "model",
      content: "Hi. I can help explain offer benchmarks, run acceptance simulations, or open the dashboard views.",
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const endRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  function triggerFileUpload() {
    fileInputRef.current?.click();
  }

  function handleFileChange(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    event.target.value = "";
    setIsLoading(true);
    setMessages((current) => [...current, { role: "user", content: `Uploading Excel/CSV data file: ${file.name}` }]);

    const formData = new FormData();
    formData.append("file", file);

    fetch("/api/upload", {
      method: "POST",
      body: formData,
    })
      .then((res) => {
        if (!res.ok) return res.json().then((err) => Promise.reject(err));
        return res.json();
      })
      .then((data) => {
        setMessages((current) => [
          ...current,
          {
            role: "model",
            content: `Successfully uploaded ${file.name} (${data.record_count} records). The acceptance model has been re-trained (ROC AUC: ${data.metrics.roc_auc}). The dashboard and options have been updated!`,
          },
        ]);
        if (onUiAction) {
          onUiAction({ type: "DATA_UPDATED" });
        }
      })
      .catch((err) => {
        setMessages((current) => [
          ...current,
          {
            role: "model",
            content: `Upload failed: ${err.detail || err.message || "Unknown error"}`,
          },
        ]);
      })
      .finally(() => setIsLoading(false));
  }

  function submit(event) {
    event.preventDefault();
    const content = input.trim();
    if (!content || isLoading) return;

    const nextMessages = [...messages, { role: "user", content }];
    setMessages(nextMessages);
    setInput("");
    setIsLoading(true);

    api.chat(nextMessages)
      .then((data) => {
        setMessages((current) => [...current, { role: "model", content: data.response || "I could not generate a response." }]);
        (data.ui_actions || []).forEach((action) => onUiAction && onUiAction(action));
      })
      .catch((err) => {
        setMessages((current) => [...current, { role: "model", content: err.detail || "The assistant could not connect. Check backend/.env and server logs." }]);
      })
      .finally(() => setIsLoading(false));
  }

  return h("aside", { className: "chat-pane" }, [
    h("div", { className: "chat-header" }, [
      h("h3", null, [h(Icon, { name: "bot", size: 18 }), "AI Recruiter Assistant"]),
    ]),
    h("div", { className: "chat-messages" }, [
      messages.map((message, index) =>
        h("div", { key: `${index}-${message.role}`, className: `chat-message ${message.role === "user" ? "user" : "model"}` },
          String(message.content || "")
            .split("\n")
            .filter((line) => line.trim().length > 0)
            .map((line, lineIndex) => h("p", { key: lineIndex }, line))
        )
      ),
      isLoading && h("div", { className: "chat-typing" }, "Assistant is thinking..."),
      h("div", { ref: endRef }),
    ]),
    h("form", { className: "chat-input-area chat-input-form", onSubmit: submit }, [
      h("input", {
        type: "file",
        ref: fileInputRef,
        onChange: handleFileChange,
        accept: ".xlsx,.xls,.csv",
        style: { display: "none" },
        id: "excel-upload-input",
      }),
      h("button", {
        type: "button",
        onClick: triggerFileUpload,
        disabled: isLoading,
        "aria-label": "Upload Excel or CSV file",
        className: "upload-btn",
      }, h(Icon, { name: "paperclip", size: 18 })),
      h("input", {
        value: input,
        onChange: (event) => setInput(event.target.value),
        placeholder: "Ask about CTC, probability, benchmarks...",
        disabled: isLoading,
      }),
      h("button", { type: "submit", disabled: isLoading || !input.trim(), "aria-label": "Send message" }, h(Icon, { name: "send", size: 18 })),
    ]),
  ]);
}

function RightCanvas({ summary, options, candidates, activeTab, setActiveTab }) {
  const renderTab = () => {
    if (activeTab === "dashboard") return h(Dashboard, { summary, candidates, onNavigate: setActiveTab });
    if (activeTab === "simulator") return h(OfferSimulator, { options });
    if (activeTab === "negotiation") return h(NegotiationSimulator, { options });
    if (activeTab === "risk") return h(RiskRadar, null);
    if (activeTab === "table") return h(RecentOffersTable, { candidates });
    return h(Dashboard, { summary, candidates, onNavigate: setActiveTab });
  };

  return h("div", { className: "canvas-pane" }, [
    h("div", { className: "tabs-header" }, [
      h("button", { className: `tab-btn ${activeTab === "dashboard" ? "active" : ""}`, onClick: () => setActiveTab("dashboard") }, [h(Icon, { name: "layout-dashboard", size: 16 }), "Dashboard"]),
      h("button", { className: `tab-btn ${activeTab === "simulator" ? "active" : ""}`, onClick: () => setActiveTab("simulator") }, [h(Icon, { name: "calculator", size: 16 }), "Simulator"]),
      h("button", { className: `tab-btn ${activeTab === "negotiation" ? "active" : ""}`, onClick: () => setActiveTab("negotiation") }, [h(Icon, { name: "handshake", size: 16 }), "Negotiation Twin"]),
      h("button", { className: `tab-btn ${activeTab === "risk" ? "active" : ""}`, onClick: () => setActiveTab("risk") }, [h(Icon, { name: "radar", size: 16 }), "Risk Radar"]),
      h("button", { className: `tab-btn ${activeTab === "table" ? "active" : ""}`, onClick: () => setActiveTab("table") }, [h(Icon, { name: "table", size: 16 }), "Recent Offers"]),
    ]),
    h("div", { className: "canvas-content" }, renderTab()),
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
    if (action.type === "FILTER_UI" && action.tab) {
      setActiveTab(action.tab);
    } else if (action.type === "DATA_UPDATED") {
      Promise.all([api.summary(), api.options(), api.candidates()])
        .then(([summaryData, optionsData, candidateData]) => {
          setSummary(summaryData);
          setOptions(optionsData);
          setCandidates(candidateData);
        })
        .catch((err) => setError(String(err)));
    }
  }

  return h("div", { className: "app" }, [
    h("header", { className: "topbar" }, [
      h("div", { className: "brand" }, [
        h("div", { className: "brand-mark" }, h(Icon, { name: "badge-indian-rupee" })),
        h("div", null, [h("h1", null, "CTC Offer Intelligence"), h("p", null, "AI-assisted offer analytics and acceptance probability support")]),
      ]),
      h("div", { className: "status-pill" }, [h(Icon, { name: "database", size: 16 }), `${summary.kpis.total_offers} offers loaded`]),
    ]),
    h("main", { className: "main-layout" }, [
      h(ChatPane, { onUiAction: handleUiAction }),
      h(RightCanvas, { summary, options, candidates, activeTab, setActiveTab }),
    ]),
  ]);
}

ReactDOM.createRoot(document.getElementById("root")).render(h(App));
