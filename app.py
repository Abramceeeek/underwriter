import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data import (df, SALARY_BANDS, QUALIFICATIONS, OPEN_ROLES, SALARY_GROWTH,
                  KEY_EMPLOYERS, SPECIALISM_PREMIUM, QUAL_PREMIUM,
                  predict_fair_salary, budget_sensitivity, flight_risk_score,
                  score_candidate, build_funnel, salary_inflation_data,
                  head_to_head, get_risk_flags,
                  _model)

st.set_page_config(page_title="Leadenhall UW Intelligence", layout="wide")

BG = "#0f1e35"
FG = "#e2e8f0"
MID = "#94a3b8"
GOLD = "#D4AF37"
FONT = "Inter, sans-serif"

CHART_LAYOUT = dict(
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(color=FG, family=FONT),
    xaxis=dict(tickfont=dict(color=FG), title_font=dict(color=FG), color=FG),
    yaxis=dict(tickfont=dict(color=FG), title_font=dict(color=FG), color=FG),
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080f1c;
    color: #e2e8f0;
}
.stApp { background-color: #080f1c; }
h2, h3 { color: #f1f5f9 !important; }
hr { border-color: rgba(255,255,255,0.06) !important; }
.stDataFrame { border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; overflow: hidden; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.75rem !important; }
.stTabs [data-baseweb="tab-list"] { background: #0b1629; border-radius: 10px; gap: 4px; padding: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #64748b; border-radius: 8px; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { background: #0f2444 !important; color: #D4AF37 !important; }
.stSelectbox label, .stSlider label, .stNumberInput label, .stMultiSelect label,
.stCheckbox label, .stRadio label { color: #94a3b8 !important; }
[data-baseweb="select"] { background: #0f1e35 !important; border-color: rgba(255,255,255,0.1) !important; }
.stButton button {
    background: linear-gradient(135deg, #0f2444, #1e3a5f) !important;
    border: 1px solid rgba(212,175,55,0.4) !important;
    color: #D4AF37 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
.insight {
    background: rgba(212,175,55,0.06);
    border-left: 3px solid #D4AF37;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    color: #e2e8f0;
}
.act-box {
    background: rgba(16,185,129,0.04);
    border-left: 3px solid #10b981;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    line-height: 1.8;
    color: #e2e8f0;
}
.kcard {
    background: #0f1e35;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border: 1px solid rgba(255,255,255,0.06);
    border-top: 2px solid #D4AF37;
}
.kcard .lbl { font-size: 0.65rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 0.3rem; }
.kcard .val { font-size: 1.8rem; font-weight: 700; color: #f1f5f9; }
.kcard .sub { font-size: 0.72rem; color: #64748b; margin-top: 0.2rem; }
</style>""", unsafe_allow_html=True)

page = st.selectbox("Page", [
    "Market Overview",
    "Candidate Database",
    "Role Matcher & Scoring",
    "Actuarial Models",
    "Insights & Interpretation",
])

specialisms = sorted(df["Specialism"].unique())


def chart(fig, height=320, **kwargs):
    layout = {**CHART_LAYOUT, "height": height, "margin": dict(l=0, r=0, t=5, b=0)}
    layout.update(kwargs)
    fig.update_layout(**layout)
    return fig


if page == "Market Overview":
    st.subheader("Underwriter Recruitment Intelligence")
    st.caption("Hays 2026 · March 2026 vacancies · 200 profiles · Marine-focused")

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Total Candidates", str(len(df)), "Across 8 specialisms"),
        ("Immediately Available", str((df["Availability"] == "Immediately").sum()),
         f"{(df['Availability'] == 'Immediately').mean() * 100:.0f}% of pool"),
        ("Avg Market Salary", f"£{df['Current Salary (£)'].mean() / 1000:.0f}k", "London benchmark"),
        ("Lloyd's Experienced", "{:.0f}%".format(df["Lloyd's Experience"].mean() * 100), "Of candidate pool"),
        ("Elevated Flight Risk", str((df["Flight Risk (%)"] >= 50).sum()), "Ready to move"),
    ]
    for col, (lbl, val, sub) in zip([c1, c2, c3, c4, c5], kpis):
        col.markdown(
            f'<div class="kcard"><div class="lbl">{lbl}</div><div class="val">{val}</div><div class="sub">{sub}</div></div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average Salary by Specialism")
        d = df.groupby("Specialism")["Current Salary (£)"].mean().sort_values().reset_index()
        fig = px.bar(d, x="Current Salary (£)", y="Specialism", orientation="h",
                     color="Current Salary (£)", color_continuous_scale=["#1e3a5f", "#60a5fa"])
        st.plotly_chart(chart(fig, coloraxis_showscale=False, xaxis_tickformat="£,.0f", yaxis_title=""),
                        use_container_width=True)

    with col2:
        st.subheader("Salary Growth by Experience Band")
        df["_band"] = pd.cut(df["Years Experience"], bins=[0, 3, 7, 12, 25],
                             labels=["Entry (1-3y)", "Mid (4-7y)", "Senior (8-12y)", "Lead (13y+)"])
        d2 = df.groupby("_band", observed=True)["Current Salary (£)"].mean().reset_index()
        fig2 = px.line(d2, x="_band", y="Current Salary (£)", markers=True,
                       color_discrete_sequence=["#60a5fa"])
        fig2.update_traces(marker_size=10, line_width=3)
        st.plotly_chart(chart(fig2, yaxis_tickformat="£,.0f", xaxis_title="", yaxis_title="Avg Salary"),
                        use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Supply vs Demand by Specialism")
        supply = df.groupby("Specialism").size().reset_index(name="Candidates Available")
        demand = pd.DataFrame(list(OPEN_ROLES.items()), columns=["Specialism", "Open Roles"])
        sd = supply.merge(demand, on="Specialism")
        sd_m = sd.melt("Specialism", var_name="Type", value_name="Count")
        fig3 = px.bar(sd_m, x="Specialism", y="Count", color="Type", barmode="group",
                      color_discrete_map={"Candidates Available": "#60a5fa", "Open Roles": "#dc2626"})
        st.plotly_chart(chart(fig3, 300, xaxis_title="",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02)),
                        use_container_width=True)

    with col4:
        st.subheader("Availability Timeline by Specialism")
        avail_data = df.groupby(["Specialism", "Availability"]).size().reset_index(name="Count")
        fig4 = px.bar(avail_data, x="Specialism", y="Count", color="Availability",
                      color_discrete_map={"Immediately": "#16a34a",
                                          "1 month notice": "#d97706",
                                          "3 months notice": "#dc2626"})
        st.plotly_chart(chart(fig4, 300, xaxis_title="",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02)),
                        use_container_width=True)

    st.subheader("Candidate Density — Specialism x Experience Band")
    heat = df.groupby(["Specialism", "_band"], observed=True).size().reset_index(name="Count")
    pivot = heat.pivot(index="Specialism", columns="_band", values="Count").fillna(0)
    fig5 = px.imshow(pivot, color_continuous_scale=["#0f2444", "#1e3a5f", "#60a5fa"], aspect="auto",
                     labels=dict(color="Candidates"))
    fig5.update_layout(margin=dict(l=0, r=0, t=5, b=0), height=280,
                       plot_bgcolor=BG, paper_bgcolor=BG, font=dict(color=FG),
                       coloraxis=dict(colorbar=dict(tickfont=dict(color=FG), title_font=dict(color=FG))))
    st.plotly_chart(fig5, use_container_width=True)


elif page == "Candidate Database":
    st.subheader("Candidate Database")
    st.caption("Browse and filter 200 London Market underwriter profiles")

    c1, c2, c3, c4 = st.columns(4)
    spec_f = c1.multiselect("Specialism", specialisms, placeholder="All")
    exp_r = c2.slider("Years Experience", 1, 20, (1, 20))
    mkt_f = c3.multiselect("Market", df["Market"].unique(), placeholder="All")
    avail_f = c4.multiselect("Availability", df["Availability"].unique(), placeholder="All")

    filt = df.copy()
    if spec_f:  filt = filt[filt["Specialism"].isin(spec_f)]
    if mkt_f:   filt = filt[filt["Market"].isin(mkt_f)]
    if avail_f: filt = filt[filt["Availability"].isin(avail_f)]
    filt = filt[(filt["Years Experience"] >= exp_r[0]) & (filt["Years Experience"] <= exp_r[1])]

    st.markdown(f"**{len(filt)} candidates** match your filters")
    st.divider()

    disp = filt[["ID", "Name", "Specialism", "Title", "Years Experience", "Tenure (Current Role)",
                 "Current Salary (£)", "Salary Expectation (£)", "Qualification",
                 "Market", "Availability", "Flight Risk (%)"]].copy()
    disp["Current Salary (£)"] = disp["Current Salary (£)"].apply(lambda x: f"£{x:,}")
    disp["Salary Expectation (£)"] = disp["Salary Expectation (£)"].apply(lambda x: f"£{x:,}")
    st.dataframe(disp, use_container_width=True, height=420)

    st.divider()
    c_a, c_b, c_c = st.columns(3)
    with c_a:
        fig = px.histogram(filt, x="Current Salary (£)", nbins=12,
                           color_discrete_sequence=["#60a5fa"], title="Salary Distribution")
        st.plotly_chart(chart(fig, 260, margin=dict(l=0, r=0, t=30, b=0)), use_container_width=True)
    with c_b:
        fig2 = px.violin(filt, x="Specialism", y="Years Experience",
                         box=True, color_discrete_sequence=["#60a5fa"], title="Experience Distribution")
        st.plotly_chart(chart(fig2, 260, margin=dict(l=0, r=0, t=30, b=0), xaxis_title=""),
                        use_container_width=True)
    with c_c:
        fig3 = px.scatter(filt, x="Tenure (Current Role)", y="Flight Risk (%)",
                          color="Specialism", hover_data=["Name", "Years Experience"],
                          title="Tenure vs Flight Risk")
        st.plotly_chart(chart(fig3, 260, margin=dict(l=0, r=0, t=30, b=0)), use_container_width=True)


elif page == "Role Matcher & Scoring":
    st.subheader("Role Matcher & Candidate Scoring")
    st.caption("Defaults to Marine Cargo. Switch specialism to explore other lines.")

    marine_idx = sorted(df["Specialism"].unique()).index("Marine Cargo") if "Marine Cargo" in df["Specialism"].unique() else 0

    c1, c2, c3 = st.columns(3)
    role_spec = c1.selectbox("Specialism Required", specialisms, index=marine_idx)
    min_years = c1.slider("Min Years Experience", 1, 15, 4)
    salary_bgt = c2.number_input("Salary Budget (£)", 30000, 220000, 75000, 5000, format="%d")
    lloyds_req = c2.checkbox("Lloyd's Experience Required?", True)
    qual_req = c3.selectbox("Minimum Qualification", QUALIFICATIONS, index=2)
    top_n = c3.slider("Show Top N Candidates", 3, 15, 5)

    role = dict(specialism=role_spec, min_years=min_years, salary_budget=salary_bgt,
                lloyds_required=lloyds_req, qualification_required=qual_req)

    if st.button("Find Best Candidates", type="primary", use_container_width=True):
        results = []
        for _, row in df.iterrows():
            sc, notes = score_candidate(row, role)
            results.append({**row.to_dict(), "Match Score": sc, "Notes": notes})
        rdf = pd.DataFrame(results).sort_values("Match Score", ascending=False)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Rankings", "Head-to-Head", "Risk Flags", "Candidate Funnel", "Radar — Top Candidate",
        ])

        with tab1:
            top = rdf.head(top_n)
            colors = ["#10b981" if s >= 70 else "#f59e0b" if s >= 50 else "#ef4444"
                      for s in top["Match Score"]]
            fig = go.Figure(go.Bar(
                x=top["Match Score"], y=top["Name"], orientation="h",
                marker_color=colors,
                text=[f"{s}%" for s in top["Match Score"]], textposition="outside"
            ))
            st.plotly_chart(chart(fig, xaxis=dict(range=[0, 115], title="Match Score (%)",
                                                  tickfont=dict(color=FG), title_font=dict(color=FG), color=FG),
                                  yaxis=dict(title="", tickfont=dict(color=FG), color=FG),
                                  margin=dict(l=0, r=60, t=5, b=0)),
                            use_container_width=True)
            st.divider()

            for rank, (_, row) in enumerate(top.iterrows(), 1):
                sc = row["Match Score"]
                label = "Strong Match" if sc >= 70 else "Possible Match" if sc >= 50 else "Weak Match"
                fr_pct, fr_label, _ = flight_risk_score(
                    row["Tenure (Current Role)"], row["Specialism"], row["Years Experience"])
                with st.expander(
                        f"#{rank}  {row['Name']} · {row['Title']} · {row['Years Experience']} yrs · {row['Specialism']}",
                        expanded=rank <= 2):
                    ca, cb = st.columns([2, 1])
                    with ca:
                        st.markdown(f"""
| Field | Detail |
|---|---|
| **Current Salary** | £{row['Current Salary (£)']:,} |
| **Salary Expectation** | £{row['Salary Expectation (£)']:,} |
| **Qualification** | {row['Qualification']} |
| **Market** | {row['Market']} |
| **Lloyd's Exp** | {'Yes' if row["Lloyd's Experience"] else 'No'} |
| **Availability** | {row['Availability']} |
| **Tenure (current role)** | {row['Tenure (Current Role)']} yrs |
| **Flight Risk** | {fr_label} ({fr_pct * 100:.0f}%) |""")
                    with cb:
                        st.metric("Match Score", f"{sc}/100", label)
                        st.markdown("**Assessment**")
                        for note in row["Notes"]:
                            st.markdown(
                                f"<div style='font-size:.82rem;margin-bottom:3px'>{note}</div>",
                                unsafe_allow_html=True)

        with tab2:
            n_compare = min(top_n, 5)
            h2h_df = head_to_head(rdf.head(n_compare), role)
            st.dataframe(h2h_df, use_container_width=True, hide_index=True)

            top1n = rdf.iloc[0]["Name"]
            top1s = rdf.iloc[0]["Match Score"]
            if len(rdf) > 1:
                top2n = rdf.iloc[1]["Name"]
                top2s = rdf.iloc[1]["Match Score"]
                gap = top1s - top2s
                st.markdown(
                    f"<div class='act-box'><b>Comparison summary:</b> "
                    f"<b>{top1n}</b> leads the shortlist with <b>{top1s}/100</b>, "
                    f"{gap} points ahead of {top2n} ({top2s}/100). "
                    f"{'The gap is significant — primary recommendation is clear.' if gap >= 10 else 'Scores are close — client interview needed to differentiate.'}"
                    f"</div>",
                    unsafe_allow_html=True)

        with tab3:

            for rank, (_, row) in enumerate(rdf.head(top_n).iterrows(), 1):
                sc = row["Match Score"]
                flags = get_risk_flags(row.to_dict(), role)
                label = "Strong Match" if sc >= 70 else "Possible Match" if sc >= 50 else "Weak Match"

                with st.expander(f"#{rank}  {row['Name']} · {sc}/100 · {label}", expanded=rank == 1):
                    for severity, flag_name, detail in flags:
                        colour = "#ef4444" if "High" in severity else \
                            "#f59e0b" if "Moderate" in severity else "#10b981"
                        st.markdown(
                            f"<div style='background:rgba(255,255,255,0.03);"
                            f"border-left:3px solid {colour};border-radius:6px;"
                            f"padding:0.6rem 0.9rem;margin-bottom:0.5rem;font-size:0.83rem;'>"
                            f"<span style='color:{colour};font-weight:700;'>{severity} — {flag_name}</span>"
                            f"<br><span style='color:#94a3b8;'>{detail}</span></div>",
                            unsafe_allow_html=True)

        with tab4:
            funnel_df = build_funnel(role)
            fig_f = go.Figure(go.Funnel(
                y=funnel_df["Stage"], x=funnel_df["Count"],
                textinfo="value+percent initial",
                marker_color=["#D4AF37", "#60a5fa", "#3b82f6", "#2563eb", "#1e3a5f", "#0f2444"],
            ))
            fig_f.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=380,
                                plot_bgcolor=BG, paper_bgcolor=BG, font=dict(color=FG))
            st.plotly_chart(fig_f, use_container_width=True)
            st.dataframe(funnel_df, use_container_width=True, hide_index=True)

        with tab5:
            top1 = rdf.iloc[0]
            qual_order = {q: i for i, q in enumerate(QUALIFICATIONS)}

            spec_sc = 30 if top1["Specialism"] == role_spec else 0
            exp_sc = (min(25, max(0, 25 - (max(0, (top1["Years Experience"] - min_years - 3)) * 2)))
                      if top1["Years Experience"] >= min_years else 0)
            sal_sc = (20 if top1["Salary Expectation (£)"] <= salary_bgt
                      else 10 if top1["Salary Expectation (£)"] <= salary_bgt * 1.1 else 0)
            lly_sc = (10 if (lloyds_req and top1["Lloyd's Experience"])
                      else 5 if not lloyds_req else 0)
            qual_sc = (10 if qual_order.get(top1["Qualification"], 0) >= qual_order.get(qual_req, 0)
                       else 0)

            cats = ["Specialism<br>(30pts)", "Experience<br>(25pts)", "Salary Fit<br>(20pts)",
                    "Lloyd's<br>(10pts)", "Qualification<br>(10pts)"]
            vals = [spec_sc / 30 * 100, exp_sc / 25 * 100, sal_sc / 20 * 100,
                    lly_sc / 10 * 100, qual_sc / 10 * 100]
            vals_c = vals + [vals[0]]
            cats_c = cats + [cats[0]]

            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=vals_c, theta=cats_c, fill="toself", name=top1["Name"],
                line_color=GOLD, fillcolor="rgba(212,175,55,0.12)"))
            fig_r.update_layout(
                polar=dict(
                    radialaxis=dict(range=[0, 100], tickfont_size=9, tickfont_color=FG),
                    angularaxis=dict(tickfont=dict(color=FG))),
                height=380, margin=dict(l=40, r=40, t=30, b=30),
                showlegend=True, plot_bgcolor=BG, paper_bgcolor=BG, font=dict(color=FG))
            st.plotly_chart(fig_r, use_container_width=True)
            st.info(f"**#1 candidate: {top1['Name']}** — Overall match score: {rdf.iloc[0]['Match Score']}/100")




elif page == "Actuarial Models":
    st.subheader("Actuarial Models")
    st.caption("Salary pricing regression  ·  Sensitivity analysis  ·  Flight risk survival model")

    tab1, tab2, tab3 = st.tabs(["Salary Pricing Model", "Sensitivity Analysis", "Flight Risk Model"])

    with tab1:

        c1, c2 = st.columns(2)
        p_spec = c1.selectbox("Specialism", specialisms, key="pm_spec")
        p_yrs = c1.slider("Years Experience", 1, 20, 6, key="pm_yrs")
        p_lly = c2.checkbox("Lloyd's Experience?", True, key="pm_lly")
        p_qual = c2.selectbox("Qualification", QUALIFICATIONS, index=3, key="pm_qual")

        pt, lo, hi = predict_fair_salary(p_yrs, p_spec, p_lly, p_qual)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Expected Fair Salary", f"£{pt:,}")
        mc2.metric("95% CI Lower Bound", f"£{lo:,}")
        mc3.metric("95% CI Upper Bound", f"£{hi:,}")

        st.divider()
        st.subheader("Model Coefficients")
        coef_df = pd.DataFrame({
            "Factor": ["Years Experience", "Specialism Premium Index", "Lloyd's Flag", "Qualification Premium"],
            "Coefficient (£)": [f"£{c:,.0f}" for c in _model.coef_],
            "Interpretation": [
                f"Each additional year adds ~£{_model.coef_[0]:,.0f} to expected salary",
                f"Specialism premium multiplier (Reinsurance=1.42 vs Marine Liability=1.00 baseline)",
                f"Lloyd's syndicate experience adds ~£{_model.coef_[2]:,.0f} on average",
                f"Each qualification tier adds defined loading (ACII = +£6,000, FCII = +£11,000)",
            ]
        })
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

    with tab2:

        c1, c2, c3 = st.columns(3)
        sa_spec = c1.selectbox("Specialism", specialisms, key="sa_spec")
        sa_yrs = c1.slider("Min Years Exp", 1, 12, 4, key="sa_yrs")
        sa_bgt = c2.number_input("Base Budget (£)", 30000, 180000, 70000, 5000,
                                 format="%d", key="sa_bgt")
        sa_lly = c2.checkbox("Lloyd's Required?", True, key="sa_lly")
        sa_qual = c3.selectbox("Min Qualification", QUALIFICATIONS, index=1, key="sa_qual")

        sens_df = budget_sensitivity(sa_spec, sa_yrs, sa_lly, sa_qual, sa_bgt)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sens_df["Budget (£)"], y=sens_df["Candidate Count"],
            mode="lines+markers", name="Candidate Count",
            line=dict(color="#60a5fa", width=3), marker_size=6,
            fill="tozeroy", fillcolor="rgba(96,165,250,0.08)"
        ))
        fig.add_vline(x=sa_bgt, line_dash="dash", line_color="#dc2626",
                      annotation_text="Current Budget", annotation_position="top right")
        fig.update_layout(
            plot_bgcolor=BG, paper_bgcolor=BG, height=340,
            margin=dict(l=0, r=0, t=10, b=0),
            font=dict(color=FG, family=FONT),
            xaxis=dict(title="Salary Budget (£)", tickformat="£,.0f",
                       tickfont=dict(color=FG), title_font=dict(color=FG), color=FG),
            yaxis=dict(title="Available Candidates", rangemode="tozero",
                       tickfont=dict(color=FG), title_font=dict(color=FG), color=FG)
        )
        st.plotly_chart(fig, use_container_width=True)

        c_base = sens_df.iloc[(sens_df["Budget (£)"] - sa_bgt).abs().argsort()[:1]]["Candidate Count"].values[0]
        up5 = int(sa_bgt * 1.05)
        up10 = int(sa_bgt * 1.10)
        c_up5 = sens_df.iloc[(sens_df["Budget (£)"] - up5).abs().argsort()[:1]]["Candidate Count"].values[0]
        c_up10 = sens_df.iloc[(sens_df["Budget (£)"] - up10).abs().argsort()[:1]]["Candidate Count"].values[0]

        st.caption(f"At £{sa_bgt:,}: {c_base} candidates · +5% (£{up5:,}): {c_up5} · +10% (£{up10:,}): {c_up10}")

    with tab3:
        st.subheader("Tenure Probability Curve")
        spec_for_curve = st.selectbox("Select Specialism for Curve", specialisms, key="fr_spec")
        tenure_range = np.linspace(0.1, 9, 100)

        curve_probs = [flight_risk_score(t, spec_for_curve, 6)[0] * 100 for t in tenure_range]
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=tenure_range, y=curve_probs,
                                   mode="lines", line=dict(color="#60a5fa", width=3),
                                   fill="tozeroy", fillcolor="rgba(96,165,250,0.1)",
                                   name="Flight Risk %"))
        fig_c.add_hline(y=50, line_dash="dash", line_color="#dc2626",
                        annotation_text="High Risk Threshold (50%)")
        fig_c.add_hline(y=30, line_dash="dot", line_color="#d97706",
                        annotation_text="Moderate Risk (30%)")
        fig_c.update_layout(
            plot_bgcolor=BG, paper_bgcolor=BG, height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            font=dict(color=FG, family=FONT),
            xaxis=dict(title="Tenure in Current Role (years)",
                       tickfont=dict(color=FG), title_font=dict(color=FG), color=FG),
            yaxis=dict(title="Flight Risk Probability (%)", range=[0, 75],
                       tickfont=dict(color=FG), title_font=dict(color=FG), color=FG)
        )
        st.plotly_chart(fig_c, use_container_width=True)

        st.divider()
        st.subheader("Flight Risk Distribution — Current Pool")
        fr_dist = df.groupby("Specialism")["Flight Risk (%)"].mean().sort_values(ascending=False).reset_index()
        fig_fr = px.bar(fr_dist, x="Specialism", y="Flight Risk (%)",
                        color="Flight Risk (%)", color_continuous_scale=["#1e3a5f", "#ef4444"])
        st.plotly_chart(chart(fig_fr, 280, coloraxis_showscale=False, xaxis_title="",
                              yaxis_title="Avg Flight Risk (%)"),
                        use_container_width=True)

        st.divider()
        st.subheader("High-Priority Candidates — Ready to Move Now")
        hot = df[df["Flight Risk (%)"] >= 50].sort_values("Flight Risk (%)", ascending=False)
        if len(hot) == 0:
            st.info("No candidates currently above 50% flight risk threshold.")
        else:
            display_hot = hot[["Name", "Specialism", "Title", "Years Experience",
                               "Tenure (Current Role)", "Flight Risk (%)", "Availability",
                               "Current Salary (£)"]].copy()
            display_hot["Current Salary (£)"] = display_hot["Current Salary (£)"].apply(lambda x: f"£{x:,}")
            st.dataframe(display_hot, use_container_width=True, hide_index=True)


elif page == "Insights & Interpretation":
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Salary Premium vs Market Average")
        avg_all = df["Current Salary (£)"].mean()
        spec_avg = df.groupby("Specialism")["Current Salary (£)"].mean()
        prem = ((spec_avg - avg_all) / avg_all * 100).sort_values().reset_index()
        prem.columns = ["Specialism", "Premium (%)"]
        fig = go.Figure(go.Bar(
            x=prem["Premium (%)"], y=prem["Specialism"], orientation="h",
            marker_color=["#10b981" if v > 0 else "#ef4444" for v in prem["Premium (%)"]],
            text=[f"{v:+.1f}%" for v in prem["Premium (%)"]],
            textposition="outside"
        ))
        fig.add_vline(x=0, line_color="#aaa", line_width=1)
        st.plotly_chart(chart(fig, margin=dict(l=0, r=60, t=5, b=0),
                              xaxis_title="vs Market Average", yaxis_title=""),
                        use_container_width=True)

    with col2:
        st.subheader("Salary Inflation Tracker (2024-2026)")
        infl = salary_inflation_data()
        fig2 = px.line(infl, x="Year", y="Avg Salary", color="Specialism",
                       markers=True, color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(chart(fig2, margin=dict(l=0, r=0, t=10, b=0),
                              yaxis_tickformat="£,.0f", yaxis_title="Avg Salary (£)", xaxis_title=""),
                        use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Salary Expectation Gap by Specialism")
        df["Gap (£)"] = df["Salary Expectation (£)"] - df["Current Salary (£)"]
        gap = df.groupby("Specialism")["Gap (£)"].mean().sort_values(ascending=False).reset_index()
        fig3 = px.bar(gap, x="Specialism", y="Gap (£)",
                      color="Gap (£)", color_continuous_scale=["#1e3a5f", "#60a5fa"])
        st.plotly_chart(chart(fig3, 300, coloraxis_showscale=False, xaxis_title="", yaxis_tickformat="£,.0f"),
                        use_container_width=True)

    with col4:
        st.subheader("Salary vs Experience (All Specialisms)")
        fig4 = px.scatter(df, x="Years Experience", y="Current Salary (£)",
                          color="Specialism", hover_data=["Name", "Title"],
                          color_discrete_sequence=px.colors.qualitative.Safe)
        z = np.polyfit(df["Years Experience"], df["Current Salary (£)"], 1)
        x_line = np.linspace(df["Years Experience"].min(), df["Years Experience"].max(), 100)
        fig4.add_trace(go.Scatter(x=x_line, y=np.polyval(z, x_line),
                                  mode="lines", name="Market Trend",
                                  line=dict(color=GOLD, width=2, dash="dash")))
        st.plotly_chart(chart(fig4, 300, yaxis_tickformat="£,.0f"), use_container_width=True)

    st.divider()
    st.subheader("Market Concentration — Candidate Pool Treemap")
    tree = df.groupby("Specialism").agg(
        Candidates=("Name", "count"),
        Avg_Salary=("Current Salary (£)", "mean")).reset_index()
    tree["label"] = tree.apply(
        lambda r: f"{r['Specialism']}<br>{r['Candidates']} candidates<br>£{r['Avg_Salary']:,.0f} avg", axis=1)
    fig5 = px.treemap(tree, path=["Specialism"], values="Candidates",
                      color="Avg_Salary", color_continuous_scale=["#0f2444", "#1e3a5f", "#60a5fa"],
                      custom_data=["Candidates", "Avg_Salary"])
    fig5.update_traces(
        texttemplate="<b>%{label}</b><br>%{customdata[0]} candidates<br>£%{customdata[1]:,.0f}",
        textfont_size=13)
    fig5.update_layout(margin=dict(l=0, r=0, t=5, b=0), height=320, font=dict(color=FG),
                       coloraxis_colorbar=dict(title="Avg Salary"))
    st.plotly_chart(fig5, use_container_width=True)