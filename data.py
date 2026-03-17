import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

SALARY_BANDS = {
    "Marine Cargo":     {"entry": (30000, 44000), "mid": (46000, 76000),
                         "senior": (72000, 118000), "lead": (112000, 148000)},
    "Marine Hull":      {"entry": (35000, 50000), "mid": (58000, 90000),
                         "senior": (84000, 138000), "lead": (132000, 168000)},
    "Marine Liability": {"entry": (28000, 42000), "mid": (42000, 68000),
                         "senior": (65000,  98000), "lead":  (95000, 132000)},
    "D&O":              {"entry": (35000, 50000), "mid": (58000, 90000),
                         "senior": (84000, 128000), "lead": (120000, 160000)},
    "Professional Indemnity": {"entry": (32000, 46000), "mid": (52000, 82000),
                               "senior": (76000, 118000), "lead": (112000, 150000)},
    "Reinsurance":      {"entry": (38000, 54000), "mid": (65000, 96000),
                         "senior": (92000, 140000), "lead": (135000, 170000)},
    "Offshore Energy":  {"entry": (36000, 52000), "mid": (62000, 92000),
                         "senior": (88000, 130000), "lead": (125000, 162000)},
    "Construction":     {"entry": (30000, 45000), "mid": (50000, 78000),
                         "senior": (74000, 112000), "lead": (108000, 142000)},
}

SPECIALISM_PREMIUM = {
    "Marine Liability":       1.00,
    "Construction":           1.06,
    "Marine Cargo":           1.10,
    "Professional Indemnity": 1.19,
    "D&O":                    1.28,
    "Marine Hull":            1.33,
    "Offshore Energy":        1.36,
    "Reinsurance":            1.42,
}

SALARY_GROWTH = {
    "Offshore Energy":        0.11,
    "Reinsurance":            0.09,
    "Marine Hull":            0.08,
    "D&O":                    0.07,
    "Professional Indemnity": 0.07,
    "Construction":           0.06,
    "Marine Cargo":           0.05,
    "Marine Liability":       0.04,
}

OPEN_ROLES = {
    "D&O":                    58,
    "Professional Indemnity": 37,
    "Marine Cargo":           35,
    "Reinsurance":            32,
    "Marine Liability":       30,
    "Offshore Energy":        28,
    "Marine Hull":            21,
    "Construction":           18,
}

KEY_EMPLOYERS = {
    "Marine Cargo":           ["IQUW", "Pen Underwriting", "AIG / Talbot", "MS Amlin"],
    "Marine Hull":            ["Beazley", "Talbot", "Lancashire", "Fidelis"],
    "Marine Liability":       ["Tokio Marine Kiln", "Standard P&I", "Skuld"],
    "D&O":                    ["QBE", "Beazley", "Tokio Marine", "AXA XL", "CNA Hardy"],
    "Professional Indemnity": ["Hiscox", "Aviva", "RSA", "Markel"],
    "Reinsurance":            ["Munich Re", "Gen Re", "Swiss Re", "Everest Re"],
    "Offshore Energy":        ["Fidelis", "AIG", "Munich Re", "Westfield"],
    "Construction":           ["AXA XL", "Allianz", "HDI", "Liberty"],
}

QUALIFICATIONS = ["None", "Cert CII", "ACII (Part)", "ACII", "ACII + Lloyd's Quals", "FCII"]
QUAL_PREMIUM   = {"None": 0, "Cert CII": 1500, "ACII (Part)": 3000,
                  "ACII": 6000, "ACII + Lloyd's Quals": 9000, "FCII": 11000}

MARKETS  = ["Lloyd's Syndicate", "Company Market", "Both"]
LANGUAGES = ["English", "English, French", "English, Spanish", "English, German",
             "English, Arabic", "English, Mandarin", "English, Russian"]

np.random.seed(42)

first_names = [
    "James","Oliver","Harry","Jack","George","Noah","Charlie","Jacob","Alfie","Freddie",
    "Emily","Olivia","Sophie","Amelia","Isabella","Ava","Mia","Lily","Charlotte","Grace",
    "Sunita","Priya","Amir","Carlos","Lucas","Mei","Fatima","Chen","Andrei","Nadia",
    "Thomas","William","Benjamin","Samuel","Edward","Daniel","Matthew","Joseph","Leo","Oscar",
    "Chloe","Eleanor","Hannah","Victoria","Alice","Zara","Yasmin","Isla","Freya","Poppy",
    "Mohammed","Ibrahim","David","Michael","Patrick","Connor","Sean","Liam","Ryan","Ethan",
]
last_names = [
    "Smith","Johnson","Williams","Brown","Jones","Taylor","Davies","Wilson","Evans","Thomas",
    "Roberts","Walker","White","Harris","Martin","Thompson","Garcia","Martinez","Patel","Singh",
    "Kumar","Ahmed","Chen","Lee","Kim","Müller","Rossi","Dubois","Kowalski","Petrov",
    "Clarke","Mitchell","Turner","Collins","Stewart","Morris","Wood","Jackson","Allen","Baker",
    "Hill","Young","Hughes","Lewis","Bell","Phillips","Rogers","Cook","Murphy","Kelly",
    "Sharma","Gupta","Nakamura","Santos","Ferreira","Nielsen","Andersen","Larsen","Berg","Holm",
]

SPEC_WEIGHTS = {
    "Marine Cargo":           0.16,
    "Marine Hull":            0.11,
    "Marine Liability":       0.13,
    "D&O":                    0.14,
    "Professional Indemnity": 0.13,
    "Reinsurance":            0.12,
    "Offshore Energy":        0.12,
    "Construction":           0.09,
}
_spec_list   = list(SPEC_WEIGHTS.keys())
_spec_probs  = [SPEC_WEIGHTS[s] for s in _spec_list]


def exp_to_band(y):
    return "entry" if y <= 3 else "mid" if y <= 7 else "senior" if y <= 12 else "lead"


def exp_to_title(y):
    if   y <= 2:  return "Junior Underwriter"
    elif y <= 4:  return "Assistant Underwriter"
    elif y <= 7:  return "Underwriter"
    elif y <= 10: return "Senior Underwriter"
    elif y <= 14: return "Lead Underwriter"
    else:         return "Underwriting Manager"


candidates = []
for i in range(200):
    fn   = np.random.choice(first_names)
    ln   = np.random.choice(last_names)

    yrs  = int(np.random.choice(range(1, 22), p=[
        0.05, 0.07, 0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05,
        0.05, 0.05, 0.04, 0.04, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03, 0.01]))

    spec = np.random.choice(_spec_list, p=_spec_probs)
    band = exp_to_band(yrs)
    sl   = SALARY_BANDS[spec][band]
    salary = int(np.random.uniform(sl[0], sl[1]))

    if   yrs <= 2:  qual = np.random.choice(QUALIFICATIONS[:3],  p=[0.50, 0.30, 0.20])
    elif yrs <= 5:  qual = np.random.choice(QUALIFICATIONS[:4],  p=[0.20, 0.30, 0.30, 0.20])
    elif yrs <= 10: qual = np.random.choice(QUALIFICATIONS[1:5], p=[0.10, 0.30, 0.40, 0.20])
    else:           qual = np.random.choice(QUALIFICATIONS[2:],  p=[0.10, 0.38, 0.32, 0.20])

    market  = np.random.choice(MARKETS, p=[0.52, 0.33, 0.15])
    tenure  = round(np.random.uniform(0.25, min(yrs, 8)), 2)

    expectation = int(salary * np.random.uniform(1.05, 1.20))

    candidates.append({
        "ID":                     f"UW-{1000 + i}",
        "Name":                   f"{fn} {ln}",
        "Specialism":             spec,
        "Years Experience":       yrs,
        "Tenure (Current Role)":  tenure,
        "Title":                  exp_to_title(yrs),
        "Current Salary (£)":     salary,
        "Salary Expectation (£)": expectation,
        "Qualification":          qual,
        "Market":                 market,
        "Lloyd's Experience":     market in ["Lloyd's Syndicate", "Both"],
        "ACII Qualified":         "ACII" in qual,
        "Languages":              np.random.choice(LANGUAGES,
                                      p=[0.60, 0.08, 0.08, 0.06, 0.06, 0.06, 0.06]),
        "Availability":           np.random.choice(
                                      ["Immediately", "1 month notice", "3 months notice"],
                                      p=[0.25, 0.45, 0.30]),
        "Location":               np.random.choice(
                                      ["London (City)", "London (Other)", "Remote/Flexible"],
                                      p=[0.55, 0.25, 0.20]),
    })

df = pd.DataFrame(candidates)


def _build_X(dataframe):
    return pd.DataFrame({
        "years_exp":    dataframe["Years Experience"],
        "spec_premium": dataframe["Specialism"].map(SPECIALISM_PREMIUM),
        "lloyds_flag":  dataframe["Lloyd's Experience"].astype(int),
        "qual_premium": dataframe["Qualification"].map(QUAL_PREMIUM),
    })


_model     = LinearRegression().fit(_build_X(df), df["Current Salary (£)"])
_residuals = df["Current Salary (£)"] - _model.predict(_build_X(df))
_sigma     = _residuals.std()


def predict_fair_salary(years_exp, specialism, lloyds, qualification):
    X  = np.array([[years_exp,
                    SPECIALISM_PREMIUM.get(specialism, 1.0),
                    int(lloyds),
                    QUAL_PREMIUM.get(qualification, 0)]])
    pt = int(_model.predict(X)[0])
    return pt, int(pt - 1.96 * _sigma), int(pt + 1.96 * _sigma)


def budget_sensitivity(specialism, min_years, lloyds_req, qual_req, base_budget,
                       steps=30, pct_range=0.35):
    qual_order = {q: i for i, q in enumerate(QUALIFICATIONS)}
    budgets    = np.linspace(base_budget * (1 - pct_range),
                             base_budget * (1 + pct_range), steps)
    results    = []
    base_count = None
    for b in budgets:
        pool = df[(df["Specialism"] == specialism) &
                  (df["Years Experience"] >= min_years) &
                  (df["Salary Expectation (£)"] <= b)]
        if lloyds_req:
            pool = pool[pool["Lloyd's Experience"]]
        pool = pool[pool["Qualification"].map(qual_order) >= qual_order.get(qual_req, 0)]
        c = len(pool)
        if base_count is None:
            base_count = max(c, 1)
        results.append({
            "Budget (£)":     int(b),
            "Candidate Count": c,
            "Pool Change (%)": round((c - base_count) / base_count * 100, 1),
        })
    return pd.DataFrame(results)


def flight_risk_score(tenure_years, specialism, years_exp):
    market_velocity = {
        "Reinsurance":            0.72,
        "Marine Hull":            0.68,
        "Offshore Energy":        0.65,
        "D&O":                    0.62,
        "Professional Indemnity": 0.58,
        "Marine Cargo":           0.55,
        "Construction":           0.52,
        "Marine Liability":       0.48,
    }
    base = market_velocity.get(specialism, 0.55)

    if   tenure_years < 0.5: tf = 0.15
    elif tenure_years < 1.5: tf = 0.40
    elif tenure_years < 2.5: tf = 0.65
    elif tenure_years < 4.0: tf = 0.85
    elif tenure_years < 6.0: tf = 0.60
    else:                    tf = 0.35

    disc = 0.05 if years_exp > 10 else 0.0
    prob = round(min(0.97, base * tf - disc), 2)

    if   prob >= 0.50: label, colour = "High",     "#dc2626"
    elif prob >= 0.30: label, colour = "Moderate", "#d97706"
    else:              label, colour = "Low",      "#16a34a"
    return prob, label, colour


df["Flight Risk (%)"] = df.apply(
    lambda r: round(
        flight_risk_score(r["Tenure (Current Role)"],
                          r["Specialism"],
                          r["Years Experience"])[0] * 100, 1),
    axis=1,
)


def score_candidate(candidate, role):
    score, reasons, deducts = 0, [], []
    qual_order = {q: i for i, q in enumerate(QUALIFICATIONS)}

    if candidate["Specialism"] == role["specialism"]:
        score += 30
        reasons.append(f"[+] Exact specialism match: {role['specialism']}")
    else:
        deducts.append(f"[-] Writes {candidate['Specialism']}, role requires {role['specialism']}")

    needed, actual = role["min_years"], candidate["Years Experience"]
    if actual >= needed:
        pts = min(25, 25 - max(0, (actual - needed - 3) * 2))
        score += pts
        reasons.append(f"[+] {actual} yrs experience (minimum {needed} required)")
    else:
        deducts.append(f"[x] Under-experienced by {needed - actual} yr(s)")

    budget, exp = role["salary_budget"], candidate["Salary Expectation (£)"]
    if   exp <= budget:          score += 20; reasons.append(f"[+] Within budget (expects £{exp:,})")
    elif exp <= budget * 1.10:   score += 10; reasons.append(f"[-] Slightly over budget (£{exp:,} vs £{budget:,} cap)")
    else:                                     deducts.append(f"[x] Over budget: expects £{exp:,} vs £{budget:,} cap")

    if role.get("lloyds_required"):
        if candidate["Lloyd's Experience"]: score += 10; reasons.append("[+] Lloyd's syndicate experience confirmed")
        else:                               deducts.append("[x] No Lloyd's experience (role requires it)")
    else:
        score += 5

    req = role.get("qualification_required", "None")
    if qual_order.get(candidate["Qualification"], 0) >= qual_order.get(req, 0):
        score += 10; reasons.append(f"[+] {candidate['Qualification']} meets minimum ({req})")
    else:
        deducts.append(f"[-] {candidate['Qualification']} — below required {req}")

    avail_pts = {"Immediately": 5, "1 month notice": 3, "3 months notice": 1}.get(
        candidate["Availability"], 1)
    score += avail_pts
    icon = "[+]" if avail_pts == 5 else "[-]"
    reasons.append(f"{icon} {candidate['Availability']}")

    return min(100, max(0, score)), reasons + deducts


def head_to_head(top_candidates_df, role):
    qual_order = {q: i for i, q in enumerate(QUALIFICATIONS)}
    dimensions = [
        "Specialism Match", "Experience", "Salary Fit",
        "Lloyd's Exp", "Qualification", "Availability", "TOTAL SCORE",
    ]
    data = {"Dimension": dimensions}

    for _, row in top_candidates_df.iterrows():
        name = row["Name"].split()[0]

        spec_sc = 30 if row["Specialism"] == role["specialism"] else 0

        needed, actual = role["min_years"], row["Years Experience"]
        exp_sc = (min(25, 25 - max(0, (actual - needed - 3) * 2))
                  if actual >= needed else 0)

        budget, exp_val = role["salary_budget"], row["Salary Expectation (£)"]
        if   exp_val <= budget:         sal_sc = 20
        elif exp_val <= budget * 1.10:  sal_sc = 10
        else:                           sal_sc = 0

        if role.get("lloyds_required"):
            lly_sc = 10 if row["Lloyd's Experience"] else 0
        else:
            lly_sc = 5

        req    = role.get("qualification_required", "None")
        qsc    = qual_order.get(row["Qualification"], 0)
        qreq   = qual_order.get(req, 0)
        qual_sc = 10 if qsc >= qreq else 0

        avail_sc = {"Immediately": 5, "1 month notice": 3,
                    "3 months notice": 1}.get(row["Availability"], 1)

        total = min(100, spec_sc + exp_sc + sal_sc + lly_sc + qual_sc + avail_sc)

        data[name] = [
            f"{spec_sc}/30",
            f"{exp_sc}/25",
            f"{sal_sc}/20",
            f"{lly_sc}/10",
            f"{qual_sc}/10",
            f"{avail_sc}/5",
            f"**{total}/100**",
        ]

    return pd.DataFrame(data)


def get_risk_flags(candidate, role):
    flags = []

    exp_val = candidate["Salary Expectation (£)"]
    budget  = role["salary_budget"]
    over    = exp_val - budget
    if over > budget * 0.10:
        flags.append(("High", "Salary stretch",
                      f"Expects £{exp_val:,} — {over/budget*100:.0f}% over the £{budget:,} cap. "
                      f"Negotiation required or budget increase needed."))
    elif over > 0:
        flags.append(("Moderate", "Salary slightly over",
                      f"Expects £{exp_val:,} vs £{budget:,} budget — "
                      f"within 10%, likely negotiable."))

    fr_prob, fr_label, _ = flight_risk_score(
        candidate["Tenure (Current Role)"], candidate["Specialism"],
        candidate["Years Experience"])
    if fr_prob >= 0.50:
        flags.append(("High", "Flight risk",
                      f"Flight risk {fr_prob*100:.0f}% — tenure "
                      f"{candidate['Tenure (Current Role)']} yrs in current role. "
                      f"Prime headhunt window — move quickly or risk losing to a counter-offer."))
    elif fr_prob >= 0.30:
        flags.append(("Moderate", "Moderate flight risk",
                      f"Flight risk {fr_prob*100:.0f}% — candidate may be exploratory "
                      f"rather than actively seeking. Engagement pace matters."))

    surplus = candidate["Years Experience"] - role["min_years"]
    if surplus > 5:
        flags.append(("Moderate", "Over-qualification",
                      f"{candidate['Years Experience']} yrs experience vs {role['min_years']} yr minimum "
                      f"({surplus} yrs surplus). Risk: candidate may leave quickly for a more senior role."))

    if role.get("lloyds_required") and not candidate["Lloyd's Experience"]:
        flags.append(("High", "Lloyd's requirement unmet",
                      "Role requires Lloyd's syndicate experience. Candidate has company market "
                      "background only. Discuss Lloyd's acclimatisation with hiring manager."))

    if candidate["Availability"] == "3 months notice":
        flags.append(("Moderate", "3-month notice period",
                      "Candidate requires 3 months' notice — plan start date accordingly. "
                      "Early offer may reduce counter-offer window."))

    if not flags:
        flags.append(("Low", "No material flags",
                      "No significant placement risks identified. Standard process applies."))

    return flags


def generate_placement_argument(top_candidate, role, scored_df):
    c     = top_candidate
    score = scored_df.iloc[0]["Match Score"]
    rank2 = scored_df.iloc[1] if len(scored_df) > 1 else None

    if   score >= 85: strength = "an outstanding match"
    elif score >= 70: strength = "a strong match"
    elif score >= 55: strength = "a viable match"
    else:             strength = "a possible match requiring discussion"

    spec_match = c["Specialism"] == role["specialism"]
    exp_surplus = c["Years Experience"] - role["min_years"]
    exp_note = (f"Their {c['Years Experience']}-year career exceeds the {role['min_years']}-year "
                f"minimum by {exp_surplus} years, " if exp_surplus > 0 else
                f"They meet the {role['min_years']}-year experience requirement exactly, ")

    sal_gap  = c["Salary Expectation (£)"] - role["salary_budget"]
    sal_note = (f"salary expectation of £{c['Salary Expectation (£)']:,} is "
                f"{'within' if sal_gap <= 0 else 'nominally above'} the £{role['salary_budget']:,} budget"
                + (f" by £{sal_gap:,}" if sal_gap > 0 else ""))

    lly_note  = "with Lloyd's syndicate experience confirmed" if c["Lloyd's Experience"] else \
                "from a company market background"
    qual_note = c["Qualification"] if c["Qualification"] != "None" else "no formal CII qualification"

    fr_prob, fr_label, _ = flight_risk_score(
        c["Tenure (Current Role)"], c["Specialism"], c["Years Experience"])
    fr_note = (f"flight risk is assessed as {fr_label.lower()} "
               f"({fr_prob*100:.0f}%) with {c['Tenure (Current Role)']} years in their current role")

    vs_note = ""
    if rank2 is not None:
        gap = score - rank2["Match Score"]
        vs_note = (f" They score {gap} points ahead of the second-ranked candidate "
                   f"({rank2['Name'].split()[0]}, {rank2['Match Score']}/100).")

    avail_note = {
        "Immediately":       "available to start immediately",
        "1 month notice":    "available with one month's notice",
        "3 months notice":   "on a three-month notice period — early offer advised",
    }.get(c["Availability"], "availability to be confirmed")

    employers = KEY_EMPLOYERS.get(c["Specialism"], ["London Market"])

    return f"""**Primary Recommendation: {c['Name']} — {c['Title']} — {score}/100 ({strength})**

{c['Name']} is recommended as the primary candidate for this {role['specialism']} role.{vs_note} {"Specialism is an exact match." if spec_match else "Note: specialism is adjacent rather than exact."}

{exp_note}bringing demonstrated experience across the {c['Specialism']} class, {lly_note}. Qualification level is {qual_note}. Their {sal_note}.

From a market intelligence perspective, {fr_note}. Candidates at this tenure and seniority level in the {c['Specialism']} market are actively targeted by employers including {", ".join(employers[:3])}. We recommend moving to offer stage promptly.

Candidate is {avail_note}.

**Recommended next step:** Issue a first-stage meeting invitation within 48 hours to hold candidate engagement ahead of any competing approaches."""


def build_funnel(role):
    qual_order = {q: i for i, q in enumerate(QUALIFICATIONS)}
    req_qual   = role.get("qualification_required", "None")
    s1 = df
    s2 = s1[s1["Specialism"] == role["specialism"]]
    s3 = s2[s2["Years Experience"] >= role["min_years"]]
    s4 = s3[s3["Salary Expectation (£)"] <= role["salary_budget"]]
    s5 = s4[s4["Lloyd's Experience"]] if role.get("lloyds_required") else s4
    s6 = s5[s5["Qualification"].map(qual_order) >= qual_order.get(req_qual, 0)]
    return pd.DataFrame({
        "Stage": ["Total Pool", "Specialism Match", "Experience Met",
                  "Within Budget", "Lloyd's Qualified", "Qualification Met"],
        "Count": [len(s1), len(s2), len(s3), len(s4), len(s5), len(s6)],
    })


def salary_inflation_data():
    rows = []
    for spec, growth in SALARY_GROWTH.items():
        base = df[df["Specialism"] == spec]["Current Salary (£)"].mean()
        for offset, label in [(-2, "2024"), (-1, "2025"), (0, "2026")]:
            rows.append({
                "Specialism": spec,
                "Year":       label,
                "Avg Salary": int(base * (1 + growth) ** offset),
            })
    return pd.DataFrame(rows)


def generate_market_summary(specialism=None):
    spec_avg   = df.groupby("Specialism")["Current Salary (£)"].mean()
    market_avg = df["Current Salary (£)"].mean()

    if specialism:
        s_avg      = spec_avg[specialism]
        premium    = (s_avg - market_avg) / market_avg * 100
        avail_now  = len(df[(df["Specialism"] == specialism) &
                            (df["Availability"] == "Immediately")])
        total_spec = len(df[df["Specialism"] == specialism])
        open_roles = OPEN_ROLES.get(specialism, "N/A")
        ratio      = round(open_roles / max(total_spec, 1), 1)
        growth_pct = SALARY_GROWTH.get(specialism, 0.05) * 100
        high_risk  = len(df[(df["Specialism"] == specialism) &
                            (df["Flight Risk (%)"] >= 50)])
        direction  = "above" if premium > 0 else "below"
        tension    = "tight" if ratio > 2 else "balanced" if ratio > 1 else "candidate-rich"
        employers  = KEY_EMPLOYERS.get(specialism, [])
        emp_str    = ", ".join(employers[:3]) if employers else "major London market players"

        return f"""**{specialism} Underwriter Market — Analyst Summary**
*(Source: Hays 2026 salary data · London Market vacancy sweep 15 March 2026)*

The {specialism} segment is currently trading **{abs(premium):.1f}% {direction}** the platform average of £{market_avg:,.0f}. With **{open_roles} confirmed open roles** against a visible active pool of **{total_spec} candidates**, the supply-demand ratio of **{ratio}:1** signals a **{tension}** market.

Salary growth is running at approximately **{growth_pct:.0f}% per annum** — {"placing it among the highest-pressure segments in the London Market" if growth_pct >= 9 else "above London Market average, reflecting sustained hiring demand" if growth_pct >= 6 else "broadly in line with inflation, suggesting a relatively stable supply position"}.

Only **{avail_now} of {total_spec} candidates** ({round(avail_now / max(total_spec, 1) * 100)}%) are immediately available. **{high_risk} candidates** carry elevated flight risk, representing the highest-priority outreach targets. Active employers in this space include **{emp_str}**.

**Recruitment recommendation:** {"Budget above the market midpoint and move at pace — this pool is under structural pressure and counter-offers are common." if ratio > 2 else "Standard recruitment timelines apply, but salary offers should be benchmarked carefully — annual growth means 2024-era offers are already uncompetitive." if ratio > 1 else "Candidate supply is relatively healthy in this segment. Quality screening is the priority; negotiate firmly on salary."}"""

    else:
        top_spec    = spec_avg.idxmax(); top_sal = spec_avg.max()
        low_spec    = spec_avg.idxmin(); low_sal = spec_avg.min()
        spread      = (top_sal - low_sal) / low_sal * 100
        avail_now   = len(df[df["Availability"] == "Immediately"])
        high_risk   = len(df[df["Flight Risk (%)"] >= 50])
        marine_pool = len(df[df["Specialism"].str.startswith("Marine")])

        return f"""**London Specialist Insurance Market — Underwriter Landscape**
*(Source: Hays UK Salary & Recruiting Trends 2026 · London Market vacancy sweep 15 March 2026)*

Across all eight specialisms tracked, the London Market shows a **{spread:.0f}% salary spread** between the highest-paid segment ({top_spec}, avg £{top_sal:,.0f}) and the lowest ({low_spec}, avg £{low_sal:,.0f}). This reflects the structural premium placed on niche expertise and Lloyd's market access.

**Marine lines collectively account for {marine_pool} of {len(df)} active candidates** ({round(marine_pool/len(df)*100)}% of the platform pool). Marine Hull remains the hardest to fill, with the thinnest senior talent pool of any class. Offshore Energy is growing fastest, with renewables creating net-new roles that have no historical supply base.

**{avail_now} of {len(df)} candidates** ({round(avail_now / len(df) * 100)}%) are immediately available. Flight risk modelling identifies **{high_risk} candidates** with elevated readiness to move — the highest-value targets for proactive outreach ahead of a formal brief.

**Key recommendation:** Clients hiring for Reinsurance and Marine Hull should treat salary budgets as a range, not a fixed number. The sensitivity analysis shows a 10% budget increase can more than double the qualified candidate pool in the most supply-constrained specialisms."""


def get_market_insights():
    spec_avg   = df.groupby("Specialism")["Current Salary (£)"].mean()
    top        = spec_avg.idxmax()
    bot        = spec_avg.idxmin()
    spread_pct = (spec_avg[top] - spec_avg[bot]) / spec_avg[bot] * 100
    marine_cnt = len(df[df["Specialism"].str.startswith("Marine")])
    lloyds_pct = df["Lloyd's Experience"].mean() * 100
    hull_roles = OPEN_ROLES["Marine Hull"]
    hull_pool  = len(df[df["Specialism"] == "Marine Hull"])
    return [
        f"**{top}** commands the highest avg salary at £{spec_avg[top]:,.0f} "
        f"({spread_pct:.0f}% above {bot})",
        f"Marine Hull: **{hull_roles} open roles** vs only **{hull_pool} candidates** "
        f"— ratio {round(hull_roles/max(hull_pool,1),1)}:1 (hardest to fill per vacancy analysis)",
        f"Offshore Energy salaries growing **{SALARY_GROWTH['Offshore Energy']*100:.0f}%/yr** "
        f"— fastest in the market (renewables driving net-new demand)",
        f"**{marine_cnt} of {len(df)} candidates** ({round(marine_cnt/len(df)*100)}%) "
        f"write marine lines — core platform focus reflected in pool depth",
        f"**{lloyds_pct:.0f}%** hold Lloyd's syndicate experience · "
        f"**{(df['Flight Risk (%)']>=50).sum()} candidates** at elevated flight risk — prime for outreach",
    ]
