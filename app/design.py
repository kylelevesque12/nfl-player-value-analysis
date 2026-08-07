"""The app's design system: one set of tokens, one stylesheet, one component kit.

Everything visual in this app comes from here. That is the point — before this
module the styling lived in three separate injectors that each overrode the one
before it (the base layer painted the sidebar white, the brand layer repainted
it navy), so changing a color meant finding which layer won.

The rules:

1. **`inject_design_system` is the only function in the app that emits CSS.**
   A section module that needs a new look adds a component here rather than a
   ``<style>`` block of its own. `tests/test_design_system.py` enforces this.
2. **Components are built from tokens, never from raw values.** If a component
   needs a color or a size that is not a token, the token set is missing
   something — add it rather than hard-coding a hex.
3. **Semantic color, not decorative color.** `--positive` / `--negative` /
   `--uncertain` say what a number *means*. Uncertainty is deliberately its own
   role, because a range that the model is unsure about must not read like a
   neutral grey afterthought — this app's whole claim is that it tells you what
   it does not know.

The token values are one place, so a future restyle is an edit to `_TOKENS`
rather than a search across the app.
"""

from __future__ import annotations

import html as _html

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
# Type and spacing both use a modest ratio: this is a dense data product, and a
# dramatic scale wastes vertical space that the board needs for rows.
_TOKENS = """
:root {
  /* Surfaces and ink */
  --surface:          #f6f8fb;
  --surface-raised:   #ffffff;
  --surface-sunken:   #eef3f9;
  --border:           #dfe7ef;
  --border-strong:    #cdd9e8;
  --ink:              #16263a;
  --ink-muted:        #5b6b7c;
  --ink-subtle:       #8a97a5;
  --ink-inverse:      #eaf1f8;

  /* Brand */
  --navy:             #0d2b45;
  --blue:             #1565c0;
  --sky:              #4a90d9;
  --tint:             #eef3f9;

  /* Semantic roles: what a number MEANS, not how it looks. */
  --positive:         #157a6e;
  --positive-tint:    #e6f2f0;
  --negative:         #c8553d;
  --negative-tint:    #fbece9;
  --uncertain:        #b08900;
  --uncertain-tint:   #fbf3d9;
  --neutral:          #5b6b7c;

  /* Type scale */
  --text-xs:          0.72rem;
  --text-sm:          0.82rem;
  --text-base:        0.94rem;
  --text-md:          1.05rem;
  --text-lg:          1.30rem;
  --text-xl:          1.70rem;
  --text-2xl:         2.10rem;

  /* Weights */
  --weight-normal:    500;
  --weight-medium:    600;
  --weight-bold:      700;
  --weight-heavy:     800;

  /* Spacing scale */
  --space-1:          0.25rem;
  --space-2:          0.45rem;
  --space-3:          0.70rem;
  --space-4:          1.00rem;
  --space-5:          1.40rem;
  --space-6:          2.00rem;

  /* Shape */
  --radius-sm:        8px;
  --radius-md:        12px;
  --radius-lg:        16px;
  --radius-pill:      999px;

  /* Elevation */
  --shadow-sm:        0 1px 3px rgba(13, 43, 69, 0.06);
  --shadow-md:        0 4px 14px rgba(13, 43, 69, 0.10);
  --shadow-lg:        0 6px 20px rgba(13, 43, 69, 0.18);
}
"""

# ---------------------------------------------------------------------------
# Component stylesheet
# ---------------------------------------------------------------------------
_COMPONENTS = """
/* --- Page shell ------------------------------------------------------ */
.stApp { background: var(--surface); }
.block-container { padding-top: var(--space-6); padding-bottom: var(--space-6); }
.main a { color: var(--blue); }
.main h1 { color: var(--navy); font-weight: var(--weight-heavy); letter-spacing: -0.01em; }
.main h2 { color: var(--navy); font-weight: var(--weight-bold); font-size: var(--text-lg); }
.main h3 { color: var(--navy); font-weight: var(--weight-bold); font-size: var(--text-md); }

/* --- Hero ------------------------------------------------------------ */
.ds-hero {
  background: linear-gradient(120deg, var(--navy) 0%, var(--blue) 70%, #2f7fd1 100%);
  color: #fff;
  padding: var(--space-5) var(--space-5);
  border-radius: var(--radius-lg);
  margin-bottom: var(--space-4);
  box-shadow: var(--shadow-lg);
}
/* The banner title stays a real <h1> for screen readers, which means undoing
   two things Streamlit does to every heading: a generated
   `.st-emotion-cache-* h1` padding rule that outspecifies a single class (so
   this selector carries two), and an injected anchor-link element that adds a
   second line box inside the heading. */
.ds-hero h1.ds-hero__title {
  font-size: var(--text-2xl); font-weight: var(--weight-heavy);
  line-height: 1.15; margin: 0; padding: 0; color: #fff;
}
.ds-hero [data-testid="stHeaderActionElements"] { display: none; }
.ds-hero__sub {
  color: #dbe8f6; font-size: var(--text-md); margin: var(--space-2) 0 0;
  max-width: 62ch;
}
.ds-pill {
  display: inline-block; background: rgba(255,255,255,0.16); color: #fff;
  border-radius: var(--radius-pill); padding: var(--space-1) var(--space-3);
  font-size: var(--text-sm); font-weight: var(--weight-medium);
  margin: var(--space-3) var(--space-2) 0 0;
}

/* --- Section header -------------------------------------------------- */
.ds-section {
  border-left: 5px solid var(--blue);
  padding: var(--space-1) 0 var(--space-1) var(--space-3);
  margin: var(--space-1) 0 var(--space-4);
}
.ds-section__eyebrow {
  text-transform: uppercase; letter-spacing: 0.08em;
  font-size: var(--text-xs); font-weight: var(--weight-bold); color: var(--sky);
}
.ds-section__title {
  font-size: var(--text-xl); font-weight: var(--weight-heavy);
  color: var(--navy); line-height: 1.15;
}
.ds-section__sub { color: var(--ink-muted); font-size: var(--text-md); margin-top: var(--space-1); }

/* --- Stat card ------------------------------------------------------- */
.ds-stat-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: var(--space-3); margin: var(--space-2) 0 var(--space-4);
}
.ds-stat {
  background: var(--surface-raised); border: 1px solid var(--border);
  border-left: 4px solid var(--blue); border-radius: var(--radius-sm);
  padding: var(--space-3) var(--space-4); box-shadow: var(--shadow-sm);
}
.ds-stat--positive { border-left-color: var(--positive); }
.ds-stat--negative { border-left-color: var(--negative); }
.ds-stat--uncertain { border-left-color: var(--uncertain); }
.ds-stat__label {
  font-size: var(--text-xs); font-weight: var(--weight-bold); color: var(--ink-muted);
  text-transform: uppercase; letter-spacing: 0.05em;
}
.ds-stat__value {
  font-size: var(--text-lg); font-weight: var(--weight-heavy); color: var(--ink);
  line-height: 1.2; margin-top: var(--space-1);
}
.ds-stat__value .ds-stat__unit {
  font-size: var(--text-xs); font-weight: var(--weight-medium);
  color: var(--ink-subtle); margin-left: var(--space-1);
}
.ds-stat__note { font-size: var(--text-xs); color: var(--ink-subtle); margin-top: var(--space-1); }

/* Streamlit columns do not stretch their children by default, so cards in a
   row sit at different heights. Stretch them once, here, rather than in each
   page that happens to lay cards out in columns. */
[data-testid="stColumn"] > div,
[data-testid="stColumn"] [data-testid="stVerticalBlock"] { height: 100%; }

/* --- Player tile (team-colored, used for the top-projected strip) ---- */
.ds-tile {
  background: var(--surface-raised); border: 1px solid var(--border);
  border-top: 4px solid var(--team-color, var(--navy));
  border-radius: var(--radius-md);
  padding: var(--space-3) var(--space-3) var(--space-2);
  box-shadow: var(--shadow-sm); height: 100%;
}
.ds-tile__rank {
  font-size: var(--text-xs); font-weight: var(--weight-bold);
  color: var(--ink-subtle); letter-spacing: 0.05em;
}
.ds-tile__name {
  font-size: var(--text-base); font-weight: var(--weight-heavy); color: var(--navy);
  line-height: 1.2; margin: var(--space-1) 0;
}
.ds-tile__meta {
  font-size: var(--text-xs); font-weight: var(--weight-medium); color: var(--ink-muted);
  text-transform: uppercase; letter-spacing: 0.03em;
}
.ds-tile__value {
  font-size: var(--text-lg); font-weight: var(--weight-heavy); color: var(--blue);
  margin-top: var(--space-2); line-height: 1;
}
.ds-tile__value span {
  font-size: var(--text-xs); font-weight: var(--weight-medium);
  color: var(--ink-subtle); margin-left: var(--space-1);
}

/* --- Player row (compact ranked list) -------------------------------- */
.ds-row {
  display: flex; align-items: center; gap: var(--space-3);
  background: var(--surface-raised); border: 1px solid var(--border);
  border-radius: var(--radius-sm); padding: var(--space-2) var(--space-3);
  margin-bottom: var(--space-2);
}
.ds-row__rank {
  font-size: var(--text-sm); font-weight: var(--weight-bold); color: var(--ink-subtle);
  min-width: 2.2rem; text-align: right; flex: none;
}
.ds-row__main { flex: 1 1 auto; min-width: 0; }
.ds-row__name {
  font-size: var(--text-base); font-weight: var(--weight-bold); color: var(--ink);
  line-height: 1.25; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.ds-row__meta { font-size: var(--text-xs); color: var(--ink-muted); margin-top: 1px; }
.ds-row__value {
  font-size: var(--text-md); font-weight: var(--weight-heavy); color: var(--blue);
  flex: none; text-align: right;
}
.ds-row__value small {
  display: block; font-size: var(--text-xs); font-weight: var(--weight-normal);
  color: var(--ink-subtle);
}

/* --- Tier badge ------------------------------------------------------ */
.ds-tier {
  display: inline-block; border-radius: var(--radius-pill);
  padding: 1px var(--space-2); font-size: var(--text-xs);
  font-weight: var(--weight-bold); letter-spacing: 0.02em;
  border: 1px solid transparent; white-space: nowrap;
}
.ds-tier--1 { background: var(--positive-tint); color: var(--positive); border-color: var(--positive); }
.ds-tier--2 { background: var(--tint); color: var(--blue); border-color: var(--sky); }
.ds-tier--3 { background: var(--surface-sunken); color: var(--ink-muted); border-color: var(--border-strong); }
.ds-tier--4 { background: var(--surface); color: var(--ink-subtle); border-color: var(--border); }

/* --- Range bar (floor / projection / ceiling) ------------------------ */
/* The uncertainty band is the visual subject here, not decoration: the track
   is the 80% interval and the marker is the point estimate sitting inside it. */
.ds-range { min-width: 90px; }
.ds-range__track {
  position: relative; height: 7px; border-radius: var(--radius-pill);
  background: var(--uncertain-tint); border: 1px solid var(--uncertain);
  overflow: visible;
}
.ds-range__marker {
  position: absolute; top: 50%; width: 3px; height: 15px;
  background: var(--blue); border-radius: 2px; transform: translate(-50%, -50%);
}
.ds-range__labels {
  display: flex; justify-content: space-between;
  font-size: var(--text-xs); color: var(--ink-subtle); margin-top: 2px;
}

/* --- Delta (positive / negative / flat) ------------------------------ */
.ds-delta { font-weight: var(--weight-bold); font-size: var(--text-sm); }
.ds-delta--up { color: var(--positive); }
.ds-delta--down { color: var(--negative); }
.ds-delta--flat { color: var(--neutral); }

/* --- Callout --------------------------------------------------------- */
.ds-callout {
  display: block; border-radius: var(--radius-sm);
  padding: var(--space-3) var(--space-4); margin: var(--space-3) 0;
  font-size: var(--text-base); line-height: 1.5;
  border: 1px solid var(--border); border-left: 4px solid var(--neutral);
  background: var(--surface-raised); color: var(--ink);
}
.ds-callout--caveat { border-left-color: var(--uncertain); background: var(--uncertain-tint); }
.ds-callout--positive { border-left-color: var(--positive); background: var(--positive-tint); }
.ds-callout--negative { border-left-color: var(--negative); background: var(--negative-tint); }
.ds-callout--info { border-left-color: var(--blue); background: var(--tint); }
.ds-callout__label {
  display: inline-block; font-size: var(--text-xs); font-weight: var(--weight-bold);
  text-transform: uppercase; letter-spacing: 0.06em; margin-right: var(--space-2);
  color: var(--ink-muted);
}

/* --- Streamlit element overrides ------------------------------------- */
div[data-testid="stMetric"] {
  background: var(--surface-raised); border: 1px solid var(--border);
  border-left: 4px solid var(--blue); border-radius: var(--radius-sm);
  padding: var(--space-3) var(--space-4); box-shadow: var(--shadow-sm);
}
[data-testid="stMetricLabel"] p { color: var(--ink-muted); font-size: var(--text-sm); }
[data-testid="stMetricValue"] { color: var(--ink); }

div[data-testid="stDataFrame"] {
  border: 1px solid var(--border); border-radius: var(--radius-sm);
  overflow: hidden; background: var(--surface-raised);
}
.stDataFrame thead tr th { background: var(--tint) !important; }

div[data-testid="stExpander"] {
  background: var(--surface-raised); border: 1px solid var(--border);
  border-radius: var(--radius-sm);
}
[data-testid="stExpander"] summary { font-weight: var(--weight-medium); }

.stButton > button {
  border-radius: var(--radius-sm); border: 1px solid var(--border-strong);
  font-weight: var(--weight-medium);
}
.stButton > button:hover { border-color: var(--blue); color: var(--blue); }

div[data-testid="stAlert"] { border-radius: var(--radius-sm); }

/* --- Sidebar --------------------------------------------------------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--navy) 0%, #143a5e 100%);
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: var(--ink-inverse) !important; }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
  color: #b9cbe0 !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
  background: var(--surface-raised); color: var(--ink);
}
section[data-testid="stSidebar"] div[data-baseweb="select"] * { color: var(--ink); }
section[data-testid="stSidebar"] code {
  background: rgba(255,255,255,0.12); color: var(--ink-inverse);
}
.ds-brand {
  font-size: var(--text-md); font-weight: var(--weight-heavy); color: #fff;
  letter-spacing: -0.01em; padding: var(--space-2) 0 var(--space-1);
  line-height: 1.25; white-space: nowrap;
}
.ds-brand__sub {
  display: block; font-size: var(--text-xs); font-weight: var(--weight-normal);
  color: #b9cbe0; letter-spacing: 0.02em; margin-top: var(--space-1);
}
/* Nav: hide the radio circles so the options read as full-width rows. */
section[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }
section[data-testid="stSidebar"] [role="radiogroup"] label {
  width: 100%; margin: 2px 0; padding: var(--space-2) var(--space-3);
  border-radius: var(--radius-sm); cursor: pointer; transition: background 0.15s ease;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: rgba(255,255,255,0.15); box-shadow: inset 3px 0 0 var(--sky);
}
section[data-testid="stSidebar"] [role="radiogroup"] label p {
  font-size: var(--text-base); font-weight: var(--weight-normal);
}

/* --- Narrow viewports ------------------------------------------------ */
/* Goal 8 does the real phone pass; this keeps the type scale from
   overflowing in the meantime. */
/* Tiles sit side by side only on wide screens, and only there does a
   two-line name (Jaxon Smith-Njigba) knock its tile out of alignment with the
   rest of the row. Reserving the second line below this width would just add
   dead space to a stack of full-width cards. */
@media (min-width: 901px) {
  .ds-tile__name { min-height: 2.4em; }
}

@media (max-width: 900px) {
  .block-container { padding-left: var(--space-4); padding-right: var(--space-4); }
  /* Matches the specificity of the base rule above, which carries two classes
     to beat Streamlit's own generated h1 styling — a single-class selector
     here would be silently overridden. */
  .ds-hero h1.ds-hero__title { font-size: var(--text-xl); }
  .ds-section__title { font-size: var(--text-lg); }
}
"""


def inject_design_system() -> None:
    """Inject the tokens and the component stylesheet. The app's only CSS."""
    st.markdown(f"<style>{_TOKENS}{_COMPONENTS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def _esc(value) -> str:
    """Escape interpolated text. Every component below renders raw HTML, so
    anything derived from data (a player name, a team code) goes through here —
    a stray ``&`` or ``<`` in a name would otherwise break the markup."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return _html.escape(str(value))


def hero(title: str, subtitle: str = "", pills: list[str] | None = None) -> None:
    """The page-top banner. Used once, on Home."""
    pill_html = "".join(f'<span class="ds-pill">{_esc(p)}</span>' for p in (pills or []))
    st.markdown(
        f'<div class="ds-hero">'
        f'<h1 class="ds-hero__title">{_esc(title)}</h1>'
        f'<p class="ds-hero__sub">{_esc(subtitle)}</p>'
        f"{pill_html}</div>",
        unsafe_allow_html=True,
    )


def section_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    """Every section's top: a small colored eyebrow, a title, a one-line sub."""
    sub = f'<div class="ds-section__sub">{_esc(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="ds-section">'
        f'<div class="ds-section__eyebrow">{_esc(eyebrow)}</div>'
        f'<div class="ds-section__title">{_esc(title)}</div>'
        f"{sub}</div>",
        unsafe_allow_html=True,
    )


def brand_block(title: str, subtitle: str) -> None:
    """The sidebar wordmark."""
    st.sidebar.markdown(
        f'<div class="ds-brand">{_esc(title)}'
        f'<span class="ds-brand__sub">{_esc(subtitle)}</span></div>',
        unsafe_allow_html=True,
    )


def stat_card_html(
    label: str,
    value: str,
    unit: str = "",
    note: str = "",
    tone: str = "",
) -> str:
    """One stat. ``tone`` is a semantic role: positive / negative / uncertain."""
    modifier = f" ds-stat--{tone}" if tone in {"positive", "negative", "uncertain"} else ""
    unit_html = f'<span class="ds-stat__unit">{_esc(unit)}</span>' if unit else ""
    note_html = f'<div class="ds-stat__note">{_esc(note)}</div>' if note else ""
    return (
        f'<div class="ds-stat{modifier}">'
        f'<div class="ds-stat__label">{_esc(label)}</div>'
        f'<div class="ds-stat__value">{_esc(value)}{unit_html}</div>'
        f"{note_html}</div>"
    )


def stat_grid(cards: list[str]) -> None:
    """Lay out stat cards in a responsive grid that wraps on narrow screens."""
    if not cards:
        return
    st.markdown(
        f'<div class="ds-stat-grid">{"".join(cards)}</div>', unsafe_allow_html=True
    )


def player_tile_html(
    rank: int | str,
    name: str,
    position: str,
    team: str,
    value: str,
    value_unit: str = "proj PPR",
    meta: str = "",
    team_color: str = "",
) -> str:
    """A team-colored tile for a featured player."""
    color = f" style=\"--team-color:{_esc(team_color)}\"" if team_color else ""
    meta_line = " · ".join(x for x in [_esc(team), _esc(meta)] if x)
    return (
        f'<div class="ds-tile"{color}>'
        f'<div class="ds-tile__rank">#{_esc(rank)} · {_esc(position)}</div>'
        f'<div class="ds-tile__name">{_esc(name)}</div>'
        f'<div class="ds-tile__meta">{meta_line}</div>'
        f'<div class="ds-tile__value">{_esc(value)}<span>{_esc(value_unit)}</span></div>'
        f"</div>"
    )


def player_row_html(
    rank: int | str,
    name: str,
    meta: str,
    value: str,
    value_note: str = "",
    badge: str = "",
) -> str:
    """A compact ranked row. ``badge`` takes raw component HTML (a tier badge)."""
    note = f"<small>{_esc(value_note)}</small>" if value_note else ""
    return (
        f'<div class="ds-row">'
        f'<div class="ds-row__rank">{_esc(rank)}</div>'
        f'<div class="ds-row__main">'
        f'<div class="ds-row__name">{_esc(name)} {badge}</div>'
        f'<div class="ds-row__meta">{_esc(meta)}</div>'
        f"</div>"
        f'<div class="ds-row__value">{_esc(value)}{note}</div>'
        f"</div>"
    )


def player_list(rows: list[str]) -> None:
    if rows:
        st.markdown("".join(rows), unsafe_allow_html=True)


def tier_badge_html(tier: int | float | None, label: str = "") -> str:
    """Tier 1 is the strongest group. Tiers past 4 share the quietest style —
    the distinction stops being meaningful that far down the board."""
    if tier is None or pd.isna(tier):
        return ""
    step = min(max(int(tier), 1), 4)
    text = label or f"Tier {int(tier)}"
    return f'<span class="ds-tier ds-tier--{step}">{_esc(text)}</span>'


def range_bar_html(
    low: float | None,
    point: float | None,
    high: float | None,
    show_labels: bool = True,
) -> str:
    """The 80% interval as a band with the point estimate marked inside it.

    Returns "" unless the range is usable, so a missing interval shows nothing
    rather than a misleading full-width bar.
    """
    if any(v is None or pd.isna(v) for v in (low, point, high)):
        return ""
    low, point, high = float(low), float(point), float(high)
    if high <= low:
        return ""
    pct = max(0.0, min(100.0, (point - low) / (high - low) * 100.0))
    labels = (
        f'<div class="ds-range__labels"><span>{low:.0f}</span>'
        f"<span>{high:.0f}</span></div>"
        if show_labels
        else ""
    )
    return (
        f'<div class="ds-range"><div class="ds-range__track">'
        f'<div class="ds-range__marker" style="left:{pct:.1f}%"></div>'
        f"</div>{labels}</div>"
    )


def delta_html(change: float | None, threshold: float = 0.0, fmt: str = "{:+.0f}") -> str:
    """A signed change, colored by the semantic roles rather than a raw hex."""
    if change is None or pd.isna(change):
        return ""
    change = float(change)
    if change > threshold:
        tone, arrow = "up", "▲"
    elif change < -threshold:
        tone, arrow = "down", "▼"
    else:
        tone, arrow = "flat", "—"
    return f'<span class="ds-delta ds-delta--{tone}">{arrow} {fmt.format(change)}</span>'


def callout(body: str, label: str = "", tone: str = "info") -> None:
    """A visible note. ``caveat`` is the tone for "here is what we do not know",
    which is product content in this app, not a disclaimer to be hidden."""
    modifier = (
        f" ds-callout--{tone}"
        if tone in {"caveat", "positive", "negative", "info"}
        else ""
    )
    label_html = f'<span class="ds-callout__label">{_esc(label)}</span>' if label else ""
    st.markdown(
        f'<div class="ds-callout{modifier}">{label_html}{_esc(body)}</div>',
        unsafe_allow_html=True,
    )


def caveat_callout(body: str, label: str = "Caveat") -> None:
    """A limitation the reader needs to see. Kept as its own named component so
    that every caveat in the app looks the same and none can be quietly
    downgraded to body text."""
    callout(body, label, tone="caveat")


def source_footer(text: str) -> None:
    st.divider()
    st.caption(text)
