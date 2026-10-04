import math
from collections import Counter
from typing import Any, Dict, List, Optional


def _fmt(v: Optional[float], unit: str = "", n: int = 0) -> str:
    if v is None:
        return "—"
    if n == 0:
        return f"{v:.0f}{unit}"
    return f"{v:.{n}f}{unit}"


def _short_source(s: str, n: int = 28) -> str:
    if not s:
        return "—"
    s = s.rsplit("/", 1)[-1]
    if s.lower().endswith(".md") or s.lower().endswith(".pdf"):
        s = s.rsplit(".", 1)[0]
    return s if len(s) <= n else s[: n - 1] + "…"


def format_query_line(query: Dict[str, Any]) -> str:
    mat = query.get("material", "unknown")
    P = query.get("Power")
    v = query.get("Velocity")
    h = query.get("Hatch spacing")
    t = query.get("layer thickness")
    bD = query.get("beam D")
    ved = query.get("VED_Jmm3")
    if ved is None:
        try:
            denom = v * (h * 1e-3) * (t * 1e-3)
            ved = P / denom if denom > 0 else None
        except (TypeError, ZeroDivisionError):
            ved = None
    parts = [
        f"material={mat}",
        f"P={_fmt(P, ' W')}",
        f"v={_fmt(v, ' mm/s')}",
        f"h={_fmt(h, ' µm')}",
        f"t={_fmt(t, ' µm')}",
        f"beam D={_fmt(bD, ' µm')}",
        f"derived VED={_fmt(ved, ' J/mm³', 1)}",
    ]
    return "Query: " + ", ".join(parts) + "."


def _window_summary(rows: List[Dict[str, Any]],
                   query: Optional[Dict[str, Any]] = None) -> str:
    if not rows:
        return ""
    veds = [r.get("VED_Jmm3") or r.get("audit", {}).get("r_VED")
            for r in rows]
    veds = [v for v in veds if isinstance(v, (int, float)) and v > 0]
    densities = [r.get("density_pct") for r in rows
                 if isinstance(r.get("density_pct"), (int, float))]
    defects = [r.get("defect") for r in rows if r.get("defect")]

    parts: List[str] = []
    if veds:
        parts.append(f"VED range {min(veds):.0f}–{max(veds):.0f} J/mm³")
    if densities:
        n_high = sum(1 for d in densities if d >= 99.0)
        parts.append(f"{n_high}/{len(densities)} rows report density ≥ 99%")
    if defects:
        dist = Counter(defects).most_common()
        parts.append("defect outcomes: " +
                    ", ".join(f"{d}={n}" for d, n in dist))

    q_ved = None
    if query is not None:
        q_ved = query.get("VED_Jmm3")
        if q_ved is None:
            P = query.get("Power"); v = query.get("Velocity")
            h = query.get("Hatch spacing"); t = query.get("layer thickness")
            try:
                denom = v * (h * 1e-3) * (t * 1e-3)
                q_ved = P / denom if denom > 0 else None
            except (TypeError, ZeroDivisionError):
                q_ved = None
    if q_ved and q_ved > 0:
        n_close = n_close_good = n_close_def = 0
        for r in rows:
            rved = r.get("VED_Jmm3") or r.get("audit", {}).get("r_VED")
            if not isinstance(rved, (int, float)) or rved <= 0:
                continue
            ratio = max(rved, q_ved) / min(rved, q_ved)
            if ratio > 1.3:
                continue
            n_close += 1
            d = r.get("defect")
            if d == "none":
                n_close_good += 1
            elif d in ("lof", "balling", "keyhole", "porosity"):
                n_close_def += 1
        if n_close:
            parts.append(
                f"within ±30% of query VED ({q_ved:.0f}): {n_close_good} 'none', "
                f"{n_close_def} defect, {n_close - n_close_good - n_close_def} unlabeled"
            )

    if not parts:
        return ""
    return "Window summary: " + "; ".join(parts) + "."


def _direct_match_note(rows: List[Dict[str, Any]],
                      query: Optional[Dict[str, Any]] = None) -> str:
    """Flag measured density only when all four process parameters match."""
    if not rows or query is None:
        return ""

    def close(measured: Any, target: Any) -> bool:
        try:
            measured, target = float(measured), float(target)
        except (TypeError, ValueError):
            return False
        return (math.isfinite(measured) and math.isfinite(target)
                and measured > 0 and target > 0
                and abs(measured - target) / target <= 0.15)

    dimensions = (("P_W", "Power"), ("v_mms", "Velocity"),
                  ("h_um", "Hatch spacing"),
                  ("t_um", "layer thickness"))
    notes_good: List[str] = []
    notes_def:  List[str] = []
    for r in rows:
        if not all(close(r.get(measured), query.get(target))
                   for measured, target in dimensions):
            continue
        density = r.get("density_pct")
        if not isinstance(density, (int, float)) or not math.isfinite(density):
            continue
        note = f"#{r.get('rank', '?')} ({density:.1f}%)"
        if density >= 99.0:
            notes_good.append(note)
        else:
            notes_def.append(note)
    if not (notes_good or notes_def):
        return ""
    bits = []
    if notes_good:
        bits.append(f"row(s) {', '.join(notes_good)} support good")
    if notes_def:
        bits.append(f"row(s) {', '.join(notes_def)} support defective")
    return "Direct density matches (P, v, h, t each within ±15%): " + "; ".join(bits) + "."


def _row_to_table_cells(row: Dict[str, Any], rank: int) -> List[str]:
    return [
        str(rank),
        _fmt(row.get("P_W"),       n=0),
        _fmt(row.get("v_mms"),     n=0),
        _fmt(row.get("h_um"),      n=0),
        _fmt(row.get("t_um"),      n=0),
        _fmt(row.get("VED_Jmm3"),  n=1) if row.get("VED_Jmm3") is not None
            else _fmt(row.get("audit", {}).get("r_VED"), n=1),
        (row.get("defect") or "—"),
        _fmt(row.get("density_pct"), unit="%", n=1),
        (row.get("evidence_type") or row.get("claim_strength") or "—"),
        _short_source(row.get("source_file", "")),
    ]


_TABLE_HEADER = ["#", "P (W)", "v (mm/s)", "h (µm)", "t (µm)",
                "VED", "defect", "density", "type", "source"]


def build_evidence_pack(rows: List[Dict[str, Any]],
                       query: Dict[str, Any],
                       include_snippets: bool = True,
                       max_snippet_chars: int = 220,
                       ) -> str:
    """Render rows + query as a markdown evidence pack.

    Returns a self-contained string suitable for inlining under a
    'Retrieved Literature Evidence:' header in the KD-agent prompt.
    """
    if not rows:
        return (
            format_query_line(query) + "\n"
            "No structured evidence rows were retrieved for this material.\n"
            "Rely on internal physics knowledge for this prediction."
        )

    lines: List[str] = [format_query_line(query), ""]

    cell_rows = [_row_to_table_cells(r, r.get("rank", i + 1))
                 for i, r in enumerate(rows)]
    widths = [max(len(h), *(len(row[i]) for row in cell_rows))
              for i, h in enumerate(_TABLE_HEADER)]

    def _fmt_line(cells: List[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    lines.append(_fmt_line(_TABLE_HEADER))
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for c in cell_rows:
        lines.append(_fmt_line(c))

    summary = _window_summary(rows, query)
    if summary:
        lines.append("")
        lines.append(summary)

    direct = _direct_match_note(rows, query)
    if direct:
        lines.append(direct)

    if include_snippets:
        lines.append("")
        lines.append("Snippets:")
        for r in rows:
            rank = r.get("rank", "?")
            src = _short_source(r.get("source_file", ""))
            snippet = (r.get("snippet") or "").strip().replace("\n", " ")
            if len(snippet) > max_snippet_chars:
                snippet = snippet[: max_snippet_chars - 1] + "…"
            lines.append(f"[{rank}] ({src}) {snippet}")

    return "\n".join(lines)


def rows_to_numbered_list(rows: List[Dict[str, Any]]) -> List[str]:
    """Backward-compat path: render each row as a single bullet for legacy
    prompt templates that still expect `List[str]` of plain text snippets."""
    out: List[str] = []
    for i, r in enumerate(rows, 1):
        bits = []
        if r.get("P_W"):     bits.append(f"P={r['P_W']:.0f} W")
        if r.get("v_mms"):   bits.append(f"v={r['v_mms']:.0f} mm/s")
        if r.get("h_um"):    bits.append(f"h={r['h_um']:.0f} µm")
        if r.get("t_um"):    bits.append(f"t={r['t_um']:.0f} µm")
        ved = r.get("VED_Jmm3") or r.get("audit", {}).get("r_VED")
        if ved:              bits.append(f"VED={ved:.1f} J/mm³")
        if r.get("defect"):  bits.append(f"defect={r['defect']}")
        if r.get("density_pct") is not None:
            bits.append(f"density={r['density_pct']:.1f}%")
        params = ", ".join(bits) if bits else "—"
        snippet = (r.get("snippet") or "").strip()
        src = _short_source(r.get("source_file", ""))
        out.append(f"({params}) — \"{snippet}\" [{src}]")
    return out


__all__ = [
    "build_evidence_pack", "rows_to_numbered_list",
    "format_query_line",
]
