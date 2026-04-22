r"""Clean pandoc-generated LaTeX body for robust pdflatex compilation.

All LaTeX commands are written with raw strings (r"...") so Python does
not interpret escape sequences like \f, \t, \n as control characters.
"""
from pathlib import Path

BODY = Path(r"D:\coding\repositorio-git\Wraith\paper\latex\wraith_paper_body.tex")

text = BODY.read_text(encoding="utf-8")

# ─── 1. Strip title/author front matter ──────────────────────────────
HR = r"\begin{center}\rule{0.5\linewidth}{0.5pt}\end{center}"
idx = text.find(HR)
if idx >= 0:
    text = text[idx + len(HR):].lstrip()

# ─── 2. Convert Abstract subsection to abstract environment ─────────
ABSTRACT_MARKER = r"\subsection{Abstract}\label{abstract}"
if text.startswith(ABSTRACT_MARKER):
    text = text[len(ABSTRACT_MARKER):].lstrip()
    end = text.find(HR)
    if end > 0:
        abstract_body = text[:end].rstrip()
        rest = text[end + len(HR):].lstrip()
        text = r"\begin{abstract}" + "\n" + abstract_body + "\n" + r"\end{abstract}" + "\n\n" + rest

# ─── 3. Simplify longtable column specs ─────────────────────────────
#   2-3 cols:  use  l   (auto-width, short content)
#   4+  cols:  use  p{W}  (forced width, auto-wrap long cells)
def simplify_longtables(s):
    out, i = [], 0
    marker = r"\begin{longtable}[]{@{}"
    while i < len(s):
        j = s.find(marker, i)
        if j < 0:
            out.append(s[i:])
            break
        out.append(s[i:j])
        k = s.find(r"@{}}", j + len(marker))
        if k < 0:
            out.append(s[j:])
            break
        block = s[j:k + 4]
        ncols = block.count(r">{") or 2
        if ncols == 1:
            cols = "l"
        elif ncols == 2:
            # 2-col: first col narrower (labels), second wider (descriptions/citations)
            cell1 = r">{\raggedright\arraybackslash}p{0.33\linewidth}"
            cell2 = r">{\raggedright\arraybackslash}p{0.60\linewidth}"
            cols = cell1 + cell2
        else:
            # 3+ cols: equal-width paragraph columns
            per_col = (1.0 - 0.02 * ncols) / ncols
            width = f"{per_col:.3f}" + r"\linewidth"
            cell = r">{\raggedright\arraybackslash}p{" + width + "}"
            cols = cell * ncols
        out.append(r"\begin{longtable}[]{@{}" + cols + r"@{}}")
        i = k + 4
    return "".join(out)

text = simplify_longtables(text)

# ─── 4. Strip leftover `\real{...}` column calculations ─────────────
def strip_real(s):
    out, i = [], 0
    marker = r"{(\columnwidth"
    while i < len(s):
        j = s.find(marker, i)
        if j < 0:
            out.append(s[i:])
            break
        out.append(s[i:j])
        k = s.find(r"\real{", j)
        if k < 0:
            out.append(s[j:])
            break
        m = s.find(r"}}", k)
        if m < 0:
            out.append(s[j:])
            break
        i = m + 2
    return "".join(out)

text = strip_real(text)

# ─── 5. Strip minipage wrappers (they force cells to \linewidth) ────
def strip_minipages(s):
    begin = r"\begin{minipage}"
    end = r"\end{minipage}"
    out, i = [], 0
    while i < len(s):
        j = s.find(begin, i)
        if j < 0:
            out.append(s[i:])
            break
        out.append(s[i:j])
        header_end = s.find("\n", j)
        if header_end < 0:
            out.append(s[j:])
            break
        k = s.find(end, header_end)
        if k < 0:
            out.append(s[j:])
            break
        inner = s[header_end + 1:k].strip()
        out.append(inner)
        i = k + len(end)
    return "".join(out)

text = strip_minipages(text)

# ─── 6. Wrap longtables with smaller font + tight colsep ────────────
def wrap_longtables_small(s):
    begin = r"\begin{longtable}"
    end = r"\end{longtable}"
    prefix = "{" + r"\footnotesize" + r"\setlength{\tabcolsep}{3pt}" + "\n"
    suffix = "\n}"
    out, i = [], 0
    while i < len(s):
        j = s.find(begin, i)
        if j < 0:
            out.append(s[i:])
            break
        out.append(s[i:j])
        k = s.find(end, j)
        if k < 0:
            out.append(s[j:])
            break
        block = s[j:k + len(end)]
        out.append(prefix + block + suffix)
        i = k + len(end)
    return "".join(out)

text = wrap_longtables_small(text)

# ─── 7. Force width=\linewidth on every image ───────────────────────
def force_image_width(s):
    marker = r"\pandocbounded{\includegraphics["
    out, i = [], 0
    while i < len(s):
        j = s.find(marker, i)
        if j < 0:
            out.append(s[i:])
            break
        out.append(s[i:j])
        # Find the path: .../]{charts/...}}
        path_start = s.find("]{", j)
        if path_start < 0:
            out.append(s[j:])
            break
        path_end = s.find("}", path_start + 2)
        if path_end < 0:
            out.append(s[j:])
            break
        path = s[path_start + 2:path_end]
        # Close the \pandocbounded{...}
        after_pandocbounded = path_end + 2  # skip the final }}
        out.append(r"\includegraphics[width=\linewidth,keepaspectratio]{" + path + "}")
        i = after_pandocbounded
    return "".join(out)

text = force_image_width(text)

# ─── 8. Strip redundant "Figure NN: " prefix from captions ──────────
import re
cap_pat = re.compile(r"caption\{Figure\s+\d+:\s*", re.IGNORECASE)
text = cap_pat.sub(lambda m: "caption{", text)

# ─── 9. Replace problematic Unicode chars ───────────────────────────
UNICODE_MAP = {
    "\u2500": "-", "\u2502": "|", "\u250c": "+", "\u2510": "+",
    "\u2514": "+", "\u2518": "+", "\u251c": "+", "\u2524": "+",
    "\u252c": "+", "\u2534": "+", "\u253c": "+",
    "\u2550": "=", "\u2551": "|", "\u2554": "+", "\u2557": "+",
    "\u255a": "+", "\u255d": "+", "\u2560": "+", "\u2563": "+",
    "\u2566": "+", "\u2569": "+", "\u256c": "+", "\u256d": "+",
    "\u256e": "+", "\u256f": "+", "\u2570": "+",
    "\u2580": "#", "\u2584": "#", "\u2588": "#", "\u2591": ".",
    "\u2592": ":", "\u2593": "#",
    "\u2713": "Y", "\u2717": "X", "\u2022": "*", "\u2605": "*",
    "\u2606": "*", "\u25e6": "-", "\u2043": "-", "\u2219": ".",
    "\u21d2": "=>", "\u21d4": "<=>", "\u2193": "v", "\u2191": "^",
    "\u21a6": "|->", "\u2026": "...", "\u2192": "->", "\u2190": "<-",
    "\u2194": "<->",
}
for k, v in UNICODE_MAP.items():
    text = text.replace(k, v)

BODY.write_text(text, encoding="utf-8")

remaining = sorted(set(ord(c) for c in text if ord(c) > 255))
print(f"Lines: {len(text.split(chr(10)))}")
print(f"Remaining high-unicode codepoints: {[f'U+{cp:04X}' for cp in remaining]}")
print("OK")
