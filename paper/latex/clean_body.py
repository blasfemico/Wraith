"""Clean pandoc-generated LaTeX body for robust pdflatex compilation."""
import re

BODY = r'D:\coding\repositorio-git\Wraith\paper\latex\wraith_paper_body.tex'

with open(BODY, encoding='utf-8') as f:
    text = f.read()

# 1) Fix pandoc longtable column specs — replace the verbose
#    `>{\raggedright\arraybackslash}p{(\columnwidth - 8\tabcolsep) * \real{0.2}}` mess
#    with simple 'l' columns. pdflatex handles that reliably.
longtable_pat = re.compile(
    r"\\begin\{longtable\}\[\]\{@\{\}(?:[^@]*?)@\{\}\}",
    re.DOTALL,
)

def replace_longtable(m):
    block = m.group(0)
    ncols = block.count(r'>{')
    if ncols == 0:
        ncols = 2  # fallback
    cols = 'l' * ncols
    return r'\begin{longtable}[]{@{}' + cols + r'@{}}'

text = longtable_pat.sub(replace_longtable, text)

# 2) Remove any stray minipage/p{} column calcs that may have been injected
text = re.sub(
    r"\{\(\\columnwidth - \d+\\tabcolsep\) \* \\real\{[\d.]+\}\}",
    "", text,
)

# 3) ASCII-ify box drawing / emoji / fancy arrows that pdflatex chokes on
box_repl = {
    "\u2500": "-",  "\u2502": "|",  "\u250c": "+",  "\u2510": "+",
    "\u2514": "+",  "\u2518": "+",  "\u251c": "+",  "\u2524": "+",
    "\u252c": "+",  "\u2534": "+",  "\u253c": "+",
    "\u2550": "=",  "\u2551": "|",  "\u2554": "+",  "\u2557": "+",
    "\u255a": "+",  "\u255d": "+",  "\u2560": "+",  "\u2563": "+",
    "\u2566": "+",  "\u2569": "+",  "\u256c": "+",  "\u256d": "+",
    "\u256e": "+",  "\u256f": "+",  "\u2570": "+",
    "\u2580": "#",  "\u2584": "#",  "\u2588": "#",  "\u2591": ".",
    "\u2592": ":",  "\u2593": "#",
}
for k, v in box_repl.items():
    text = text.replace(k, v)

other_repl = {
    "\u2713": "Y",  "\u2717": "X",  "\u2022": "*",  "\u2605": "*",
    "\u2606": "*",  "\u25e6": "-",  "\u2043": "-",  "\u2219": ".",
    "\u21d2": "=>", "\u21d4": "<=>","\u2193": "v",  "\u2191": "^",
    "\u21a6": "|->","\u2026": "...","\u2192": "->", "\u2190": "<-",
    "\u2194": "<->","\u2194": "<->",
}
for k, v in other_repl.items():
    text = text.replace(k, v)

with open(BODY, "w", encoding="utf-8") as f:
    f.write(text)

non_ascii = set(c for c in text if ord(c) > 255)
print("Remaining U+>FF codepoints:",
      sorted(f"U+{ord(c):04X}" for c in non_ascii)[:30])
print("Lines:", len(text.split("\n")))
print("OK")
