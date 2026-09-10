"""
Post-process nbconvert HTML exports to replace the four CSS variable
values that fail WCAG 1.4.3 (4.5:1 minimum contrast) against the
#f5f5f5 code-cell background, and to force the input-prompt opacity
to 1 so the 'In [ ]:' label is not rendered at 50% opacity.

This script is meant to run AFTER every `jupyter nbconvert --to html` call.
It is idempotent: running it twice does not change the values a second time.
"""

import re, os

ROOT = 'c:/Users/vkaki/2026_Sum_Virtual_GreatBasin_Water'

# CSS variable replacements: (old_value, new_value, description)
CSS_VAR_FIXES = [
    ('#408080', '#006060',
     'comment-color: was 4.2:1, now 5.9:1 on #f5f5f5'),
    ('#a2f', '#7700b3',
     'operator-color short: was 4.2:1, now 7.3:1'),
    # The shorthand #a2f and full form may both appear
    ('#aa22ff', '#7700b3',
     'operator-color full: was 4.2:1, now 7.3:1'),
    ('#080', '#006600',
     'number-color short: was 4.3:1, now 6.7:1'),
    ('#008800', '#006600',
     'number-color full: was 4.3:1, now 6.7:1'),
    # error color
    ('#f00', '#b00000',
     'error-color short: was 3.7:1, now 6.7:1'),
    ('#ff0000', '#b00000',
     'error-color full: was 3.7:1, now 6.7:1'),
]

# Prompt opacity fix: injected before </head>
PROMPT_STYLE = (
    '<style>\n'
    '/* WCAG 1.4.3: restore prompt opacity so In/Out labels meet 4.5:1.\n'
    '   The JupyterLab theme applies --jp-cell-prompt-not-active-opacity (0.5)\n'
    '   via the CSS var, which dilutes both InputPrompt and OutputPrompt. */\n'
    '.jp-InputPrompt,\n'
    '.jp-OutputPrompt {\n'
    '  opacity: 1 !important;\n'
    '  color: #595959 !important;  /* 7.0:1 on white */\n'
    '}\n'
    '</style>\n'
    '</head>'
)


def patch_html(path):
    with open(path, 'rb') as f:
        raw = f.read()
    html = raw.decode('utf-8', errors='replace')
    original = html

    # ── 1. Replace failing CSS variable values ──────────────────────────
    # The variables are defined inside a :root { } block in a <style> tag.
    # We only replace inside that block to avoid accidentally touching
    # color values in user code outputs.
    def fix_in_root(m):
        block = m.group(0)
        for old, new, desc in CSS_VAR_FIXES:
            before = block.count(old)
            block = block.replace(old, new)
            after = block.count(old)
            if before != after:
                print(f'    [{before - after}x] {desc}')
        return block

    html = re.sub(r':root\s*\{[^}]+\}', fix_in_root, html)

    # ── 2. Inject prompt opacity fix before </head> ──────────────────────
    if 'opacity: 1 !important' not in html:
        html = html.replace('</head>', PROMPT_STYLE, 1)
        print('    [+] injected InputPrompt opacity fix')

    if html != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f'    [ok] patched {os.path.basename(path)}')
    else:
        print(f'    [=] no changes needed in {os.path.basename(path)}')


HTMLS = [
    'threshold_grid_search_demo.html',
    'csda_Planet_fetcher_final.html',
    'notebook.html',
]

for name in HTMLS:
    path = os.path.join(ROOT, name)
    if os.path.exists(path):
        print(f'\n{name}:')
        patch_html(path)
    else:
        print(f'\n{name}: NOT FOUND (run nbconvert first)')

print('\nDone.')
