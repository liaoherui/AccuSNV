"""Sphinx configuration for the AccuSNV documentation."""

project = 'AccuSNV'
copyright = '2026, Lieberman Lab, MIT'
author = 'Herui Liao, Alex Crits-Christoph and the Lieberman Lab'
release = '1.1.0'
version = '1.1'

extensions = [
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_design',
]

myst_enable_extensions = ['colon_fence', 'deflist', 'attrs_inline', 'substitution', 'dollarmath']
myst_heading_anchors = 3

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_logo = '_static/figures/logo.png'
html_title = 'AccuSNV documentation'
html_favicon = '_static/figures/logo.png'

html_theme_options = {
    'logo_only': False,
    'navigation_depth': 3,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'style_external_links': True,
}

# Sphinx wraps every table in a horizontally scrolling container via custom.css, so wide
# column tables (the SNV table reference) stay readable rather than squeezing every cell.
html_show_sourcelink = False
