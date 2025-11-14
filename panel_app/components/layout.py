"""
Dashboard layout and styling.
"""

import panel as pn

def create_layout(tabs):
    title = pn.pane.Markdown(
        "# Cross Commodity Spread Options Lab",
        style={"font-size": "28px", "text-align": "center"}
    )

    return pn.Column(
        title,
        pn.Spacer(height=20),
        tabs
    )

