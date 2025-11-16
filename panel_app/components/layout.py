# panel_app/components/layout.py
import panel as pn
from typing import Dict, Any


def create_layout(views: Dict[str, Any]) -> pn.Column:
    """Create the main application layout with navigation.

    Args:
        views: Dictionary of view components with their names as keys

    Returns:
        A Panel layout with navigation and main content area
    """
    # Navigation configuration
    nav_items = [
        ('📈 Pricing Explorer', 'pricing'),
        ('📊 Volatility Surface', 'vol_surface'),
        ('📉 Backtest Results', 'backtest')
    ]

    # Create navigation buttons
    nav_buttons = pn.Row(sizing_mode='stretch_width')
    for item in nav_items:
        btn = pn.widgets.Button(
            name=item[0],
            button_type='light',
            width=200,
            margin=(5, 10)
        )
        btn.param.watch(
            lambda event, view=item[1]: main_area.objects.clear() or main_area.append(views[view].layout),
            'clicks'
        )
        nav_buttons.append(btn)

    # Main content area
    main_area = pn.Column(
        views[nav_items[0][1]].layout,
        sizing_mode='stretch_both'
    )

    # Combine everything
    return pn.Column(
        pn.Row(
            pn.pane.Markdown("## CSO Analytics Dashboard", margin=(10, 20)),
            nav_buttons,
            sizing_mode='stretch_width'
        ),
        pn.layout.Divider(),
        main_area,
        sizing_mode='stretch_both'
    )