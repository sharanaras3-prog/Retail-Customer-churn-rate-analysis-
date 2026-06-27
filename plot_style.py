import plotly.graph_objects as go
import plotly.express as px

PALETTE = {
    "primary":   "#1A3C5E",
    "accent":    "#E05C2A",
    "green":     "#2D7D46",
    "purple":    "#7B3FA0",
    "orange":    "#C0720F",
    "light_bg":  "#F7F5F2",
    "grid":      "#E2DDD8",
}

MODEL_COLORS = {
    "KNN":               "#1A3C5E",
    "Decision Tree":     "#E05C2A",
    "Random Forest":     "#2D7D46",
    "Gradient Boosting": "#7B3FA0",
}

LAYOUT_DEFAULTS = dict(
    font_family="Arial",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=50, b=40, l=40, r=20),
    xaxis=dict(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False),
    yaxis=dict(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False),
)

def apply_layout(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=PALETTE["primary"])),
        height=height,
        **LAYOUT_DEFAULTS,
    )
    return fig
