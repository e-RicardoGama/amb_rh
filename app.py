import dash
import dash_bootstrap_components as dbc

FONT_AWESOME = ["https://use.fontawesome.com/releases/v5.10.2/css/all.css"]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX,dbc.icons.FONT_AWESOME])

app.title = "Amb"
server = app.server
app.scripts.config.serve_locally = True
server = app.server