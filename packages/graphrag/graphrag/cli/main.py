import typer
app = typer.Typer(help="GraphRAG mini skeleton")

@app.command()
def index(root: str = "."):
    """建索引（占位）。"""
    typer.echo(f"index root={root}")

@app.command()
def query(root: str = ".", q: str = ""):
    """查询（占位）。"""
    typer.echo(f"query root={root} q={q}")
