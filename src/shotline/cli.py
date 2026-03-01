"""CLI entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(name="shotline", help="Photo processing pipeline.", no_args_is_help=True)
err = Console(stderr=True)
out = Console()


@app.callback()
def _init() -> None:
    import shotline.processors  # noqa: F401


@app.command()
def run(
    input_path: Annotated[Path, typer.Argument(help="Input image or directory", exists=True)],
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output path")] = None,
    steps: Annotated[
        str | None, typer.Option("--steps", "-s", help="Comma-separated steps")
    ] = None,
    config_path: Annotated[Path | None, typer.Option("--config", "-c", help="Config file")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="JSON output to stdout")] = False,
) -> None:
    """Process image(s) through the pipeline."""
    from shotline.config import load_config
    from shotline.io import SUPPORTED_EXTENSIONS
    from shotline.pipeline import Pipeline

    cfg = load_config(config_path)
    step_list = steps.split(",") if steps else None

    # Collect files
    if input_path.is_dir():
        files = [
            f for f in sorted(input_path.iterdir()) if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not files:
            err.print(f"[red]No supported images in {input_path}[/red]")
            raise typer.Exit(code=1)
        out_dir = output or input_path / "processed"
    else:
        files = [input_path]
        out_dir = None

    try:
        pipeline = Pipeline(steps=step_list, config=cfg)
    except Exception as e:
        _handle_error(e, json_output)
        raise typer.Exit(code=1) from e

    results = []
    for f in files:
        if out_dir:
            out_path = out_dir / f"{f.stem}{cfg.output.suffix}{_out_ext(cfg)}"
        elif output:
            out_path = output
        else:
            out_path = f.with_stem(f.stem + cfg.output.suffix)

        try:
            r = pipeline.run(f, out_path)
            results.append({"input": str(f), "output": str(out_path), **r.to_dict()})
            if not json_output:
                err.print(f"  [green]✓[/green] {f.name} → {out_path}")
        except Exception as e:
            if json_output:
                results.append({"input": str(f), "error": str(e)})
            else:
                err.print(f"  [red]✗[/red] {f.name}: {e}")

    if json_output:
        sys.stdout.write(json.dumps(results, indent=2) + "\n")
    elif len(files) > 1:
        err.print(f"[green]Done.[/green] {len(results)} images processed.")


def _out_ext(cfg: object) -> str:
    return f".{cfg.output.format}"  # type: ignore[attr-defined]


def _handle_error(e: Exception, json_output: bool) -> None:
    if json_output:
        sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
    else:
        err.print(f"[red]Error:[/red] {e}")


@app.command("list")
def list_steps(
    json_output: Annotated[bool, typer.Option("--json", help="JSON output")] = False,
) -> None:
    """List available processing steps."""
    from shotline.processor import get_processor, list_processors

    processors = list_processors()
    if json_output:
        data = []
        for meta in processors:
            proc = get_processor(meta.name)
            data.append(
                {
                    "name": meta.name,
                    "display_name": meta.display_name,
                    "description": meta.description,
                    "order": meta.order,
                    "requires_model": meta.requires_model,
                    "status": proc.status().value,
                }
            )
        sys.stdout.write(json.dumps(data, indent=2) + "\n")
    else:
        for meta in processors:
            proc = get_processor(meta.name)
            s = proc.status().value
            icon = "[green]✓[/green]" if s == "available" else "[yellow]○[/yellow]"
            out.print(f"  {meta.order:3d}. {icon} {meta.display_name:<25} [dim]{meta.name}[/dim]")


@app.command()
def models(
    action: Annotated[str, typer.Argument(help="download | status | clean")],
    name: Annotated[str | None, typer.Argument(help="Model name or 'all'")] = None,
) -> None:
    """Manage AI models."""
    from shotline.models import ModelManager

    manager = ModelManager()
    if action == "download":
        manager.download(name or "all")
    elif action == "status":
        manager.print_status()
    elif action == "clean":
        manager.clean(name)
    else:
        err.print(f"[red]Unknown action:[/red] {action}")
        raise typer.Exit(code=1)


def main() -> None:
    import shotline.processors  # noqa: F401

    app()
