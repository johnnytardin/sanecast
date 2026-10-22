"""Ponto de entrada da demonstração de como projetar dados futuros a partir do histórico.
Um comando para cada etapa; `demo` roda tudo.

    uv run cli.py demo                  # simula os dados, avalia o modelo, gera o gráfico de hoje e o painel
    uv run cli.py simulate              # só gera data/transactions_1min.csv, truth_1min.csv e future_1min.csv
    uv run cli.py evaluate              # mede o erro do modelo e dos adversários
    uv run cli.py today --now 15:30     # gráfico SVG do dia, como se fossem 15:30
    uv run cli.py dashboard             # painel HTML interativo: últimos dias, hoje e os próximos 30 dias
    uv run cli.py tune                  # testa variações dos hiperparâmetros (validação e teste)

Os dados simulados acompanham a data do computador: terminam hoje e o painel mostra 30 dias à frente. Quando os comandos
`evaluate`, `today`, `dashboard` ou `tune` encontram dados simulados de outro dia, geram de novo até hoje.

Para usar dados reais, coloque um CSV com as colunas ts, channel, count em data/transactions_1min.csv
(veja docs/DATA.md), apague data/future_1min.csv e data/truth_1min.csv e rode `today` ou `dashboard`.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sanecast import quantile_model  # noqa: E402
from sanecast.config import BIN_MINUTES, DATA_DIR, OUT_DIR, ROOT  # noqa: E402
from sanecast.dashboard import build_dashboard  # noqa: E402
from sanecast.data import load_future, load_minutes, load_series, load_truth  # noqa: E402
from sanecast.pipeline import Forecaster  # noqa: E402
from sanecast.report import render_day_svg  # noqa: E402
from sanecast.simulate import simulate  # noqa: E402
from sanecast.snapshots import compute_snapshots  # noqa: E402
from sanecast.tuning import run_grid  # noqa: E402
from sanecast.workflow import run_evaluation  # noqa: E402

EVALUATION_DAYS = 21  # `evaluate` mede, por padrão, os últimos 21 dias dos dados
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def is_stale_simulation():
    """True se os dados são do simulador (o arquivo do futuro começa logo depois do último dia do histórico) e esse
    último dia não é hoje. Dados reais nunca batem com o arquivo do futuro, então não são tocados."""
    csv, future = DATA_DIR / "transactions_1min.csv", DATA_DIR / "future_1min.csv"
    if not (csv.exists() and future.exists()):
        return False
    last = pd.read_csv(csv, usecols=["ts"], parse_dates=["ts"])["ts"].max().normalize()
    first_future = pd.read_csv(future, nrows=1, parse_dates=["ts"])["ts"][0].normalize()
    return (first_future - last).days == 1 and last != pd.Timestamp.today().normalize()


def ensure_data():
    """Gera os dados simulados até hoje se não existirem ou se forem de um dia anterior."""
    if not (DATA_DIR / "transactions_1min.csv").exists() or is_stale_simulation():
        print("Generating simulated data up to today...")
        simulate()


def shown(path):
    """Caminho relativo à pasta do projeto, igual dentro e fora do Docker."""
    return Path(path).resolve().relative_to(ROOT)


def parse_now(text):
    hours, minutes = text.split(":")
    return (int(hours) * 60 + int(minutes)) // BIN_MINUTES


def print_evaluation(result):
    """Imprime as tabelas principais. Os CSVs completos ficam em out/."""
    print("\nDirty-data detection in the history (the model receives no marks at all)")
    print("channel    found(real dirty bins)  correct(marks)  false alarm on seasonal days")
    for channel, (found, correct, false_alarm) in result.cleaning.items():
        print(f"{channel:9s}  {found:21.1%}  {correct:14.1%}  {false_alarm:27.1%}")
    m = result.metrics
    for horizon in ("5min", "1h", "day_ahead"):
        table = m[m.horizon == horizon].pivot_table(index=["channel", "scenario"], columns="method",
                                                    values="wape_exp", sort=False)
        print(f"\nWAPE vs true rate (%), horizon {horizon}. Lower is better.")
        print((table * 100).round(2).to_string())
    observed = m[(m.horizon == "5min") | (m.method == "noise_floor")]
    table = observed.pivot_table(index=["channel", "scenario"], columns="method", values="wape_obs", sort=False)
    columns = [c for c in ("noise_floor", "model", "model_lgbm_p50", "no_cleaning", "last_week") if c in table]
    print("\nWAPE vs observed (%), 5 min. 'noise_floor' is the error of a PERFECT forecast: nobody beats it.")
    print((table[columns] * 100).round(2).to_string())
    print("\nCoverage of the P10-P90 band (target 80%; 'event' is measured against the true rate)")
    print(result.coverage.pivot_table(index=["channel", "scenario"], columns="horizon", values="coverage",
                                      sort=False).round(2).to_string())


def cmd_simulate(args):
    path = simulate(today=args.today)
    print(f"Data written to {shown(path)} (+ truth_1min.csv, future_1min.csv)")


def cmd_evaluate(args):
    ensure_data()
    days, series = load_series()
    end = args.end or days[-1]
    start = args.start or end - pd.Timedelta(days=EVALUATION_DAYS - 1)
    result = run_evaluation(days, series, load_truth(days), start, end,
                            with_statsforecast=not args.no_statsforecast)
    OUT_DIR.mkdir(exist_ok=True)
    result.metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
    result.coverage.to_csv(OUT_DIR / "coverage.csv", index=False)
    print_evaluation(result)
    print(f"\nFull tables in {shown(OUT_DIR)}/metrics.csv and coverage.csv")


def cmd_today(args):
    ensure_data()
    days, series = load_series()
    day = pd.Timestamp(args.day) if args.day else days[-1]
    d = list(days).index(day)
    now_bin = parse_now(args.now)
    forecaster = Forecaster(days, series)
    models = quantile_model.train(forecaster, day)
    snapshots = compute_snapshots(forecaster, models, [d], [now_bin])
    subtitle = (f"{day:%d/%m/%Y}. Dados simulados. Agora são {args.now}. O modelo só usa o que veio até esse horário; "
                f"o que aconteceu depois aparece só para comparar.")
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"projection_{day.date()}.svg"
    path.write_text(render_day_svg(snapshots[d], now_bin, subtitle), encoding="utf-8")
    print(f"Chart written to {shown(path)}")


def cmd_dashboard(args):
    ensure_data()
    days, series = load_series()
    forecaster = Forecaster(days, series)
    models = quantile_model.train(forecaster, days[len(days) - 1 - args.past_days])
    future_days, future_series = load_future()
    if future_days is not None:
        future_days, future_series = future_days[:args.future_days], {c: m[:args.future_days] for c, m in future_series.items()}
    minute_counts = {}
    for name in ("transactions_1min.csv", "future_1min.csv"):
        minute_days, minute_series = load_minutes(name=name)
        if minute_days is not None:
            for i, date in enumerate(minute_days):
                minute_counts[str(date.date())] = {channel: matrix[i] for channel, matrix in minute_series.items()}
    path = build_dashboard(forecaster, models, args.past_days, OUT_DIR / "dashboard.html", (future_days, future_series), minute_counts)
    print(f"Dashboard written to {shown(path)}\nOpen it in a browser (double click, no server needed).")


def cmd_tune(args):
    ensure_data()
    days, series = load_series()
    result = run_grid(Forecaster(days, series), load_truth(days))
    OUT_DIR.mkdir(exist_ok=True)
    result.to_csv(OUT_DIR / "tuning_grid.csv", index=False)
    print("Error (%) vs the true rate, mean over channels and horizons. Choose by validation; test confirms.")
    print(result[["variant", "validation:mean", "test:mean"]].round(2).to_string())


def cmd_demo(args):
    cmd_simulate(argparse.Namespace(today=None))
    cmd_evaluate(argparse.Namespace(start=None, end=None, no_statsforecast=False))
    cmd_today(argparse.Namespace(day=None, now="15:30"))
    cmd_dashboard(argparse.Namespace(past_days=7, future_days=30))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("simulate", help="generate the simulated data up to today (and 30 days ahead)")
    p.add_argument("--today", default=None, help="last day of the history, YYYY-MM-DD (default: today's date).")
    p.set_defaults(func=cmd_simulate)
    p = sub.add_parser("evaluate", help="measure the error of the model and the baselines")
    p.add_argument("--start", default=None, help="first evaluated day (default: 20 days before --end)")
    p.add_argument("--end", default=None, help="last evaluated day (default: last day in the data)")
    p.add_argument("--no-statsforecast", action="store_true", help="skip the StatsForecast models (faster)")
    p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("today", help="SVG chart of a day, as if it were a given time")
    p.add_argument("--day", default=None, help="day (default: last day in the data)")
    p.add_argument("--now", default="15:30", help="time HH:MM (default %(default)s)")
    p.set_defaults(func=cmd_today)
    p = sub.add_parser("dashboard", help="interactive HTML dashboard")
    p.add_argument("--past-days", type=int, default=7, help="days before today in the dashboard (default %(default)s)")
    p.add_argument("--future-days", type=int, default=30, help="simulated days after today (default %(default)s, max 30)")
    p.set_defaults(func=cmd_dashboard)
    sub.add_parser("tune", help="test hyperparameter variations").set_defaults(func=cmd_tune)
    sub.add_parser("demo", help="run everything in sequence").set_defaults(func=cmd_demo)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
