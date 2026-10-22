"""Gráfico SVG do dia, por canal: observado, projeção do resto do dia e faixa P10 a P90.

O texto do gráfico é em pt-BR. O SVG é autocontido e abre em qualquer navegador.
"""
import numpy as np

from .config import BINS_PER_DAY, BIN_MINUTES

STATE_STYLE = {"OUTAGE": ("#fee2e2", "#be123c", "queda"), "RECOVERY": ("#ffedd5", "#c2410c", "retenção"),
               "SURGE": ("#ede9fe", "#6d28d9", "pico")}
BLUE, GREEN, GRAY, LIGHT_BLUE = "#1d4ed8", "#059669", "#94a3b8", "#93c5fd"


def _state_runs(states):
    """Trechos consecutivos de um mesmo estado diferente de NORMAL: (início, fim, estado)."""
    runs, t = [], 0
    while t < len(states):
        j = t
        while j < len(states) and states[j] == states[t]:
            j += 1
        if states[t] != "NORMAL":
            runs.append((t, j, states[t]))
        t = j
    return runs


def render_day_svg(channels, now_bin, subtitle):
    """Monta o SVG de um dia. `channels` é {canal: dict(real, profile, state, snaps)}, como em snapshots.compute_snapshots."""
    width, left, right, panel_h = 1200, 90, 1160, 250
    height = 150 + panel_h * len(channels)
    hh, mm = divmod(now_bin * BIN_MINUTES, 60)

    def text(x, y, s, size=11, color="#0f172a", weight=400, anchor="start"):
        return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{s}</text>'

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'font-family="Inter, Arial, Helvetica, sans-serif"><rect width="100%" height="100%" fill="#fff"/>',
           text(40, 34, "Transações por minuto: observado e projeção do dia todo", 19, weight=700),
           text(40, 54, subtitle, 11.5, "#64748b")]
    for i, (channel, c) in enumerate(channels.items()):
        top = 140 + i * panel_h
        bottom = top + panel_h - 70
        p50, lo, hi = c["snaps"][now_bin]
        real, profile = c["real"], c["profile"]
        future = real[now_bin + 1 :]
        has_future = len(p50) > 0  # às 23:55 não sobra intervalo para projetar
        ymax = max(real.max(), hi.max(initial=0), profile.max()) / BIN_MINUTES * 1.1
        x = lambda b: left + b / BINS_PER_DAY * (right - left)
        y = lambda v: bottom - v / ymax * (bottom - top)
        current = real[now_bin] / BIN_MINUTES
        total_projected = real[: now_bin + 1].sum() + p50.sum()
        out.append(text(40, top - 30, channel, 14, weight=700))
        if has_future:
            error = np.abs(future - p50).sum() / future.sum()
            floor = np.sqrt(2 * p50 / np.pi).sum() / p50.sum()  # erro mínimo possível por ruído de contagem
            inside = ((future >= lo) & (future <= hi)).mean()
            head = (f"agora {current:,.0f}/min  |  próximos 5 min {p50[0] / BIN_MINUTES:,.0f}/min "
                    f"(faixa {lo[0] / BIN_MINUTES:,.0f} a {hi[0] / BIN_MINUTES:,.0f})  |  estado {c['state'][now_bin]}")
            detail = (f"depois de agora, a projeção ficou {error:.1%} longe do real (mínimo possível por ruído: {floor:.1%})  |  "
                      f"real dentro da faixa de 80%: {inside:.0%}  |  "
                      f"total do dia: projetado {total_projected:,.0f}, observado {real.sum():,.0f} "
                      f"({total_projected / real.sum() - 1:+.1%})")
        else:
            head = f"agora {current:,.0f}/min  |  fim do dia: não sobra horário para projetar  |  estado {c['state'][now_bin]}"
            detail = f"total do dia: {real.sum():,.0f} transações observadas"
        out.append(text(40 + 8 * len(channel) + 30, top - 30, head, 11, "#334155"))
        out.append(text(40, top - 14, detail, 11, "#334155"))
        for k, (a, b, state) in enumerate(_state_runs(c["state"][: now_bin + 1])):
            fill, stroke, name = STATE_STYLE[state]
            out.append(f'<rect x="{x(a)}" y="{top}" width="{x(b) - x(a)}" height="{bottom - top}" fill="{fill}"/>')
            out.append(text(x(a) - 3 if k % 2 == 0 else x(b) + 3, top + 13, name, 9.5, stroke, 600,
                            "end" if k % 2 == 0 else "start"))
        for hour in range(0, 25, 3):
            out.append(f'<line x1="{x(hour * 12)}" y1="{top}" x2="{x(hour * 12)}" y2="{bottom}" stroke="#f1f5f9"/>')
            out.append(text(x(hour * 12), bottom + 16, f"{hour:02d}:00", 10, "#64748b", anchor="middle"))
        for k in range(5):
            value = ymax * k / 4
            out.append(f'<line x1="{left}" y1="{y(value)}" x2="{right}" y2="{y(value)}" stroke="#e2e8f0"/>')
            out.append(text(left - 8, y(value) + 4, f"{value:,.0f}", 10, "#64748b", anchor="end"))
        bins = np.arange(now_bin + 1, BINS_PER_DAY)
        if has_future:
            band = " ".join(f"{x(b):.1f},{y(v / BIN_MINUTES):.1f}" for b, v in zip(bins, hi)) + " " + \
                   " ".join(f"{x(b):.1f},{y(v / BIN_MINUTES):.1f}" for b, v in zip(bins[::-1], lo[::-1]))
            out.append(f'<polygon points="{band}" fill="{GREEN}" fill-opacity="0.16"/>')

        def line(values, color, width_, dash="", first_bin=0):
            pts = " ".join(f"{x(first_bin + j):.1f},{y(v / BIN_MINUTES):.1f}" for j, v in enumerate(values))
            return (f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width_}" '
                    f'stroke-dasharray="{dash}"/>')

        out.append(line(profile, GRAY, 1.2, "2,3"))
        if has_future:
            out.append(line(future, LIGHT_BLUE, 1.4, first_bin=now_bin + 1))
            out.append(line(p50, GREEN, 2.4, "7,5", first_bin=now_bin + 1))
        out.append(line(real[: now_bin + 1], BLUE, 2.2))
        out.append(f'<line x1="{x(now_bin)}" y1="{top}" x2="{x(now_bin)}" y2="{bottom}" stroke="#334155" stroke-width="1.2"/>')
        out.append(f'<circle cx="{x(now_bin)}" cy="{y(current)}" r="4.5" fill="{BLUE}" stroke="#fff" stroke-width="2"/>')
        out.append(text(x(now_bin), bottom + 32, f"agora {hh:02d}:{mm:02d}", 10.5, weight=700, anchor="middle"))
    legend_y = height - 24
    items = ((40, BLUE, 2.2, "", "observado até agora"), (220, LIGHT_BLUE, 1.4, "", "observado depois (só no simulador)"),
             (470, GREEN, 2.4, "7,5", f"projeção feita às {hh:02d}:{mm:02d} (P50)"), (710, None, 0, "", "faixa de 80%"),
             (860, GRAY, 1.2, "2,3", "projeção das 00:00 (dias parecidos)"))
    for lx, color, w, dash, label in items:
        if color is None:
            out.append(f'<rect x="{lx}" y="{legend_y - 7}" width="30" height="14" fill="{GREEN}" fill-opacity="0.16"/>')
        else:
            out.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 30}" y2="{legend_y}" stroke="{color}" '
                       f'stroke-width="{w}" stroke-dasharray="{dash}"/>')
        out.append(text(lx + 38, legend_y + 4, label, 11, "#334155"))
    out.append("</svg>")
    return "\n".join(out)
