#!/usr/bin/env python3
"""Воспроизводимый расчёт стоимости прогонов LLM и генерация SVG-диаграмм исследования 2026."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESEARCH_DIR = ROOT / "docs" / "research"
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

# Параметры типового запроса и полного пакета модуля «Уровень сервиса» (128 требований)
N_IN_REQ = 2000
N_OUT_REQ = 800

# Полный пакет: 128 требований * 2 шага генерации = 256 базовых вызовов
# + 15% автоповторов валидатора = ~295 вызовов.
# В среднем на 1 вызов: 2200 входных токенов (из них 1400 - стабильный системный промпт и few-shot,
# которые кэшируются начиная со 2-го вызова; 800 - текст требования и фрагмент ТЗ) и 900 выходных токенов.
BATCH_CALLS = round(128 * 2 * 1.15)  # 294 вызова
CACHED_PREFIX_PER_CALL = 1400
UNCACHED_INPUT_PER_CALL = 800
OUTPUT_PER_CALL = 900


def calc_single_request_cost(p_in: float, p_out: float) -> float:
    return (N_IN_REQ * p_in + N_OUT_REQ * p_out) / 1_000_000


def calc_batch_cost(p_in: float, p_out: float, p_cache: float) -> tuple[float, float]:
    """Возвращает (стоимость без кэша USD, стоимость с кэшем префикса USD) на 128 требований."""
    total_in = BATCH_CALLS * (CACHED_PREFIX_PER_CALL + UNCACHED_INPUT_PER_CALL)
    total_out = BATCH_CALLS * OUTPUT_PER_CALL
    no_cache = (total_in * p_in + total_out * p_out) / 1_000_000

    # Первый вызов прогревает кэш, остальные 293 используют cache hit для префикса
    cached_tokens = (BATCH_CALLS - 1) * CACHED_PREFIX_PER_CALL
    uncached_tokens = CACHED_PREFIX_PER_CALL + BATCH_CALLS * UNCACHED_INPUT_PER_CALL
    with_cache = (uncached_tokens * p_in + cached_tokens * p_cache + total_out * p_out) / 1_000_000
    return no_cache, with_cache


def load_data() -> tuple[list[dict[str, str]], dict[tuple[str, str], float]]:
    with (DATA_DIR / "model_candidates_2026.csv").open(encoding="utf-8-sig", newline="") as f:
        candidates = list(csv.DictReader(f))
    benchmarks: dict[tuple[str, str], float] = {}
    with (DATA_DIR / "benchmark_results_2026.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            benchmarks[(row["candidate_id"], row["benchmark"])] = float(row["score_percent"])
    return candidates, benchmarks


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_price_quality_svg(
    candidates: list[dict[str, str]],
    benchmarks: dict[tuple[str, str], float],
) -> Path:
    """Строит диаграмму рассеяния: стоимость запроса (log10) vs GPQA Diamond (%)."""
    width, height = 980, 560
    ml, mr, mt, mb = 88, 36, 78, 86
    pw = width - ml - mr
    ph = height - mt - mb

    # Отбираем облачные модели с известной ценой > 0 и GPQA Diamond
    # Плюс добавляем Qwen3.8-27B отдельно как локальный маркер на левой границе? Нет, честнее показать её в сноске,
    # а на числовой лог-шкале отображать только платные облачные модели с ценой > 0.
    points = []
    # Ручные смещения подписей (dx, dy, anchor), чтобы гарантировать отсутствие наложений
    label_offsets = {
        "DEEPSEEK_V41_FLASH": (12, -10, "start"),
        "GPT_OSS_120B_OLD": (12, 14, "start"),
        "DEEPSEEK_V4_PRO": (12, 4, "start"),
        "CLAUDE_SONNET_55": (-12, 16, "end"),
        "GEMINI_4_CARBON": (-12, -10, "end"),
        "GPT_62_SOL": (12, -8, "start"),
        "GEMINI_31_PRO_OLD": (12, 16, "start"),
        "CLAUDE_OPUS_55": (-12, 14, "end"),
        "GPT_62_ASTRA": (-12, 14, "end"),
    }
    short_names = {
        "DEEPSEEK_V41_FLASH": "DeepSeek V4.1 Flash (Китай)",
        "DEEPSEEK_V4_PRO": "DeepSeek V4 Pro (Китай)",
        "GEMINI_4_CARBON": "Gemini 4 Carbon",
        "GPT_62_SOL": "GPT-6.2 Sol",
        "GPT_62_ASTRA": "GPT-6.2 Astra",
        "CLAUDE_SONNET_55": "Claude Sonnet 5.5",
        "CLAUDE_OPUS_55": "Claude Opus 5.5",
        "GPT_OSS_120B_OLD": "GPT-OSS 120B (старая база)",
        "GEMINI_31_PRO_OLD": "Gemini 3.1 Pro (старая база)",
    }

    for c in candidates:
        cid = c["candidate_id"]
        if cid not in short_names:
            continue
        p_in = float(c["price_in_usd_per_1m"])
        p_out = float(c["price_out_usd_per_1m"])
        gpqa = benchmarks.get((cid, "GPQA Diamond"))
        if p_in <= 0 or gpqa is None:
            continue
        cost = calc_single_request_cost(p_in, p_out)
        points.append(
            {
                "cid": cid,
                "label": short_names[cid],
                "cost": cost,
                "gpqa": gpqa,
                "eco": c["ecosystem"],
            }
        )

    min_log = math.log10(0.0005)
    max_log = math.log10(0.07)
    min_y, max_y = 76.0, 99.0

    def x_pos(cost: float) -> float:
        return ml + (math.log10(cost) - min_log) / (max_log - min_log) * pw

    def y_pos(score: float) -> float:
        return mt + (max_y - score) / (max_y - min_y) * ph

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>',
        '  .title { font-family: Arial, sans-serif; font-size: 19px; font-weight: bold; fill: #1F344C; }',
        '  .subtitle { font-family: Arial, sans-serif; font-size: 12px; fill: #526375; }',
        '  .axis-label { font-family: Arial, sans-serif; font-size: 12px; font-weight: bold; fill: #243240; }',
        '  .tick { font-family: Arial, sans-serif; font-size: 11px; fill: #526375; }',
        '  .pt-label { font-family: Arial, sans-serif; font-size: 11.5px; font-weight: bold; fill: #1F344C; }',
        '  .pt-sub { font-family: Arial, sans-serif; font-size: 10px; fill: #526375; }',
        '  .legend { font-family: Arial, sans-serif; font-size: 11.5px; fill: #243240; }',
        '</style>',
        f'<text x="{ml}" y="34" class="title">Соотношение цены запроса и логического вывода (GPQA Diamond), ноябрь 2026 г.</text>',
        f'<text x="{ml}" y="54" class="subtitle">Условный запрос: 2 000 вх. + 800 вых. токенов · Сравнение китайских моделей, западных флагманов и базы слайда 10 (ред. 1)</text>',
    ]

    # Выделенная зона «Оптимум цена/качество (Китай 2026)»
    zx1 = x_pos(0.0006)
    zx2 = x_pos(0.0042)
    zy1 = y_pos(94.0)
    zy2 = y_pos(89.5)
    lines.append(
        f'<rect x="{zx1:.1f}" y="{zy1:.1f}" width="{zx2-zx1:.1f}" height="{zy2-zy1:.1f}" '
        f'rx="8" fill="#EBF5EE" stroke="#2D7A4D" stroke-dasharray="5,4" stroke-width="1.2"/>'
    )
    lines.append(
        f'<text x="{zx1+10:.1f}" y="{zy1+16:.1f}" font-family="Arial, sans-serif" font-size="11" '
        f'font-weight="bold" fill="#2D7A4D">Зона оптимума «цена / мощность» (Китай, 2026)</text>'
    )

    # Горизонтальная линия локальной Qwen3.8-27B (0 ₽ API)
    qy = y_pos(89.2)
    lines.append(
        f'<line x1="{ml}" y1="{qy:.1f}" x2="{ml+pw}" y2="{qy:.1f}" stroke="#2D7A4D" stroke-width="1.2" stroke-dasharray="3,4"/>'
    )
    lines.append(
        f'<text x="{ml+pw-6}" y="{qy-6:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="10.5" fill="#2D7A4D">'
        f'Локальная Qwen3.8-27B (Apache 2.0, контур NDA): 89,2% GPQA при 0 ₽ за API</text>'
    )

    # Сетка Y
    for y_val in (80, 85, 90, 95):
        yy = y_pos(float(y_val))
        lines.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="#E5ECF2" stroke-width="1"/>')
        lines.append(f'<text x="{ml-10}" y="{yy+4:.1f}" text-anchor="end" class="tick">{y_val}%</text>')

    # Сетка X (лог-шкала)
    for x_val, lbl in (
        (0.0005, "$0.0005"),
        (0.001, "$0.001"),
        (0.002, "$0.002"),
        (0.005, "$0.005"),
        (0.01, "$0.01"),
        (0.02, "$0.02"),
        (0.05, "$0.05"),
    ):
        xx = x_pos(x_val)
        lines.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt+ph}" stroke="#E5ECF2" stroke-width="1"/>')
        lines.append(f'<text x="{xx:.1f}" y="{mt+ph+20}" text-anchor="middle" class="tick">{lbl}</text>')

    # Оси
    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#95A5B6" stroke-width="1.5"/>')
    lines.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#95A5B6" stroke-width="1.5"/>')
    lines.append(
        f'<text x="{ml + pw/2:.1f}" y="{height-34}" text-anchor="middle" class="axis-label">'
        f'Стоимость 1 типового запроса, USD (логарифмическая шкала, меньше = дешевле)</text>'
    )
    lines.append(
        f'<text x="24" y="{mt + ph/2:.1f}" text-anchor="middle" transform="rotate(-90 24 {mt + ph/2:.1f})" class="axis-label">'
        f'GPQA Diamond, % (выше = мощнее логика)</text>'
    )

    # Стрелка прогресса от GPT-OSS 120B к DeepSeek V4.1 Flash
    p_old = next(p for p in points if p["cid"] == "GPT_OSS_120B_OLD")
    p_new = next(p for p in points if p["cid"] == "DEEPSEEK_V41_FLASH")
    lines.append(
        f'<line x1="{x_pos(p_old["cost"]):.1f}" y1="{y_pos(p_old["gpqa"])-9:.1f}" '
        f'x2="{x_pos(p_new["cost"]):.1f}" y2="{y_pos(p_new["gpqa"])+10:.1f}" '
        f'stroke="#2D7A4D" stroke-width="1.8" stroke-dasharray="4,3"/>'
    )
    lines.append(
        f'<text x="{x_pos(p_old["cost"])+8:.1f}" y="{(y_pos(p_old["gpqa"])+y_pos(p_new["gpqa"]))/2:.1f}" '
        f'font-family="Arial, sans-serif" font-size="10.5" font-weight="bold" fill="#2D7A4D">'
        f'+10,8 п.п. при той же цене</text>'
    )

    # Точки
    for pt in points:
        xx = x_pos(pt["cost"])
        yy = y_pos(pt["gpqa"])
        cid = pt["cid"]
        if "Китай" in pt["eco"]:
            fill = "#C0392B"
            stroke = "#7B241C"
            r = 7.5
        elif "Ретро" in pt["eco"]:
            fill = "#8E9AA7"
            stroke = "#4E5B68"
            r = 6.5
        else:
            fill = "#1D5FA7"
            stroke = "#123C69"
            r = 7.0
        lines.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        dx, dy, anchor = label_offsets[cid]
        lines.append(
            f'<text x="{xx+dx:.1f}" y="{yy+dy:.1f}" text-anchor="{anchor}" class="pt-label">'
            f'{escape_xml(pt["label"])}</text>'
        )
        lines.append(
            f'<text x="{xx+dx:.1f}" y="{yy+dy+13:.1f}" text-anchor="{anchor}" class="pt-sub">'
            f'{pt["gpqa"]:.1f}% · ${pt["cost"]:.4f}</text>'
        )

    # Легенда
    lx, ly = ml + 12, mt + ph + 44
    lines.extend(
        [
            f'<circle cx="{lx}" cy="{ly}" r="6" fill="#C0392B"/>',
            f'<text x="{lx+12}" y="{ly+4}" class="legend">Китайские модели 2026</text>',
            f'<circle cx="{lx+180}" cy="{ly}" r="6" fill="#1D5FA7"/>',
            f'<text x="{lx+192}" y="{ly+4}" class="legend">Западные флагманы (осень 2026)</text>',
            f'<circle cx="{lx+420}" cy="{ly}" r="6" fill="#8E9AA7"/>',
            f'<text x="{lx+432}" y="{ly+4}" class="legend">Устаревшие модели (слайд 10, ред. 1)</text>',
        ]
    )

    lines.append("</svg>")
    out = RESEARCH_DIR / "price_quality_2026.svg"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def build_batch_cost_svg(candidates: list[dict[str, str]]) -> Path:
    """Строит сравнительную диаграмму стоимости обработки 128 требований модуля SL."""
    order = [
        "QWEN_FLASH",
        "DEEPSEEK_V41_FLASH",
        "GPT_OSS_120B_OLD",
        "DEEPSEEK_V4_PRO",
        "GLM_53",
        "QWEN4_MAX",
        "GEMINI_4_HYDROGEN",
        "GEMINI_4_CARBON",
        "GPT_62_SOL",
        "CLAUDE_SONNET_55",
        "GEMINI_31_PRO_OLD",
        "CLAUDE_OPUS_55",
        "GPT_62_ASTRA",
    ]
    cmap = {c["candidate_id"]: c for c in candidates}
    rows = []
    for cid in order:
        c = cmap[cid]
        p_in = float(c["price_in_usd_per_1m"])
        p_out = float(c["price_out_usd_per_1m"])
        p_cache = float(c["price_cache_hit_usd_per_1m"])
        no_cache, with_cache = calc_batch_cost(p_in, p_out, p_cache)
        rows.append((cid, c["model_name"], c["ecosystem"], no_cache, with_cache))

    width, height = 980, 590
    ml, mr, mt, mb = 250, 120, 82, 56
    pw = width - ml - mr
    row_h = 34
    max_val = max(r[3] for r in rows) * 1.05

    def x_bar(val: float) -> float:
        return (val / max_val) * pw

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>',
        '  .title { font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; fill: #1F344C; }',
        '  .subtitle { font-family: Arial, sans-serif; font-size: 12px; fill: #526375; }',
        '  .name { font-family: Arial, sans-serif; font-size: 11.5px; font-weight: bold; fill: #243240; }',
        '  .val { font-family: Arial, sans-serif; font-size: 11px; fill: #243240; }',
        '  .tick { font-family: Arial, sans-serif; font-size: 11px; fill: #526375; }',
        '</style>',
        f'<text x="32" y="34" class="title">Стоимость полного прогона пакета из 128 требований (модуль «Уровень сервиса»)</text>',
        f'<text x="32" y="54" class="subtitle">Расчёт на {BATCH_CALLS} вызовов (2 шага + 15% повторов) · Верхняя полоса: без кэша; нижняя: с prompt caching</text>',
    ]

    # Сетка X
    for v in (0, 2, 4, 6, 8, 10, 12, 14, 16):
        if v > max_val:
            continue
        xx = ml + x_bar(float(v))
        lines.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt + len(rows)*row_h}" stroke="#EAEFF4" stroke-width="1"/>')
        lines.append(f'<text x="{xx:.1f}" y="{mt + len(rows)*row_h + 18}" text-anchor="middle" class="tick">${v}</text>')

    for idx, (cid, name, eco, no_cache, with_cache) in enumerate(rows):
        yy = mt + idx * row_h
        if "Китай" in eco:
            c1, c2 = "#C0392B", "#E69A8D"
        elif "Ретро" in eco:
            c1, c2 = "#7F8C8D", "#B8C2C4"
        else:
            c1, c2 = "#1D5FA7", "#85B2E0"

        w1 = max(2.0, x_bar(no_cache))
        w2 = max(2.0, x_bar(with_cache))
        lines.append(f'<text x="{ml-12}" y="{yy+16}" text-anchor="end" class="name">{escape_xml(name)}</text>')
        lines.append(f'<rect x="{ml}" y="{yy+4}" width="{w1:.1f}" height="10" rx="2" fill="{c1}"/>')
        lines.append(f'<rect x="{ml}" y="{yy+16}" width="{w2:.1f}" height="9" rx="2" fill="{c2}"/>')
        lines.append(
            f'<text x="{ml + max(w1, w2) + 8:.1f}" y="{yy+16}" class="val">'
            f'${no_cache:.2f} (кэш ${with_cache:.2f})</text>'
        )

    ly = height - 22
    lines.extend(
        [
            f'<rect x="32" y="{ly-10}" width="14" height="10" fill="#C0392B"/>',
            f'<text x="52" y="{ly-1}" class="tick">Китайские LLM 2026 (экономия 5–16×)</text>',
            f'<rect x="300" y="{ly-10}" width="14" height="10" fill="#1D5FA7"/>',
            f'<text x="320" y="{ly-1}" class="tick">Западные API 2026</text>',
            f'<rect x="470" y="{ly-10}" width="14" height="10" fill="#7F8C8D"/>',
            f'<text x="490" y="{ly-1}" class="tick">База ред. 1 (слайд 10)</text>',
            f'<text x="{width-32}" y="{ly-1}" text-anchor="end" class="tick">Локально (Qwen3.8-27B / 35B-A3B): $0.00</text>',
        ]
    )

    lines.append("</svg>")
    out = RESEARCH_DIR / "batch_cost_comparison_2026.svg"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def verify_outputs(svg_paths: list[Path]) -> None:
    for p in svg_paths:
        ET.parse(p)
        print(f"SVG OK: {p.relative_to(ROOT)} ({p.stat().st_size} B)")

    for md in [ROOT / "README.md", *RESEARCH_DIR.glob("*.md")]:
        if not md.exists():
            raise FileNotFoundError(f"Отсутствует файл документа: {md}")
        print(f"DOC OK: {md.relative_to(ROOT)} ({md.stat().st_size} B)")


def main() -> None:
    candidates, benchmarks = load_data()
    print(f"Загружено кандидатов: {len(candidates)}, записей бенчмарков: {len(benchmarks)}")
    print("\n--- РАСЧЁТ СТОИМОСТИ (1 ЗАПРОС И ПАКЕТ 128 ТРЕБОВАНИЙ) ---")
    for c in candidates:
        p_in = float(c["price_in_usd_per_1m"])
        p_out = float(c["price_out_usd_per_1m"])
        p_cache = float(c["price_cache_hit_usd_per_1m"])
        req = calc_single_request_cost(p_in, p_out)
        no_cache, with_cache = calc_batch_cost(p_in, p_out, p_cache)
        print(
            f"{c['model_name']:34s} | 1 req: ${req:.5f} | 128 reqs no-cache: ${no_cache:.2f} | with-cache: ${with_cache:.2f}"
        )

    s1 = build_price_quality_svg(candidates, benchmarks)
    s2 = build_batch_cost_svg(candidates)
    verify_outputs([s1, s2])
    print("\nВсе расчёты, графики и проверки выполнены успешно.")


if __name__ == "__main__":
    main()
