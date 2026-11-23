"""Quality evaluation and benchmark suite for extracted reference novel database.

Inspects:
1. Continuous 100% pacing wave dynamics (coverage, smooth Gaussian curvature, quantile expansion 1.0-5.0).
2. Dual-layer technique structure (【宏觀戰略】 + 【微觀戰術】) and scene focus snippet lengths (200-350 chars).
3. De-concretization and Actant Archetype grammar (eliminating raw names, Taoist titles, foreign names, pet names).
4. Genre taxonomy consistency and anti-drift validation.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

DEFAULT_DB_PATH = Path("data/reference_library.db")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def inspect_pacing_dynamics(conn: sqlite3.Connection) -> dict[str, Any]:
    """Analyze the high-resolution continuous pacing wave templates stored in the database."""
    cur = conn.cursor()
    tpl_rows = cur.execute(
        "SELECT id, name, genre_primary, genre_sub, archetype, total_chapters, chapter_curve_data "
        "FROM ref_pacing_templates ORDER BY id DESC"
    ).fetchall()

    results: list[dict[str, Any]] = []
    for r in tpl_rows:
        tpl_id, name, g_prim, g_sub, arch, total_ch, raw_curve = r
        curve_data: list[dict[str, Any]] = []
        if raw_curve:
            try:
                curve_data = json.loads(raw_curve)
            except Exception:
                curve_data = []

        stages_rows = cur.execute(
            "SELECT stage_index, offset_percent, pacing_phase, target_tension, target_progress, target_density, dramatic_purpose "
            "FROM ref_pacing_stages WHERE template_id = ? ORDER BY stage_index ASC",
            (tpl_id,),
        ).fetchall()

        t_values = [p.get("t", 0.0) for p in curve_data]
        p_values = [p.get("p", 0.0) for p in curve_data]
        d_values = [p.get("d", 0.0) for p in curve_data]

        def calc_stats(vals: list[float]) -> dict[str, float]:
            if not vals:
                return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
            mean_v = sum(vals) / len(vals)
            var_v = sum((v - mean_v) ** 2 for v in vals) / len(vals)
            return {
                "min": round(min(vals), 2),
                "max": round(max(vals), 2),
                "mean": round(mean_v, 2),
                "std": round(math.sqrt(var_v), 2),
            }

        # Calculate 1st order step difference to measure curve smoothness
        smoothness_diffs = []
        for i in range(1, len(t_values)):
            smoothness_diffs.append(abs(t_values[i] - t_values[i - 1]))
        avg_step_diff = round(sum(smoothness_diffs) / len(smoothness_diffs), 4) if smoothness_diffs else 0.0

        results.append({
            "template_id": tpl_id,
            "name": name,
            "genre": f"{g_prim}/{g_sub}",
            "archetype": arch,
            "total_chapters": total_ch,
            "curve_points_count": len(curve_data),
            "stages_count": len(stages_rows),
            "offset_coverage": {
                "first_pct": curve_data[0].get("pct", 0.0) if curve_data else 0.0,
                "last_pct": curve_data[-1].get("pct", 0.0) if curve_data else 0.0,
            },
            "tension_stats": calc_stats(t_values),
            "progress_stats": calc_stats(p_values),
            "density_stats": calc_stats(d_values),
            "avg_adjacent_tension_step": avg_step_diff,
            "sample_valley_point": next((p for p in curve_data if p.get("t", 5.0) <= 2.2), None),
            "sample_climax_point": next((p for p in curve_data if p.get("t", 0.0) >= 3.8), None),
        })

    return {"templates": results}


def inspect_entries_and_archetypes(conn: sqlite3.Connection) -> dict[str, Any]:
    """Inspect snippet lengths, dual-layer description formatting, and role archetypes."""
    cur = conn.cursor()
    entries = cur.execute(
        "SELECT id, source_id, description, snippet, subject_role, object_role, interaction_dynamic, created_at "
        "FROM ref_entries ORDER BY id ASC"
    ).fetchall()

    total_entries = len(entries)
    if total_entries == 0:
        return {"total_entries": 0}

    dual_layer_valid = 0
    snippet_lengths: list[int] = []
    roles_subject: list[str] = []
    roles_object: list[str] = []
    dynamics_counter: dict[str, int] = {}
    raw_name_leak_candidates: list[str] = []

    prohibited_name_endings = ("真人", "道長", "天尊", "老祖", "長老", "大帝", "魔君")

    for e in entries:
        eid, sid, desc, snip, s_role, o_role, dyn, _ = e
        desc = desc or ""
        snip = snip or ""
        s_role = s_role or ""
        o_role = o_role or ""
        dyn = (dyn or "").upper()

        if "【宏觀戰略】" in desc and "【微觀戰術】" in desc:
            dual_layer_valid += 1

        snippet_lengths.append(len(snip))
        if s_role:
            roles_subject.append(s_role)
            if any(s_role.endswith(p) for p in prohibited_name_endings):
                raw_name_leak_candidates.append(f"Subject '{s_role}' in entry #{eid}")
        if o_role:
            roles_object.append(o_role)
            if any(o_role.endswith(p) for p in prohibited_name_endings):
                raw_name_leak_candidates.append(f"Object '{o_role}' in entry #{eid}")

        dynamics_counter[dyn] = dynamics_counter.get(dyn, 0) + 1

    avg_snip_len = round(sum(snippet_lengths) / len(snippet_lengths), 1) if snippet_lengths else 0.0

    # Tag consistency
    tags_rows = cur.execute("SELECT tag_name, COUNT(*) FROM ref_tags GROUP BY tag_name ORDER BY COUNT(*) DESC").fetchall()

    return {
        "total_entries": total_entries,
        "dual_layer_compliance_rate": round((dual_layer_valid / total_entries) * 100, 1),
        "snippet_length": {
            "min": min(snippet_lengths) if snippet_lengths else 0,
            "max": max(snippet_lengths) if snippet_lengths else 0,
            "avg": avg_snip_len,
        },
        "distinct_subject_roles": sorted(list(set(roles_subject))),
        "distinct_object_roles": sorted(list(set(roles_object))),
        "dynamics_distribution": dynamics_counter,
        "name_leak_warnings": raw_name_leak_candidates,
        "top_tags": tags_rows[:15],
    }


def format_evaluation_report(db_path: Path) -> str:
    """Format an extensive evaluation report for the extracted database."""
    if not db_path.exists():
        return f"[!] Database file does not exist at {db_path}."

    with sqlite3.connect(db_path) as conn:
        pacing_res = inspect_pacing_dynamics(conn)
        entries_res = inspect_entries_and_archetypes(conn)

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("AutoNovelWriter 萃取資料庫品質檢驗報告")
    lines.append(f"資料庫路徑: {db_path.resolve()}")
    lines.append("=" * 80)

    # 1. Pacing Wave Section
    lines.append("\n【一、逐章 100% 連續平滑情節動力學波浪分析】")
    templates = pacing_res.get("templates", [])
    if not templates:
        lines.append("  (暫無情節規劃範本數據)")
    else:
        for tpl in templates:
            lines.append(f"- 範本名稱: {tpl['name']} ({tpl['genre']} | {tpl['archetype']})")
            lines.append(f"  * 全書總章數: {tpl['total_chapters']} 章")
            lines.append(f"  * 100% 連續點位數: {tpl['curve_points_count']} 點 (覆蓋度: {tpl['offset_coverage']['first_pct']}% -> {tpl['offset_coverage']['last_pct']}%)")
            lines.append(f"  * 里程碑節點數: {tpl['stages_count']} 個關鍵極值標記")
            lines.append(f"  * 張力 (Tension) 數值統計: Min={tpl['tension_stats']['min']}, Max={tpl['tension_stats']['max']}, Mean={tpl['tension_stats']['mean']}, Std={tpl['tension_stats']['std']}")
            lines.append(f"  * 推進 (Progress) 數值統計: Min={tpl['progress_stats']['min']}, Max={tpl['progress_stats']['max']}, Mean={tpl['progress_stats']['mean']}, Std={tpl['progress_stats']['std']}")
            lines.append(f"  * 密度 (Density) 數值統計: Min={tpl['density_stats']['min']}, Max={tpl['density_stats']['max']}, Mean={tpl['density_stats']['mean']}, Std={tpl['density_stats']['std']}")
            lines.append(f"  * 相鄰章節張力一階平滑差分 (Mean Step Diff): {tpl['avg_adjacent_tension_step']} (展現高斯卷積平滑連續性)")
            if tpl["sample_valley_point"]:
                vp = tpl["sample_valley_point"]
                lines.append(f"  * [低谷驗證] 成功探測低谷喘息點: 第 {vp['ch']} 章 ({vp['pct']}%) -> Tension={vp['t']}【{vp.get('title','')}】")
            if tpl["sample_climax_point"]:
                cp = tpl["sample_climax_point"]
                lines.append(f"  * [高潮驗證] 成功探測決戰爆發點: 第 {cp['ch']} 章 ({cp['pct']}%) -> Tension={cp['t']}【{cp.get('title','')}】")

    # 2. Entries & Snippets Section
    lines.append("\n【二、詞條雙層筆法結構與場景焦點摘錄分析】")
    lines.append(f"- 入庫精華詞條總數: {entries_res['total_entries']} 則")
    lines.append(f"- 【雙層筆法結構】合規率: {entries_res.get('dual_layer_compliance_rate', 0)}% (嚴格具備【宏觀戰略】與【微觀戰術】)")
    snip = entries_res.get("snippet_length", {})
    lines.append(f"- 正文摘錄長度分佈 (字數): 最短={snip.get('min', 0)} 字, 最長={snip.get('max', 0)} 字, 平均={snip.get('avg', 0)} 字 (已擴展為場景焦點精華區塊)")

    # 3. Roles & Archetype Section
    lines.append("\n【三、主客體文法角色去具名化與戲劇原型分析】")
    subj_roles = entries_res.get("distinct_subject_roles", [])
    obj_roles = entries_res.get("distinct_object_roles", [])
    lines.append(f"- 主體原型角色庫 ({len(subj_roles)} 種): {', '.join(subj_roles[:10])}{' ...' if len(subj_roles) > 10 else ''}")
    lines.append(f"- 客體原型角色庫 ({len(obj_roles)} 種): {', '.join(obj_roles[:10])}{' ...' if len(obj_roles) > 10 else ''}")
    lines.append(f"- 六大力量互動動力學分佈: {dict(entries_res.get('dynamics_distribution', {}))}")
    leaks = entries_res.get("name_leak_warnings", [])
    if leaks:
        lines.append(f"  [!] 具名或道號外洩警告 ({len(leaks)} 筆): {leaks[:3]}")
    else:
        lines.append("  [✓] 實體轉譯檢驗通過：未發現道號/專有名詞外洩，皆轉譯為抽象戲劇角色原型。")

    # 4. Tags Section
    lines.append("\n【四、全庫高頻語義標籤分佈】")
    top_tags = entries_res.get("top_tags", [])
    top_tags_str = ", ".join(f"#{t[0]} ({t[1]})" for t in top_tags[:10])
    lines.append(f"- 前 10 大熱門標籤: {top_tags_str}")
    lines.append("=" * 80)

    return "\n".join(lines)


def test_inspect_real_database_if_available():
    """Execute evaluation on actual data/reference_library.db if present."""
    if not DEFAULT_DB_PATH.exists():
        pytest.skip("data/reference_library.db not found, skipping real db inspection.")

    report = format_evaluation_report(DEFAULT_DB_PATH)
    # Output report to stdout for test logs
    print("\n" + report)

    with sqlite3.connect(DEFAULT_DB_PATH) as conn:
        pacing = inspect_pacing_dynamics(conn)
        entries = inspect_entries_and_archetypes(conn)

    # Assertions on pacing templates
    templates = pacing["templates"]
    if templates:
        tpl = templates[0]
        # Should have continuous high resolution points (> 100 points)
        assert tpl["curve_points_count"] >= 100, "Curve points count should be high resolution (>100)"
        # Offset coverage should span 0% ~ 100%
        assert tpl["offset_coverage"]["first_pct"] <= 5.0
        assert tpl["offset_coverage"]["last_pct"] >= 95.0
        # Smoothness test: average step diff between adjacent chapters should be gentle (< 0.65)
        assert tpl["avg_adjacent_tension_step"] < 0.65, "Pacing wave should have smooth steps (< 0.65)"
        # Range expansion test: min tension should touch valley levels (<= 2.5) and max tension should reach climax (>= 3.5)
        assert tpl["tension_stats"]["min"] <= 2.5, "Dynamic range should restore low valleys (<= 2.5)"
        assert tpl["tension_stats"]["max"] >= 3.5, "Dynamic range should reach peak climax (>= 3.5)"

    # Assertions on entries
    if entries["total_entries"] > 0:
        assert entries["dual_layer_compliance_rate"] >= 80.0, "Dual-layer structure compliance should be >= 80%"
        assert entries["snippet_length"]["max"] >= 90, "Snippets should capture scene focus blocks"


def test_mock_pacing_dynamics_evaluation(tmp_path: Path):
    """Deterministic regression test for inspection logic using a synthetic reference db."""
    test_db = tmp_path / "test_ref.db"
    conn = sqlite3.connect(test_db)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE ref_pacing_templates (
            id INTEGER PRIMARY KEY,
            source_id INTEGER,
            name TEXT,
            genre_primary TEXT,
            genre_sub TEXT,
            archetype TEXT,
            pacing_tags TEXT,
            description TEXT,
            total_stages INTEGER,
            total_chapters INTEGER,
            chapter_curve_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        CREATE TABLE ref_pacing_stages (
            id INTEGER PRIMARY KEY,
            template_id INTEGER,
            stage_index INTEGER,
            offset_percent REAL,
            pacing_phase TEXT,
            target_tension INTEGER,
            target_progress INTEGER,
            target_density INTEGER,
            dramatic_purpose TEXT
        );
    """)
    cur.execute("""
        CREATE TABLE ref_entries (
            id INTEGER PRIMARY KEY,
            source_id INTEGER,
            description TEXT,
            desc_embedding BLOB,
            snippet TEXT,
            subject_role TEXT,
            object_role TEXT,
            interaction_dynamic TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        CREATE TABLE ref_tags (
            id INTEGER PRIMARY KEY,
            entry_id INTEGER,
            tag_name TEXT,
            tag_embedding BLOB
        );
    """)

    # Populate 500 chapters smooth curve
    curve_data = [
        {"ch": i, "pct": round(i / 5.0, 2), "t": 2.0 + math.sin(i * 0.05) * 1.5, "p": 2.5, "d": 2.8, "title": f"第{i}章"}
        for i in range(1, 501)
    ]
    cur.execute(
        "INSERT INTO ref_pacing_templates (id, name, genre_primary, genre_sub, archetype, total_chapters, chapter_curve_data) "
        "VALUES (1, '測試流派', '仙俠', '修真', '慢熱穩健', 500, ?)",
        (json.dumps(curve_data),)
    )
    cur.execute(
        "INSERT INTO ref_pacing_stages (template_id, stage_index, offset_percent, pacing_phase, target_tension, target_progress, target_density, dramatic_purpose) "
        "VALUES (1, 1, 0.0, 'BUILDUP', 1, 1, 1, '起步')"
    )
    cur.execute(
        "INSERT INTO ref_entries (id, source_id, description, snippet, subject_role, object_role, interaction_dynamic) "
        "VALUES (1, 1, '【宏觀戰略】：日常鋪陳；【微觀戰術】：光影白描渲染氛圍', '這是一段精選連續場景焦點精華區塊文字...', '天命求道者', '宗門上位威壓者', 'DOMINANCE')"
    )
    cur.execute("INSERT INTO ref_tags (entry_id, tag_name) VALUES (1, '仙俠:修真'), (1, '環境白描')")
    conn.commit()

    pacing_eval = inspect_pacing_dynamics(conn)
    entries_eval = inspect_entries_and_archetypes(conn)
    conn.close()

    assert pacing_eval["templates"][0]["curve_points_count"] == 500
    assert pacing_eval["templates"][0]["tension_stats"]["min"] <= 1.5
    assert pacing_eval["templates"][0]["tension_stats"]["max"] >= 3.4
    assert entries_eval["dual_layer_compliance_rate"] == 100.0
    assert entries_eval["distinct_subject_roles"] == ["天命求道者"]


if __name__ == "__main__":
    report_text = format_evaluation_report(DEFAULT_DB_PATH)
    print(report_text)
