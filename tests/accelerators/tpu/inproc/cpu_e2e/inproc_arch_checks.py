# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""The pass rules of one architecture case (``inproc_arch_child.py``'s RESULT). ``evaluate(res)`` returns
[(check, ok, detail)]."""

from __future__ import annotations

import json

from inproc_tiny_models import SPECS

FAMILY = {
    "Qwen3_5ForCausalLM": "qwen3_5 direct",
    "Qwen3ForCausalLM": "vllm loaders",
    "LlamaForCausalLM": "vllm loaders",
    "Gemma3ForCausalLM": "vllm loaders",
    "MistralForCausalLM": "vllm loaders",
    "Qwen2ForCausalLM": "vllm loaders",
}
KV_KINDS = {
    "Qwen3_5ForCausalLM": ["FullAttentionSpec", "MambaSpec"],
    "Gemma3ForCausalLM": ["FullAttentionSpec", "SlidingWindowSpec"],
}


def _brief(x, n: int = 700) -> str:
    return json.dumps(x, default=str)[:n]


def _unchanged(m: dict) -> bool:
    return m.get("engine_changed_params") == 0


def _oracle_ok(x: dict) -> bool:
    return (x.get("oracle") or {}).get("n_mismatched") == 0


def evaluate(res: dict) -> list:
    key = res.get("key")
    spec = SPECS[key]
    arch = spec["architectures"][0]
    out: list = []

    def check(name: str, ok: bool, detail=None) -> None:
        out.append((f"[{key}] {name}", bool(ok), "" if ok else _brief(detail)))

    if res.get("error"):
        check("the case ran", False, {"error": res.get("error"), "tb": res.get("traceback")})
        return out
    if res.get("mode") == "build":
        if res.get("add_layout"):
            s0 = res.get("sync0") or {}
            err = str(s0.get("err"))
            failed = "INTEGRITY FAILED" in err or (
                "parity FAILED" in err and (s0.get("oracle") or {}).get("n_mismatched") == 0
            )
            check(
                "a layout added to the table without its test builds, and its first sync still FAILS CLOSED",
                res.get("built") is not None and failed,
                {"built": res.get("built"), "sync0": s0},
            )
        else:
            err = str(res.get("build_error"))
            check(
                f"REFUSED at build with a clear message ({arch})",
                "UNTESTED LAYOUT" in err
                and arch in err
                and "Add a test and a table entry" in err
                and res.get("built") is None,
                {"build_error": res.get("build_error"), "built": res.get("built")},
            )
        return out
    if res.get("build_error"):
        check(
            "the tested layout builds", False, {"build_error": res.get("build_error"), "lines": res.get("build_lines")}
        )
        return out
    mode = FAMILY[arch]
    direct = mode.endswith("direct")
    e = res.get("engine") or {}
    kv = res.get("kv_capacity") or {}
    compat_line = str(res.get("compat_line"))
    check(
        f"BUILD: the real in-process vLLM engine (InprocClient / UniProcExecutor) serves vLLM's own {arch}, the build "
        "line OK, "
        f"KV capacity ok, the tested layout {mode!r} named in the compat line",
        e.get("client") == "InprocClient"
        and e.get("executor") == "UniProcExecutor"
        and e.get("model_class") == arch
        and str(res.get("engine_line") or "").endswith("OK")
        and kv.get("ok") is True
        and e.get("installer_family") == mode
        and e.get("kv_spec_kinds") == KV_KINDS.get(arch, ["FullAttentionSpec"])
        and '"tested": true' in compat_line
        and f'"family": "{mode}"' in compat_line,
        {
            "engine": {k: v for k, v in e.items() if k != "param_names"},
            "kv": kv,
            "line": res.get("engine_line"),
            "compat": compat_line,
        },
    )
    if res.get("warm_line") is not None:
        w = str(res.get("warm_line"))
        check("WARM-UP: the recompile guard installed and armed", '"armed": true' in w and '"installed": true' in w, w)
    g, ga = res.get("generate") or [], res.get("generate_after") or []
    check(
        "SERVE: 4 requests through verl's vLLMHttpServer.generate on the in-process adapter at build, 2 more after "
        "every mutant",
        len(g) == 4 and len(ga) == 2 and all(x["ok"] and x["logprobs"] for x in g + ga),
        {"build": g, "after": ga},
    )
    tied = bool(spec.get("tie_word_embeddings"))
    names = e.get("param_names") or []
    check(
        f"TIE: the engine's LM head is {'tied to the embedding' if tied else 'its own parameter'}, as the config says",
        e.get("tied_lm_head") is tied and (("lm_head.weight" in names) is (not tied)),
        {"tied_lm_head": e.get("tied_lm_head")},
    )
    ss = res.get("syncs") or []
    ver = {0: "step 0", 1: "step 1", 5: "every 5 steps"}
    n = e.get("params")
    good = len(ss) == 6
    for s in ss:
        par, cov = s.get("parity") or {}, s.get("coverage") or {}
        good = (
            good
            and s.get("err") is None
            and s.get("mode") == mode
            and _oracle_ok(s)
            and s.get("health_line") is not None
        )
        if s["step"] in ver:
            good = (
                good
                and s.get("verified") == ver[s["step"]]
                and "PARITY EXACT" in str(s.get("parity_line"))
                and cov.get("ok") is True
                and cov.get("written") == n
                and par.get("post_mismatched") == 0
            )
        else:
            good = good and s.get("verified") is None and s.get("parity_line") is None
    check(
        f"SYNCS 0-5 (real AdamW steps between them): install mode {mode!r}, the engine equals the oracle on every "
        "parameter at "
        "every sync; PARITY EXACT + COVERAGE OK n/n at the verified syncs 0, 1, 5; 2, 3, 4 unverified",
        good,
        [{k: s.get(k) for k in ("step", "err", "mode", "verified", "oracle", "parity_line", "coverage")} for s in ss],
    )
    mut = res.get("mutants") or {}
    for label, what in (
        ("m6_shard_unverified", "a fused shard missing at an unverified sync (6)"),
        ("m8_whole_unverified", "a whole unfused tensor missing at an unverified sync (8)"),
        ("m10_shard_verified", "a fused shard missing at a verified sync (10)"),
        ("m25_weakening_shard", "a fused shard missing with the old weakening variables set (25)"),
    ):
        m = mut.get(label) or {}
        dropped = (m.get("what") or {}).get("drop") or ["?"]
        check(
            f"MUTANT {what}: FAILS CLOSED before the install ('EXPORT CHANGED ... missing {dropped[0]}'), the engine "
            "untouched",
            "EXPORT CHANGED" in str(m.get("err"))
            and dropped[0] in str(m.get("err"))
            and _unchanged(m)
            and m.get("parity_line") is None,
            {k: m.get(k) for k in ("err", "engine_changed_params", "parity_line")},
        )
    restores = {k: v for k, v in mut.items() if k.endswith("_restore")}
    check(
        f"RESTORE: the good sync after each mutant installs and equals the oracle ({len(restores)} syncs)",
        len(restores) >= 5 and all(v.get("err") is None and _oracle_ok(v) for v in restores.values()),
        {k: {"err": v.get("err"), "oracle": v.get("oracle")} for k, v in restores.items()},
    )
    if tied:
        m = mut.get("m15_tie_mismatch") or {}
        check(
            "TIE MISMATCH: an LM head that differs from the embedding, sent to a tied engine, FAILS CLOSED (PARITY "
            "FAILED)",
            "PARITY FAILED" in str(m.get("parity_line"))
            and m.get("err") is not None
            and (direct or "tied to model.embed_tokens.weight" in str(m.get("parity_line"))),
            {k: m.get(k) for k in ("err", "parity_line")},
        )
    m = mut.get("m20_weakening_full") or {}
    check(
        "WEAKENING VARIABLES set after the build with a full export: not read; the verified sync reads PARITY EXACT",
        m.get("err") is None and "PARITY EXACT" in str(m.get("parity_line")),
        {k: m.get(k) for k in ("err", "parity_line")},
    )
    if not direct:
        cor = res.get("corrupt") or {}
        check(
            "CORRUPTION: one element of a fused qkv_proj (and of the embedding) is detected by the package's own "
            "packed parity",
            len(cor) == 2 and all(v.get("detected") for v in cor.values()),
            cor,
        )
        fl = res.get("tpu_flip_sim") or {}
        check(
            "TPU LAYOUT: with vllm-torchtpu's transposed (n_in, n_out) dense weights simulated in place, the packed "
            "parity reads EXACT",
            (fl.get("flipped_modules") or 0) > 0
            and fl.get("post_mismatched_n") == 0
            and fl.get("unresolved_n") == 0
            and (res.get("after_flip_restore") or {}).get("n_mismatched") == 0,
            {"flip": fl, "restore": res.get("after_flip_restore")},
        )
    fs, fr = res.get("first_sync_shard") or {}, res.get("first_sync_restore") or {}
    if direct:
        ok = "direct install raised" in str(fs.get("err"))
    else:
        ok = (
            "parity FAILED" in str(fs.get("err"))
            and "qkv_proj" in str(fs.get("parity_line"))
            and "PARITY FAILED" in str(fs.get("parity_line"))
        )
    check(
        "FRESH PROCESS: a first sync (nothing pinned yet) with a fused shard missing FAILS CLOSED in the install / "
        "parity itself; "
        "the next good sync equals the oracle",
        ok and fr.get("err") is None and _oracle_ok(fr),
        {"first": {k: fs.get(k) for k in ("err", "parity_line", "engine_changed_params")}, "restore": fr.get("oracle")},
    )
    nh = res.get("first_sync_no_head") or {}
    n1, n2, n3 = nh.get("first") or {}, nh.get("second") or {}, nh.get("full") or {}
    brief_nh = {
        k: {
            f: (x or {}).get(f)
            for f in ("err", "parity_line", "coverage_line", "export_pin", "engine_vs_full_export", "oracle")
        }
        for k, x in nh.items()
    }
    if tied:
        check(
            "CONTROL (tied): a fresh process's export without the lm_head alias installs at all three syncs and equals "
            "the oracle",
            len(nh) == 3
            and all(x.get("err") is None for x in (n1, n2, n3))
            and all(
                (x.get("engine_vs_full_export") or x.get("oracle") or {}).get("n_mismatched") == 0 for x in (n1, n2, n3)
            ),
            brief_nh,
        )
    else:
        why = "coverage: 1 engine parameter(s) never written" if direct else "parity FAILED"

        def names_head(x: dict) -> bool:
            return why in str(x.get("err")) and "lm_head.weight" in str(x.get("err")) + str(x.get("parity_line"))

        def verdict_failed(x: dict) -> bool:
            return "PARITY FAILED" in str(x.get("parity_line")) and "PARITY EXACT" not in str(x.get("parity_line"))

        check(
            "FRESH PROCESS, LM HEAD NOT SENT (untied): the first sync FAILS CLOSED naming lm_head.weight, its parity "
            "line reads "
            "PARITY FAILED, so does the next sync (verified until one passes); the full export then installs, equals "
            "the oracle "
            "and is the one pinned",
            names_head(n1)
            and names_head(n2)
            and verdict_failed(n1)
            and verdict_failed(n2)
            and n3.get("err") is None
            and _oracle_ok(n3)
            and n3.get("export_pin") == 42,
            brief_nh,
        )
    rp, aw = res.get("rt_coverage_report") or {}, res.get("rt_allow_widened") or {}
    rpb, awb = rp.get("bad") or {}, aw.get("bad") or {}
    check(
        "COVERAGE variable set after the build + a whole tensor missing at a fresh first sync: still FAILS CLOSED; the "
        "full "
        "export then installs",
        (
            ("coverage: 1 engine parameter(s) never written" in str(rpb.get("err")))
            if direct
            else ("parity FAILED" in str(rpb.get("err")))
        )
        and (rp.get("good") or {}).get("err") is None
        and _oracle_ok(rp.get("good") or {}),
        {
            "bad": {k: rpb.get(k) for k in ("err", "parity_line", "coverage_line")},
            "good": (rp.get("good") or {}).get("oracle"),
        },
    )
    check(
        "ALLOWLIST variable set after the build + a fused shard missing at a fresh first sync: not read, FAILS CLOSED; "
        "the "
        "full export then installs",
        (("direct install raised" in str(awb.get("err"))) if direct else ("parity FAILED" in str(awb.get("err"))))
        and (aw.get("good") or {}).get("err") is None
        and _oracle_ok(aw.get("good") or {}),
        {
            "bad": {k: awb.get(k) for k in ("err", "parity_line", "coverage_line")},
            "good": (aw.get("good") or {}).get("oracle"),
        },
    )
    cs = res.get("constructor_switches") or {}
    check(
        "WEAKENING VARIABLES AT THE CONSTRUCTOR: not read; the constructor names each in one warning",
        len(cs) == 3 and all(v.get("constructed") and k.split("=")[0] in str(v.get("warning")) for k, v in cs.items()),
        cs,
    )
    return out
