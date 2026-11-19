# -*- coding: utf-8 -*-
"""Builds every kind of job into the ComfyUI graph that would be posted, without a GPU or ComfyUI, and checks it.

  python selftest_graph.py                      the built-in sample batch (every job type) + broken graphs
  python selftest_graph.py --manifest m.json    also build that manifest's jobs (files are not read: sizes are faked)
  python selftest_graph.py --toybox <minimax-h3-colab/toybox>
                                                also compare every graph with what H3 Toybox's own build_generate()
                                                / ph3_graph() make from the raw WF.json (the code this was ported from)

Checks: no VHS / preview node classes; every link points at an existing node; reference slots match the
mentioned inputs; intensity off removes the three action LoRAs and rewires their consumers; seconds, size,
steps and seed land in the right nodes; extensions wire Motion Context. Then it breaks graphs on purpose
(dangling link, VHS node, LoRA left in, unwired reference) and fails if any of those is NOT caught.
"""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "remote"))
import h3graph as G  # noqa: E402

IMG, VID, AUD = (".png", ".jpg", ".jpeg", ".webp", ".bmp"), (".mp4", ".mov", ".webm", ".mkv", ".avi"), \
    (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")

SAMPLE = {"jobs": [  # the manifest.md example, plus every mode / intensity / size variant
    {"id": "cast_lin", "type": "image", "prompt": "A clean character sheet photo", "aspect": "9:16", "size": "2K", "seed": 1},
    {"id": "frame01", "type": "image", "prompt": "Edit <Picture 1>. The girl from @圖片1 sits at the reading table of @圖片2",
     "inputs": ["inputs/lin.png", "inputs/library.png"], "aspect": "9:16", "size": "2K"},
    {"id": "frame02", "type": "image", "prompt": "Put both people in one room", "inputs": ["inputs/a.png", "inputs/b.png", "inputs/c.wav"],
     "aspect": "auto", "size": "1.6K", "seed": 7},
    {"id": "shot01", "type": "video", "mode": "ref", "prompt": "<Subject 1> is the girl from @圖片1 in @圖片2, voice @音訊1",
     "inputs": ["inputs/lin.png", "inputs/library.png", "inputs/lin.wav"], "seconds": 10, "aspect": "9:16",
     "resolution": "720P", "intensity": "off", "fast": True, "seed": 42},
    {"id": "shot01v", "type": "video", "mode": "ref", "prompt": "@影片1 motion, @圖片2 is the face, @圖片1 the room",
     "inputs": ["inputs/room.png", "inputs/face.png", "inputs/move.mp4", "inputs/unused.png"], "seconds": 6,
     "aspect": "16:9", "resolution": "480P", "intensity": "high", "fast": False},
    {"id": "shot01b", "type": "video", "mode": "ref", "extend": "shot01", "prompt": "she keeps reading @圖片1",
     "inputs": ["inputs/lin.png"], "seconds": 8, "aspect": "auto", "resolution": "720P", "intensity": "low"},
    {"id": "shot02", "type": "video", "mode": "frames", "first": "@frame01", "last": "inputs/end.png",
     "prompt": "she stands up", "seconds": 6, "resolution": "720P", "intensity": "mid"},
    {"id": "shot03", "type": "video", "mode": "frames", "first": "inputs/end.png", "prompt": "i2v", "seconds": 4, "aspect": "auto"},
    {"id": "shot04", "type": "video", "mode": "first_last", "last": "inputs/end.png", "prompt": "l2v", "seconds": 5, "aspect": "1:1"},
    {"id": "shot05", "type": "video", "mode": "text", "prompt": "empty street at dawn", "seconds": 3, "aspect": "21:9",
     "resolution": "1080P"},
    {"id": "shot06", "type": "video", "mode": "ref", "prompt": "<Subject 1> is the girl from @圖片1", "inputs": ["inputs/lin.png"],
     "seconds": 5, "aspect": "16:9", "resolution": "480P", "intensity": "low", "model_set": "official"},
    {"id": "shot07", "type": "video", "mode": "frames", "first": "inputs/lin.png", "prompt": "she turns to the window",
     "seconds": 5, "aspect": "16:9", "resolution": "480P", "model_set": "official"},
]}


def fake(rel):
    """A media item for a path without reading it: kind from the extension, a portrait size for pictures."""
    rel = str(rel)
    ext = Path(rel.lstrip("@")).suffix.lower() or ".png"  # "@frame01" = an earlier picture
    kind = "image" if ext in IMG else "video" if ext in VID else "audio" if ext in AUD else None
    if kind is None:
        raise ValueError(f"unknown file type: {rel}")
    w, h = (720, 1280) if kind == "image" else (1280, 720)
    return {"kind": kind, "name": "h3skill_" + Path(rel.lstrip("@")).name.replace(".", "_") + ext,
            "width": w, "height": h, "has_audio": kind == "video"}


FAILS = []
TOYBOX = None


def expect(cond, msg):
    if not cond:
        FAILS.append(msg)
    return cond


def build(j, rng):
    if j["type"] == "image":
        plan = G.plan_picture(j, [fake(r) for r in j.get("inputs", [])], rng)
        return plan, G.picture_graph(G.load_template(G.PICTURE_TEMPLATE), plan), None
    kw = {}
    mode = G.VIDEO_MODES.get(j.get("mode", "ref"))
    if mode == "ref":
        kw["inputs"] = [fake(r) for r in j.get("inputs", [])]
        if j.get("extend"):
            kw["extend"] = {"width": 1280, "height": 720, "has_audio": True, "last_frame": fake("last_frame.jpg")}
    elif mode == "frames":
        kw = {"first": fake(j["first"]) if j.get("first") else None, "last": fake(j["last"]) if j.get("last") else None}
    if j.get("drive_audio"):  # audio lock: the recording drives the clip (length faked like every other file)
        kw["drive"] = dict(fake(j["drive_audio"]), seconds=float(j.get("seconds") or 10.0))
    plan = G.plan_video(j, rng=rng, **kw)
    tail = f"h3skill_tail_{j['id']}.mp4" if plan["extend"] else None
    return plan, G.video_graph(G.load_template(G.VIDEO_TEMPLATE), plan, tail), tail


def check_job(j, plan, g, tail, template):
    jid = j["id"]
    classes = {n["class_type"] for n in g.values()}
    expect(not [c for c in classes if any(f in c for f in G.FORBIDDEN_CLASS)], f"{jid}: forbidden class in {classes}")
    for nid, n in g.items():
        for port, v in n["inputs"].items():
            if G.is_link(v):
                expect(v[0] in g, f"{jid}: {nid}.{port} -> missing {v[0]}")
    if j["type"] == "image":
        probs = G.check_picture(g, plan)
        expect(not probs, f"{jid}: {probs}")
        expect(g[G.N_IMG_ENC]["inputs"]["width"] == plan["width"], f"{jid}: width")
        return f"{plan['task']:5} {plan['width']}x{plan['height']} refs={len(plan['refs'])} seed={plan['seed']}"
    probs = G.check_video(g, plan, tail)
    expect(not probs, f"{jid}: {probs}")
    ci = g[G.N_COND]["inputs"]
    expect(g[G.N_SECONDS]["inputs"]["value"] == float(j.get("seconds", 10)), f"{jid}: seconds not in node {G.N_SECONDS}")
    # audio lock conditions a low-res canvas and samples in two plain passes (check_video covers its steps)
    size = G.drive_low_size(plan["width"], plan["height"]) if plan.get("drive") else (plan["width"], plan["height"])
    expect((ci["width"], ci["height"]) == size, f"{jid}: size not in node {G.N_COND}")
    if not plan.get("drive"):
        steps = G.STEPS["fast" if j.get("fast", True) else "normal"]
        expect((g[G.N_SCHED]["inputs"]["steps"], g[G.N_SAMPLER]["inputs"]["transition_step"]) == steps, f"{jid}: steps")
    if plan["task"] == "Ref2VA":
        n_img = sum(1 for k in ci if k.startswith("ref_images."))
        expect(n_img == len(plan["chosen"]["image"]), f"{jid}: image ref slots {n_img}")
        kind = lambda w: G.MENTION_KIND[w.lower() if w.isascii() else w]
        mentioned = {(kind(m.group(1)), int(m.group(2))) for m in G.MENTION_RE.finditer(j["prompt"])}
        for k, prefix in (("image", "ref_images."), ("video", "ref_videos."), ("audio", "ref_audios.")):
            want = sum(1 for kk, _ in mentioned if kk == k) + (1 if k == "image" and plan["extend"] else 0)
            got = sum(1 for key in ci if key.startswith(prefix))
            expect(got == want, f"{jid}: {k} slots {got} != {want} (@ mentions + extend frame)")
        expect("@" not in ci["prompt"], f"{jid}: @ mention left in the prompt")
    intensity = j.get("intensity", "off")
    if intensity == "off":
        expect(not any(n in g for n in G.LORAS_ACTION), f"{jid}: action LoRA left with intensity off")
        expect(g["331"]["inputs"]["model"] == ["404", 0], f"{jid}: attention backend not rewired to the turbo LoRA")
    else:
        for n in G.LORAS_ACTION:
            want = round(template[n]["inputs"]["strength_model"] * G.INTENSITY[intensity], 4)
            expect(g[n]["inputs"]["strength_model"] == want, f"{jid}: LoRA {n} strength")
    if plan["task"] in ("I2VA", "FL2VA", "L2VA"):
        expect(ci["prompt"].startswith(G.alignment_line(plan["task"], plan["seconds"])), f"{jid}: alignment line missing")
    return (f"{plan['task']:6} {plan['width']}x{plan['height']} {plan['seconds']:g}s steps={plan['steps']}/{plan['transition_step']} "
            f"refs={len(plan['chosen']['image'])}/{len(plan['chosen']['video'])}/{len(plan['chosen']['audio'])} "
            f"intensity={intensity} extend={plan['extend']} nodes={len(g)}"
            + (f" unused={plan['unused_inputs']}" if plan["unused_inputs"] else ""))


def run_batch(m, rng, label):
    template = G.load_template(G.VIDEO_TEMPLATE)
    print(f"== {label}")
    for j in m["jobs"]:
        if j["type"] == "tts":
            print(f"  {j['id']:10} tts (posted as h3_tts32_api.json, unchanged)")
            continue
        before = len(FAILS)
        try:
            plan, g, tail = build(j, rng)
            line = check_job(j, plan, g, tail, template)
        except Exception as e:
            FAILS.append(f"{j['id']}: {type(e).__name__}: {e}")
            line = "BUILD FAILED"
        print(f"  {'ok ' if len(FAILS) == before else 'BAD'} {j['id']:10} {line}")


def templates():
    print("== templates in workflows/ (posted as they are)")
    for name in (G.VIDEO_TEMPLATE, G.PICTURE_TEMPLATE):
        t = G.load_template(name)
        probs = G.check_graph(t)
        expect(not probs, f"{name}: {probs}")
        gone = [n for n in ("51", "252", "329", "364", "401") if n in t]
        expect(not gone, f"{name}: preview / sample nodes still there: {gone}")
        print(f"  {'ok ' if not probs and not gone else 'BAD'} {name}: {len(t)} nodes, {len({n['class_type'] for n in t.values()})} classes")


def broken(rng):
    """Deliberately broken graphs: each must be reported."""
    print("== broken graphs (each must be caught)")
    j = SAMPLE["jobs"][3]  # shot01: ref, intensity off
    plan, g, tail = build(j, rng)
    pj = SAMPLE["jobs"][1]
    pplan, pg, _ = build(pj, rng)
    cases = []
    b = copy.deepcopy(g); b["331"]["inputs"]["model"] = ["999", 0]
    cases.append(("dangling link 331.model -> 999", G.check_graph(b)))
    b = copy.deepcopy(g); b["364"] = {"class_type": "VHS_VideoCombine", "inputs": {"images": ["253", 0]}}
    cases.append(("VHS_VideoCombine left in", G.check_graph(b)))
    b = copy.deepcopy(g); b["329"] = {"class_type": "ModelPreviewOverrideKJ", "inputs": {"model": ["333", 0]}}
    b[G.N_SAMPLER]["inputs"]["model"] = ["329", 0]
    cases.append(("preview override in the model chain", G.check_graph(b)))
    raw = json.loads((Path(TOYBOX) / "WF.json").read_text(encoding="utf-8")) if TOYBOX else None
    if raw is not None:
        cases.append(("Toybox's raw WF.json posted as is", G.check_graph(raw)))
    b = G.video_graph(G.load_template(G.VIDEO_TEMPLATE), dict(plan, multiplier=1.0))  # built as if intensity were on
    cases.append(("intensity off but action LoRAs kept", G.check_video(b, plan)))
    b = copy.deepcopy(g); b["ri9"] = {"class_type": "LoadImage", "inputs": {"image": "x.png"}}
    del b[G.N_COND]["inputs"]["ref_images.ref_image_1"]
    cases.append(("reference LoadImage not wired into its slot", G.check_video(b, plan)))
    b = copy.deepcopy(g); b[G.N_SECONDS]["inputs"]["value"] = 5.0
    cases.append(("seconds not written", G.check_video(b, plan)))
    b = copy.deepcopy(g); b[G.N_SCHED]["inputs"]["steps"] = 12
    cases.append(("steps not written", G.check_video(b, plan)))
    b = copy.deepcopy(g)  # model_set official asked for, but the graph still has the default (community) models
    cases.append(("model_set official not written", G.check_video(b, dict(plan, model_set="official"))))
    # official first-frame clip built with the Ref2VA weights (the family has to follow the task)
    fplan, fg, _ = build(SAMPLE["jobs"][[x["id"] for x in SAMPLE["jobs"]].index("shot07")], rng)
    b = copy.deepcopy(fg); b[G.N_UNET]["inputs"]["unet_name"] = G.OFFICIAL["ref"]["unet"]
    cases.append(("official I2VA given the Ref2VA UNet", G.check_video(b, fplan)))
    b = copy.deepcopy(fg); b[G.N_TURBO]["inputs"]["lora_name"] = G.OFFICIAL["ref"]["turbo"]
    cases.append(("official I2VA given the Ref2VA turbo", G.check_video(b, fplan)))
    b = copy.deepcopy(pg); del b[G.N_IMG_ENC]["inputs"]["images.image_2"]
    cases.append(("picture edit reference not wired", G.check_picture(b, pplan)))
    for name, probs in cases:
        expect(probs, f"NOT CAUGHT: {name}")
        print(f"  {'caught' if probs else 'MISSED'}  {name}: {probs[0] if probs else '-'}")


def against_toybox(toybox, rng):
    """Run Toybox's build_generate() / ph3_graph() (taken out of server.py, nothing else of it is imported) on the
    same plans and compare. Only the save prefix and the UI-only _meta titles may differ."""
    import ast
    src = (Path(toybox) / "server.py").read_text(encoding="utf-8")
    ns = {"json": json, "random": random, "Path": Path, "TEMPLATE_PATH": Path(toybox) / "WF.json", "FPS": G.FPS,
          "N_COND": G.N_COND, "N_SAMPLER": G.N_SAMPLER, "N_SCHED": G.N_SCHED, "N_BRIDGE": G.N_BRIDGE, "N_VAE": G.N_VAE,
          "N_AVAE": G.N_AVAE, "N_CLIP": G.N_CLIP, "N_SECONDS": G.N_SECONDS, "N_DECODE": G.N_DECODE,
          "N_ADECODE": G.N_ADECODE, "PREVIEW_ONLY": ("401", "329", "364", "51", "252"), "N_IMG_ENC": G.N_IMG_ENC,
          "N_IMG_SAMPLER": "isamp", "N_IMG_SAVE": G.N_IMG_SAVE}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in ("bypass", "build_generate", "ph3_graph"):
            exec(compile(ast.Module([node], []), "server.py", "exec"), ns)
    pt = G.load_template(G.PICTURE_TEMPLATE)
    cfg = {"steps": {k: list(v) for k, v in G.STEPS.items()},
           "loras": {"drama": list(G.LORAS_DRAMA), "action": list(G.LORAS_ACTION)},
           "intensity_multipliers": G.INTENSITY, "extend": G.EXTEND,
           "image": {"ph3": {"unet": pt["iu"]["inputs"]["unet_name"], "lora": pt["ilo"]["inputs"]["lora_name"],
                             "clip": pt["il"]["inputs"]["clip_name"], "vae": pt["iv"]["inputs"]["vae_name"],
                             "sigmas": pt["im"]["inputs"]["sigmas"]}}}

    def norm(g):
        g = copy.deepcopy(g)
        for n in g.values():
            n.pop("_meta", None)
            if n["class_type"] in ("SaveVideo", "SaveImage"):
                n["inputs"]["filename_prefix"] = "-"
        return g

    print(f"== same graphs as H3 Toybox ({toybox})")
    for j in SAMPLE["jobs"]:
        plan, ours, tail = build(j, rng)
        if j["type"] == "image":
            toy = ns["ph3_graph"](cfg, plan["final_prompt"], plan["width"], plan["height"],
                                  [dict(r, comfy_name=r["name"]) for r in plan["refs"]], plan["seed"])
        else:
            chosen = {k: [dict(i, comfy_name=i["name"]) for i in v] for k, v in plan["chosen"].items()}
            req = {"seconds": j.get("seconds", 10), "fast": j.get("fast", True), "seed": plan["seed"],
                   "intensity": j.get("intensity", "off")}
            toy = ns["build_generate"](cfg, plan["final_prompt"], chosen, plan["task"], req,
                                       (plan["width"], plan["height"]), {"tail_name": tail} if tail else None)[0]
        a, b = norm(ours), norm(toy)
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        expect(not diff, f"{j['id']}: differs from Toybox at nodes {diff}")
        print(f"  {'identical' if not diff else 'DIFFERENT ' + str(diff)}  {j['id']}")


def audio_lock(rng):
    """drive_audio: T8 lock_source conditioning, <Audio 1> = the recording, seconds from the recording."""
    song = dict(fake("inputs/song.wav"), seconds=9.42)
    t = G.load_template(G.VIDEO_TEMPLATE)
    j = {"id": "sing", "type": "video", "mode": "ref", "prompt": "@圖片1 sings along with <Audio 1>.", "resolution": "480P",
         "aspect": "9:16"}
    plan = G.plan_video(j, inputs=[fake("inputs/lin.png")], drive=song, rng=rng)
    g = G.video_graph(t, plan)
    ci = g[G.N_COND]["inputs"]
    expect(g[G.N_COND]["class_type"] == "MiniMaxH3AudioConditioningT8" and ci["audio_mode"] == "lock_source",
           "audio lock: conditioning is T8 lock_source")
    expect(plan["seconds"] == 9.42 and g[G.N_SECONDS]["inputs"]["value"] == 9.42, "audio lock: seconds taken from the recording")
    expect("<Audio 1>" in ci["prompt"] and "<Picture 1>" in ci["prompt"] and "@" not in ci["prompt"],
           "audio lock: <Audio 1> kept for the recording, picture mention resolved")
    expect("ref_images.ref_image_0" in ci and not any(k.startswith("ref_audios.") for k in ci), "audio lock: picture ref kept, no audio ref slot")
    expect(ci["length"] == t[G.N_COND]["inputs"]["length"], "audio lock: frame count still from the seconds node")
    expect(not G.check_video(g, plan), f"audio lock: {G.check_video(g, plan)}")
    jt = {"id": "sing3", "type": "video", "mode": "text", "prompt": "a singer on stage, <Audio 1>", "seconds": 6}
    pt = G.plan_video(jt, drive=song, rng=rng); gt = G.video_graph(t, pt)
    expect(gt[G.N_COND]["inputs"].get("task_type") == "Ref2VA" and not G.check_video(gt, pt),
           "audio lock: text-only clip runs as Ref2VA (the recording is its reference)")
    bad = copy.deepcopy(gt); bad[G.N_COND]["inputs"]["task_type"] = "T2VA"
    expect(bool(G.check_video(bad, pt)), "audio lock NOT CAUGHT: text-only clip sent to T8 as T2VA")
    # NEG: these must be refused or caught
    jf = {"id": "sing2", "type": "video", "mode": "frames", "prompt": "she sings <Audio 1>", "seconds": 6}
    for name, call in (("audio in inputs", lambda: G.plan_video(j, inputs=[fake("inputs/x.wav")], drive=song)),
                       ("frames mode", lambda: G.plan_video(jf, first=fake("inputs/end.png"), drive=song)),
                       ("over 15 s", lambda: G.plan_video(dict(j, seconds=20), inputs=[fake("inputs/lin.png")], drive=song)),
                       ("not audio", lambda: G.plan_video(j, inputs=[], drive=fake("inputs/lin.png"))),
                       ("with extend", lambda: G.plan_video(j, inputs=[], drive=song, extend={"width": 1, "height": 1,
                                                                                            "last_frame": fake("l.jpg")}))):
        try:
            call()
            expect(False, f"audio lock NOT REFUSED: {name}")
        except ValueError:
            pass
    bad = copy.deepcopy(g)
    bad["da"]["inputs"]["audio"] = "other.wav"
    expect(G.check_video(bad, plan), "audio lock NOT CAUGHT: drive_audio loads another file")
    lw, lh = G.drive_low_size(plan["width"], plan["height"])
    expect(ci["width"] == lw and g["dup"]["inputs"]["mode.width"] == plan["width"] and g["dlock"]["inputs"]["source_audio"] == ["da", 0]
           and g[G.N_SAMPLER]["inputs"]["latent_image"] == ["dlock", 0] and g["dsp"]["inputs"]["step"] == plan["transition_step"],
           "audio lock: low 2/3 pass -> x1.5 lift -> recording locked back -> high pass")
    expect(all(v != [G.N_SAMPLER, 0] for n in g.values() for v in n["inputs"].values()), "audio lock: decoders read output 1")
    bad = copy.deepcopy(g)
    bad[G.N_SAMPLER]["inputs"]["latent_image"] = ["dcat", 0]
    expect(G.check_video(bad, plan), "audio lock NOT CAUGHT: recording not re-locked after the upscale")
    bad = copy.deepcopy(g)
    bad["dup"]["inputs"]["mode.width"] = lw
    expect(G.check_video(bad, plan), "audio lock NOT CAUGHT: upscale does not reach the planned size")
    bad = copy.deepcopy(g)
    bad["dn"]["inputs"]["noise_seed"] = 1
    expect(G.check_video(bad, plan), "audio lock NOT CAUGHT: seed differs")
    bad = copy.deepcopy(g)
    bad[G.N_CREATE]["inputs"]["audio"] = [G.N_ADECODE, 0]
    expect(G.check_video(bad, plan), "audio lock NOT CAUGHT: video muxed with generated audio")
    plain = G.video_graph(t, G.plan_video(dict(j, prompt="@圖片1 walks"), inputs=[fake("inputs/lin.png")], rng=rng))
    expect(plain[G.N_COND]["class_type"] == "MiniMaxH3ReferenceToVideo", "no drive_audio: graph unchanged")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--toybox", help="folder with Toybox's server.py and WF.json, to compare graphs")
    a = ap.parse_args()
    rng = random.Random(0)
    global TOYBOX
    TOYBOX = a.toybox
    templates()
    run_batch(SAMPLE, rng, "sample batch (every job type)")
    if a.manifest:
        run_batch(json.loads(Path(a.manifest).read_text(encoding="utf-8")), rng, a.manifest)
    broken(rng)
    if a.toybox:
        against_toybox(a.toybox, rng)
    # a few exact values, so a silent change in the port shows up
    expect(G.dims("9:16", "480P") == (480, 864) and G.dims("16:9", "720P") == (1280, 704), "dims()")  # Toybox round32: 720/32=22.5 rounds to 22
    expect(G.image_dims("9:16", "2K", []) == (1152, 2048), "image_dims()")
    t, ch = G.resolve_mentions("@圖片2 then @Image1 and @圖片2", [fake("a.png"), fake("b.png")])
    expect(t == "<Picture 1> then <Picture 2> and <Picture 1>" and [c["name"] for c in ch["image"]] ==
           ["h3skill_b_png.png", "h3skill_a_png.png"], "resolve_mentions() order")
    t, ch = G.resolve_mentions("<Subject 1> is in <Picture 1>; room <Picture 2>", [fake("a.png"), fake("b.png")])
    expect(len(ch["image"]) == 2, "literal <Picture N> (official guide) keeps its references")
    expect(G.raw_prompt(G.alignment_line("FL2VA", 4.0) + "\n\nbody", "FL2VA", 4.0, None, False)
           .count("How the reference pictures align") == 1, "hand-written alignment line is not duplicated")
    # TTS: lead-in as its own line, Simplified dialogue, seconds from the text, cut after the lead's silence
    p, n = G.tts_prompt("x <d>[Mandarin] 電鑽，跟阿姨借的。</d> y", True, lambda t: t.replace("電鑽", "电钻"))
    expect(p.count("<d>") == 2 and "<d>[Mandarin] 嗯。</d>" in p and "电钻" in p and n == 7, "tts_prompt(): lead line + conversion")
    expect(G.tts_prompt("<d>[English] Hi.</d>", False, None)[0] == "<d>[English] Hi.</d>", "tts_prompt(): lead off, English untouched")
    expect(G.tts_seconds(7) == round(7 / G.TTS_DENSITY + 0.4 + G.TTS_LEAD_SECONDS, 2) and G.tts_seconds(7, 5) == 5.0, "tts_seconds()")
    q, l = -60.0, -10.0   # quiet / loud, 10 ms hops
    head = [l] * 30 + [q] * 25 + [l] * 30 + [q] * 40 + [l] * 150   # 0.3 s sound at 0.00 (as clip n09 had), silence, lead 嗯, silence, the line
    expect(abs(G.lead_cut(head) - 1.21) < 1e-6, "lead_cut(): onset click skipped, cut after the lead's silence")
    expect(G.lead_cut([q] * 40 + [l] * 200) is None, "lead_cut(): no lead -> None (NEG)")
    # model names: fitted to this ComfyUI's own spelling; a file it does not have is reported
    info = {"Bridge": {"input": {"required": {"model_name": ["COMBO", {"options": ["t8_compat/b.safetensors"]}]}}},
            "Lora": {"input": {"required": {"lora_name": [["H3\\l.safetensors", "x.safetensors"], {}]}}}}
    g = {"1": {"class_type": "Bridge", "inputs": {"model_name": "t8_compat\\b.safetensors"}},
         "2": {"class_type": "Lora", "inputs": {"lora_name": "H3/l.safetensors"}},
         "3": {"class_type": "Lora", "inputs": {"lora_name": "l.safetensors"}},
         "4": {"class_type": "Lora", "inputs": {"lora_name": "gone.safetensors"}}}
    changes, missing = G.fit_combo_names(g, info)
    expect(g["1"]["inputs"]["model_name"] == "t8_compat/b.safetensors", "backslash name fitted to t8_compat/")
    expect(g["2"]["inputs"]["lora_name"] == g["3"]["inputs"]["lora_name"] == "H3\\l.safetensors", "name fitted to H3\\")
    expect(missing == ["4.lora_name: gone.safetensors"], "missing model file reported")
    audio_lock(rng)
    print()
    if FAILS:
        print("FAILED:\n- " + "\n- ".join(FAILS))
        sys.exit(1)
    print("ALL OK")


if __name__ == "__main__":
    main()
