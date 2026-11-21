# -*- coding: utf-8 -*-
"""Local side of the h3-multimodal-colab skill: sends one batch (images / videos / extensions / TTS) to a Colab GPU,
watches it, brings the files back and releases the GPU.

  python h3run.py check                          tools, login, compute units
  python h3run.py run job/manifest.json [--gpu L4] [--keep] [--session NAME] [--max-minutes 120]
  python h3run.py sessions                       runtimes still alive on your account (they cost units)
  python h3run.py stop --session NAME            release one runtime
  python h3run.py stop --all                     release every runtime this skill started

The batch is run in the background on the VM (start.py) and polled with short `colab exec` calls (status.py):
a long-lived exec can drop its websocket; short ones do not. Results land in <manifest folder>/out/.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILL = HERE.parent
REMOTE = SKILL / "remote"
GPUS = ("L4", "G4", "A100", "H100")   # colab-cli turns an unknown name into A100 without a word: only these pass
TYPES = ("image", "video", "tts")
PREFIX = "h3skill-"
BATCH_ROOT = "/content/h3_batches"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")


def colab_exe():
    for c in (shutil.which("colab"), Path.home() / ".local" / "bin" / "colab.exe", Path.home() / ".local" / "bin" / "colab"):
        if c and Path(c).exists():
            return str(c)
    sys.exit("找不到 colab 指令：請先執行安裝（install/install.ps1 或 install/install.sh）")


def colab(*args, timeout=600, check=True):
    """check=False: a call that hangs (colab-cli's websocket sometimes stalls) returns "" instead of raising."""
    try:
        # stdin closed: a backgrounded `colab exec` that inherits an open stdin can sit waiting on it
        p = subprocess.run([colab_exe(), *map(str, args)], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=ENV, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        if check:
            raise RuntimeError(f"colab {' '.join(map(str, args))} 超過 {timeout} 秒沒有回應")
        return ""
    out = (p.stdout or "") + (p.stderr or "")
    if check and p.returncode != 0:
        raise RuntimeError(f"colab {' '.join(map(str, args))} 失敗（{p.returncode}）：\n{out[-2000:]}")
    return out


def token_ok():
    return (Path.home() / ".config" / "colab-cli" / "token.json").is_file()


def usage():
    out = colab("usage", timeout=120, check=False)
    m = re.search(r"([\d.]+)\s*(?:compute units|運算單元|units)", out, re.I)
    return (float(m.group(1)) if m else None), out.strip()


def runtime_rows():
    out = colab("sessions", timeout=120).strip()
    rows = re.findall(r"^\[([^\]]+)\]\s+(\S+)\s+\| Hardware: (\S+) \|.*$", out, re.M)
    if not rows and "[colab] No active sessions found on server." not in out.splitlines():
        raise RuntimeError("無法確認 Colab runtime 清單")
    if any("| Hardware:" in line and not re.fullmatch(
            r"\[[^\]]+\]\s+\S+\s+\| Hardware: \S+ \|.*", line) for line in out.splitlines()):
        raise RuntimeError("Colab runtime 清單格式無法辨識")
    return rows, out


def sessions():
    rows, out = runtime_rows()
    return [name for name, _, _ in rows if name.startswith(PREFIX)], out


# ---------------------------------------------------------------- keep-alive (the GPU waits for the next batch)
KEEP = Path.home() / ".config" / "h3skill" / "keep"


def keep_file(session):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session):
        raise RuntimeError(f"不支援的 session 名稱：{session!r}")
    return KEEP / f"{session}.json"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="w", encoding="utf-8") as f:
        tmp = Path(f.name)
        try:
            json.dump(value, f, ensure_ascii=False, indent=1)
            f.close()
            tmp.replace(path)
        finally:
            f.close()
            tmp.unlink(missing_ok=True)


def set_keep(session, expires, pid=None, endpoint=None):
    """expires = epoch seconds, or None while a batch is running (the watchdog never stops a busy GPU)."""
    KEEP.mkdir(parents=True, exist_ok=True)
    old = json.loads(keep_file(session).read_text(encoding="utf-8")) if keep_file(session).exists() else {}
    save_json(keep_file(session), {"session": session, "expires": expires,
                                  "endpoint": endpoint or old.get("endpoint"),
                                  "watchdog_pid": pid or old.get("watchdog_pid")})


def drop_keep(session):
    keep_file(session).unlink(missing_ok=True)


@contextmanager
def local_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"另一個本機程序正在操作：{path.stem}") from exc
        yield


def release(session, endpoint=None):
    path = keep_file(session)
    if not endpoint and path.exists():
        endpoint = json.loads(path.read_text(encoding="utf-8")).get("endpoint")
    before, _ = runtime_rows()
    current = next((ep for n, ep, _ in before if n == session), None)
    if endpoint and current and endpoint != current:
        raise RuntimeError(f"{session} 已指向另一台 runtime，未執行關閉")
    endpoint = endpoint or current
    if not endpoint and any(n == "?" for n, _, _ in before):
        raise RuntimeError(f"無法確認 {session} 是否為未具名 runtime，未清除紀錄")
    stop_error = None
    if current:
        try:
            colab("stop", "-s", session, timeout=300)
        except RuntimeError as exc:
            stop_error = exc
    after, _ = runtime_rows()
    if any(n == session or (endpoint and ep == endpoint) for n, ep, _ in after):
        raise RuntimeError(f"未確認 {session} 已釋放；請重試 stop --session {session}") from stop_error
    drop_keep(session)
    print(f"🛑 已確認釋放 GPU：{session}", flush=True)


def pid_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_watchdog(session):
    """A detached local process that releases the GPU when the keep time runs out, even after the agent's turn
    (or the whole chat) has ended. One per session."""
    info = json.loads(keep_file(session).read_text(encoding="utf-8"))
    if pid_alive(info.get("watchdog_pid")):
        return
    kw = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    if os.name == "nt":
        kw["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_watchdog", "--session", session], env=ENV, **kw)
    set_keep(session, info.get("expires"), p.pid)


def watchdog_once(session):
    """One expiry check. Unknown busy state must not release another host's GPU."""
    path = keep_file(session)
    if not path.exists():
        return "gone"
    with local_lock(path.with_suffix(".lock")):
        if not path.exists():
            return "gone"
        info = json.loads(path.read_text(encoding="utf-8"))
        if info.get("expires") is None or time.time() < info["expires"]:
            return "waiting"
        rows, _ = runtime_rows()
        endpoint = info.get("endpoint")
        try:
            busy = runtime_busy(session) if any(n == session and ep == endpoint for n, ep, _ in rows) else False
        except RuntimeError:
            return "unknown"
        if busy:
            return "busy"
        release(session, endpoint)
        return "released"


def watchdog(session):
    log = KEEP / f"{session}.log"

    def say(msg):
        with log.open("a", encoding="utf-8") as f:
            f.write(time.strftime("%m-%d %H:%M:%S ") + msg + "\n")

    say("watchdog started")
    while True:
        time.sleep(30)
        try:
            state = watchdog_once(session)
        except RuntimeError as exc:
            say(str(exc))
            continue
        if state == "gone":
            say("keep file removed (stopped by hand): exit")
            return
        if state == "released":
            say("released")
            return


# ---------------------------------------------------------------- manifest
def load_manifest(path):
    path = Path(path).resolve()
    m = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    ms = m.get("model_set")  # top-level default for the video jobs (references/official.md)
    for j in m.get("jobs") or []:
        if j.get("type") == "video" and ms and "model_set" not in j:
            j["model_set"] = ms
    jobs, seen, problems = m.get("jobs") or [], set(), []
    if not jobs:
        problems.append("jobs 是空的")
    for j in jobs:
        jid = str(j.get("id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", jid):
            problems.append(f"id 只能用英數、- 和 _：{jid!r}")
        if jid in seen:
            problems.append(f"id 重複：{jid}")
        if j.get("model_set", "default") not in ("default", "official"):
            problems.append(f"{jid}：model_set 只能是 default 或 official")
        if j.get("type") not in TYPES:
            problems.append(f"{jid}：type 要是 {TYPES}")
        if not str(j.get("prompt", "")).strip():
            problems.append(f"{jid}：沒有 prompt")
        files = list(j.get("inputs", [])) + list(j.get("voices", [])) + [j[k] for k in ("first", "last", "drive_audio") if j.get(k)]
        if j.get("extend") and j["extend"] not in seen and not str(j["extend"]).startswith("@"):
            files.append(j["extend"])  # not an earlier job in this batch: a video file the user gave
        for f in files:
            if str(f).startswith("@"):  # an earlier job's output in this batch
                if f[1:] not in seen:
                    problems.append(f"{jid}：{f} 要指向排在它前面的 job")
                continue
            if not (base / f).is_file():
                problems.append(f"{jid}：找不到輸入檔 {f}（路徑相對於 manifest 所在資料夾）")
        if j.get("type") == "video" and j.get("extend") and j.get("mode", "ref") != "ref":
            problems.append(f"{jid}：延伸只能用 ref 模式")
        if j.get("type") == "image" and len(j.get("inputs", [])) > 16:
            problems.append(f"{jid}：圖片編輯與參考最多 16 張")
        if j.get("type") == "tts" and len(j.get("voices", [])) > 3:
            problems.append(f"{jid}：TTS 最多 3 條參考聲音")
        if j.get("type") in ("video", "tts") and not 2 <= float(j.get("seconds", 10)) <= 15:
            problems.append(f"{jid}：seconds 要在 2–15 之間")
        seen.add(jid)
    if problems:
        sys.exit("manifest 有問題，沒有開機：\n- " + "\n- ".join(problems))
    return m, base


# ---------------------------------------------------------------- run
def remote_exec(session, code, timeout=120, tries=3):
    """A short `colab exec`; a stalled one is retried (the code is idempotent)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code.rstrip("\n") + "\nprint('H3EXEC_OK')\n")  # proof it ran: some code prints nothing
    try:
        for k in range(tries):
            out = colab("exec", "-s", session, "-f", f.name, "--timeout", timeout, timeout=timeout + 30, check=False)
            if "H3EXEC_OK" in out:
                return out.replace("H3EXEC_OK", "").strip()
            print(f"…exec 沒有回應，重試（{k + 1}/{tries}）", flush=True)
        raise RuntimeError("colab exec 連續沒有回應")
    finally:
        os.unlink(f.name)


def skip_finished(m, base, force):
    """Resume: a job whose file is already in out/ is not made again; later jobs that point at it with @id
    get that local file instead (it is uploaded like any other input)."""
    if force:
        return m
    done = {}
    for j in m["jobs"]:
        hit = sorted((base / "out").glob(f"{j['id']}.*"))
        hit = [h for h in hit if h.suffix in (".png", ".jpg", ".webp", ".mp4", ".wav", ".flac")]
        if hit:
            done[j["id"]] = "out/" + hit[0].name
    if not done:
        return m
    print(f"⏭️ 已經做好、這次跳過：{', '.join(done)}（要重做加 --force）", flush=True)
    swap = lambda v: done.get(v[1:], v) if isinstance(v, str) and v.startswith("@") else v
    jobs = []
    for j in m["jobs"]:
        if j["id"] in done:
            continue
        j = dict(j)
        for k in ("inputs", "voices"):
            if k in j:
                j[k] = [swap(v) for v in j[k]]
        for k in ("first", "last", "extend", "drive_audio"):
            if k in j:
                j[k] = swap(j[k])
        jobs.append(j)
    if not jobs:
        sys.exit("這一批全部都做好了（out/ 裡都有檔案）。要重做加 --force")
    return dict(m, jobs=jobs)


DEFAULT_RES = {"L4": "480P", "G4": "720P", "A100": "720P", "H100": "720P"}  # the creator: G4 means 720P by default


def choose_gpu_auto(m):
    """Boot-time auto-tiering: L4 for light/interactive runs; A100 (40GB) for heavy batches."""
    imgs = [j for j in m["jobs"] if j.get("type") == "image"]
    vids = [j for j in m["jobs"] if j.get("type") == "video"]
    if any(str(j.get("size", "2K")).upper() in ("2.7K", "3K", "4K") for j in imgs):
        return "A100", "包含 2.7K/3K/4K 高畫質大圖"
    if len(imgs) >= 5:
        return "A100", f"單批包含 {len(imgs)} 張圖片（≥5 張大批次）"
    if any(str(j.get("resolution", "")).upper() == "720P" for j in vids):
        return "A100", "包含 720P 影片"
    if len(vids) >= 2 or sum(float(j.get("seconds", 10)) for j in vids) >= 10:
        return "A100", "包含多支或 ≥10 秒影片任務"
    return "L4", "一般 2K 生圖／短片試拍（省點模式）"


def spec_summary(m, gpu):
    vids = [j for j in m["jobs"] if j["type"] == "video"]
    secs = sum(float(j.get("seconds", 10)) for j in vids)
    shapes = sorted({j.get("aspect", "auto") for j in vids})
    res = sorted({j.get("resolution") for j in vids})
    kinds = {t: sum(1 for j in m["jobs"] if j["type"] == t) for t in TYPES}
    est = ""
    if gpu == "L4":  # measured 2026-11-04: a 5 s 480P Ref2VA clip took 160 s on L4 (720P has 2.25x the pixels)
        per = {"480P": 32, "720P": 72}
        est = f"；L4 估計影片約 {sum(per.get(j.get('resolution'), 72) * float(j.get('seconds', 10)) for j in vids) / 60:.0f} 分鐘（不含開機與下載模型）"
    return (f"圖片 {kinds['image']}、影片 {kinds['video']}（共 {secs:g} 秒，解析度 {'/'.join(res) or '-'}，"
            f"比例 {'/'.join(shapes) or '-'}）、配音 {kinds['tts']}，GPU {gpu}{est}")


def apply_defaults(m, args, gpu):
    if args.idle_minutes is None:
        args.idle_minutes = 15.0 if gpu == "L4" else 10.0
    for job in m["jobs"]:
        if job["type"] == "video" and not job.get("resolution"):
            job["resolution"] = DEFAULT_RES[gpu]


def create_runtime(gpu, auto):
    rows, _ = runtime_rows()
    if any(name == "?" for name, _, _ in rows):
        raise RuntimeError("有未具名 runtime，無法確認它不是舊批；先用 colab sessions 辨識或 colab stop 釋放，再開新批")
    for target in (gpu, "L4") if gpu != "L4" and auto else (gpu,):
        session = f"{PREFIX}{time.strftime('%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        print(f"🚀 開一台 {target}：{session}", flush=True)
        try:
            colab("new", "-s", session, "--gpu", target, timeout=900)
        except RuntimeError as exc:
            if target != "L4" and auto and "Allocation refused (precondition failed)" in str(exc):
                print(f"⚠️ Colab 拒絕配置 {target}，改試 L4；不改已指定的規格", flush=True)
                continue
            raise RuntimeError(f"未確認 {session} 已開機：{exc}。請先查 sessions，不要直接重開。") from exc
        return session, target
    raise RuntimeError("無法開機")


def start_batch(record):
    """Start one batch. A lost reply is reconciled by batch ID; only a confirmed miss is cleaned up."""
    session, batch_id = record["session"], record["batch_id"]
    try:
        out = remote_exec(session, batch_script(batch_id, "start.py"), timeout=300, tries=1)
        return batch_reply(out, "H3START", batch_id)
    except RuntimeError as start_error:
        try:
            st = poll(session, batch_id)
        except RuntimeError:
            st = None
        if st and st.get("batch_id") == batch_id and st.get("stage") != "not accepted":
            return dict(started=True, batch_id=batch_id, reconciled=True)
        cleanup_batch(session, batch_id)
        raise RuntimeError(f"未確認本批已啟動，已清理未接受的上傳：{start_error}") from start_error


def cleanup_batch(session, batch_id):
    if not re.fullmatch(r"[0-9a-f]{32}", batch_id):
        raise RuntimeError("批次 ID 格式錯誤")
    remote_exec(session, f"""
from pathlib import Path
import shutil
root = Path({BATCH_ROOT!r})
job = root / {batch_id!r}
if (job / "accepted.json").is_file():
    raise SystemExit("accepted")
shutil.rmtree(job, ignore_errors=True)
(root / f"{batch_id}.tar.gz").unlink(missing_ok=True)
""", timeout=60, tries=1)


def batch_script(batch_id, script):
    if not re.fullmatch(r"[0-9a-f]{32}", batch_id):
        raise RuntimeError("批次 ID 格式錯誤")
    return f"ROOT = {BATCH_ROOT!r}\nBATCH_ID = {batch_id!r}\n" + (REMOTE / script).read_text(encoding="utf-8")


def batch_reply(out, marker, batch_id):
    matches = re.findall(rf"^{marker} (\{{.*\}})\s*$", out, re.M)
    if len(matches) != 1:
        raise RuntimeError(f"未收到 {marker} 批次回應")
    reply = json.loads(matches[0])
    if reply.get("batch_id") != batch_id:
        raise RuntimeError("遠端回應的批次 ID 不符，拒絕下載或交付")
    return reply


def poll(session, batch_id):
    try:
        out = remote_exec(session, batch_script(batch_id, "status.py"), timeout=60, tries=1)
    except RuntimeError:
        return None
    return batch_reply(out, "H3STATUS", batch_id)


def runtime_busy(session):
    out = remote_exec(session, f"""
import fcntl, subprocess
from pathlib import Path
root = Path({BATCH_ROOT!r})
root.mkdir(exist_ok=True)
with (root / 'worker.lock').open('a') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        busy = subprocess.run(['pgrep', '-f', r'/content/h3job/remote_job[.]py'], stdout=subprocess.DEVNULL).returncode == 0
    except BlockingIOError:
        busy = True
print('H3BUSY ' + str(int(busy)))
""", timeout=60)
    match = re.search(r"^H3BUSY ([01])\s*$", out, re.M)
    if not match:
        raise RuntimeError("未確認 runtime 是否正在執行其他批次")
    return match[1] == "1"


def download_result(session, batch_id, jid, row, out_dir):
    name = row.get("file", "")
    size = row.get("bytes")
    digest = row.get("sha256", "")
    if (not isinstance(name, str) or not re.fullmatch(re.escape(jid) + r"\.[A-Za-z0-9]+", name)
            or type(size) is not int or size <= 0
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise RuntimeError(f"{jid}：遠端產物的檔名、大小或 SHA-256 不合法")
    dst = out_dir / name
    # Each attempt uses a fresh temporary location, never a pre-existing output.
    for _ in range(3):
        with tempfile.TemporaryDirectory(prefix=".h3-download-", dir=out_dir) as scratch:
            tmp = Path(scratch) / name
            try:
                colab("download", "-s", session, f"{BATCH_ROOT}/{batch_id}/out/{name}", tmp, timeout=1200)
                if tmp.stat().st_size != size:
                    raise RuntimeError("大小不符")
                with tmp.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                        raise RuntimeError("SHA-256 不符")
                tmp.replace(dst)
            except (RuntimeError, OSError) as exc:
                last = exc
            else:
                print(f"⬇️ {dst}", flush=True)
                return dst
    raise RuntimeError(f"{jid}：下載未通過完整性驗證：{last}")


def upload_batch(session, batch_id, m, base):
    files = {f for j in m["jobs"] for f in list(j.get("inputs", [])) + list(j.get("voices", []))
             + [j[k] for k in ("first", "last", "extend", "drive_audio") if j.get(k) and (base / j[k]).is_file()]
             if not str(f).startswith("@")}
    with tempfile.TemporaryDirectory(prefix="h3-upload-") as scratch:
        bundle = Path(scratch) / f"{batch_id}.tar.gz"
        manifest = Path(scratch) / "manifest.json"
        manifest.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
        with tarfile.open(bundle, "w:gz", dereference=True) as tf:
            for name in ("remote_job.py", "h3graph.py"):
                tf.add(REMOTE / name, arcname=name)
            for name in ("h3_tts32_api.json", "h3_video_api.json", "h3_image_api.json"):
                tf.add(SKILL / "workflows" / name, arcname=name)
            tf.add(manifest, arcname=manifest.name)
            for f in sorted(files):
                print(f"📦 {f}", flush=True)
                tf.add(base / f, arcname="inputs/" + f.replace("\\", "/"))
        remote_exec(session, f"from pathlib import Path\nPath({BATCH_ROOT!r}).mkdir(exist_ok=True)")
        colab("upload", "-s", session, bundle, f"{BATCH_ROOT}/{bundle.name}", timeout=900)


def monitor(record, base, max_minutes):
    session, batch_id = record["session"], record["batch_id"]
    expected = set(record["job_ids"])
    out_dir = base / "out"
    out_dir.mkdir(exist_ok=True)
    t0, misses, last_line = time.monotonic(), 0, None
    while True:
        if time.monotonic() - t0 > max_minutes * 60:
            raise RuntimeError(f"已達本機監控上限（{max_minutes:g} 分鐘），遠端未停止")
        st = poll(session, batch_id)
        if st is None:
            misses += 1
            if misses >= 10:
                raise RuntimeError("連續 10 次讀不到遠端進度")
        else:
            misses = 0
            if st.get("stage") == "not accepted":
                raise RuntimeError("遠端沒有接受這一批，未下載任何產物")
            jobs = st.get("jobs", {})
            if set(jobs) - expected:
                raise RuntimeError("遠端工作清單有不屬於本批的 ID")
            line = f"{st.get('stage')} · " + " ".join(f"{k}:{v['status']}" for k, v in jobs.items())
            if line != last_line:
                print(line, flush=True)
                for msg in st.get("log", [])[-2:]:
                    print("    " + msg, flush=True)
                last_line = line
            if st.get("done"):
                break
        time.sleep(record["poll_interval"])

    results = {"session": session, "batch_id": batch_id, "gpu": record["gpu"], "jobs": {},
               "stage": st.get("stage"), "error": st.get("error"), "runtime_released": None}
    errors = {}
    for jid in record["job_ids"]:
        row = jobs.get(jid, {})
        result = results["jobs"][jid] = dict(row, local=None)
        if row.get("status") != "done":
            errors[jid] = row.get("error") or "遠端未回報完成"
            continue
        try:
            result["local"] = str(download_result(session, batch_id, jid, row, out_dir))
        except RuntimeError as exc:
            errors[jid] = str(exc)
            result.update(status="download_failed", error=str(exc))
    results["delivery_errors"] = errors
    results_path = out_dir / "results.json"
    save_json(results_path, results)
    if any(j.get("status") == "download_failed" for j in results["jobs"].values()):
        raise RuntimeError("有產物未完整下載，保留 runtime 供續收；詳見 out/results.json")

    # No GPU release is claimed until the server confirms absence.
    if record["stop_after"] or record["idle_minutes"] <= 0:
        for _ in range(5):  # final status may be committed just before the worker exits
            if not runtime_busy(session):
                break
            time.sleep(2)
        else:
            raise RuntimeError("runtime 上仍有工作，未執行關閉")
        release(session, record["endpoint"])
        results["runtime_released"] = True
    else:
        set_keep(session, time.time() + record["idle_minutes"] * 60, endpoint=record["endpoint"])
        start_watchdog(session)
        results["runtime_released"] = False
        print(f"⏸️ GPU 保留 {record['idle_minutes']:g} 分鐘；下一批可加 --session {session}", flush=True)
    save_json(results_path, results)
    count = sum(bool(j["local"]) for j in results["jobs"].values())
    if errors or results["error"]:
        raise RuntimeError(f"僅驗證 {count}/{len(expected)} 件成品；詳見 {results_path}")
    print(f"✅ 驗證並下載 {count}/{len(expected)} 件 → {out_dir}", flush=True)


def run(args):
    m, base = load_manifest(args.manifest)
    if args.skip_existing:
        m = skip_finished(m, base, args.force)
    explicit_gpu = args.gpu or m.get("gpu")
    auto = not explicit_gpu or str(explicit_gpu).upper() == "AUTO"
    gpu, reason = choose_gpu_auto(m) if auto else (str(explicit_gpu).upper(), "手動指定")
    if gpu not in GPUS:
        raise RuntimeError(f"GPU 只接受 {GPUS} 或 AUTO")
    if args.with_pictures:
        m["install_pictures"] = True
    if sum(j["type"] == "video" for j in m["jobs"]) >= 2 and not args.confirmed:
        if args.session and auto:
            _, gpu = identify_runtime(runtime_rows()[0], args.session)
        apply_defaults(m, args, gpu)
        print(f"📋 這批：{spec_summary(m, gpu)}", flush=True)
        raise RuntimeError("兩支以上影片須確認解析度、比例和秒數後，再加 --confirmed 送出")
    if not token_ok():
        raise RuntimeError("還沒登入 Colab：先跑 python colab_login.py url")
    batch_id = uuid.uuid4().hex
    batch_dir = base / ".h3-batches"
    with local_lock(batch_dir / "output.lock"):
        rows, _ = runtime_rows()
        if args.session:
            session = args.session
            endpoint, actual_gpu = identify_runtime(rows, session)
            if not auto and gpu != actual_gpu:
                raise RuntimeError(f"{session} 是 {actual_gpu}，與指定的 {gpu} 不符")
            gpu, reason = actual_gpu, "沿用指定 runtime"
        else:
            units, _ = usage()
            print(f"💳 剩餘運算單元：{units if units is not None else '讀不到'}", flush=True)
            if units is not None and units < 5:
                raise RuntimeError("運算單元不到 5，先不開機")
            session, gpu = create_runtime(gpu, auto)
            endpoint, gpu = identify_runtime(runtime_rows()[0], session)
        apply_defaults(m, args, gpu)
        print(f"📋 這批：{spec_summary(m, gpu)}（{reason}）", flush=True)
        with local_lock(keep_file(session).with_suffix(".lock")):
            if args.session and runtime_busy(session):
                raise RuntimeError(f"{session} 正在跑其他批次，未啟動新批")
            record = dict(session=session, endpoint=endpoint, batch_id=batch_id, gpu=gpu,
                          job_ids=[j["id"] for j in m["jobs"]], idle_minutes=args.idle_minutes,
                          stop_after=args.stop_after, poll_interval=12 if all(j["type"] == "image" for j in m["jobs"]) else 30)
            record_path = batch_dir / f"{batch_id}.json"
            save_json(record_path, record)
            # If the client disappears, leave recovery time but do not keep an idle VM forever.
            expires = time.time() + (args.max_minutes + max(args.idle_minutes, 15)) * 60
            set_keep(session, expires, endpoint=endpoint)
            start_watchdog(session)
            show_recovery(record_path, session)
            upload_batch(session, batch_id, m, base)
            reply = start_batch(record)
            if reply.get("started") is not True:
                raise RuntimeError(f"遠端拒絕啟動本批：{reply.get('reason')}")
            monitor(record, base, args.max_minutes)


def identify_runtime(rows, session):
    match = [(endpoint, gpu) for name, endpoint, gpu in rows if name == session]
    if len(match) != 1 or match[0][1] not in GPUS:
        raise RuntimeError(f"未在 Colab 確認到支援的 runtime：{session}；請查 sessions")
    return match[0]


def show_recovery(record_path, session):
    script = Path(__file__).resolve()
    print(f'中斷後只續監或下載：python "{script}" resume "{record_path}"', flush=True)
    print(f'不用保留 GPU：python "{script}" stop --session {session}', flush=True)


def resume(args):
    path = Path(args.record).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    batch_id, session = record["batch_id"], record["session"]
    if path.parent.name != ".h3-batches" or path.name != f"{batch_id}.json":
        raise RuntimeError("請指定 run 印出的 .h3-batches/<批次 ID>.json")
    with local_lock(path.parent / "output.lock"), local_lock(keep_file(session).with_suffix(".lock")):
        endpoint, _ = identify_runtime(runtime_rows()[0], session)
        if endpoint != record["endpoint"]:
            raise RuntimeError("runtime 已更換，不能續監舊批次")
        show_recovery(path, session)
        monitor(record, path.parent.parent, args.max_minutes)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    r = sub.add_parser("run")
    r.add_argument("manifest")
    r.add_argument("--gpu")
    r.add_argument("--session")
    r.add_argument("--idle-minutes", type=float, default=None,
                   help="做完先留著 GPU 等下一批，幾分鐘沒有新工作就自動關（預設 L4=15、A100/G4=10；0＝做完馬上關）")
    r.add_argument("--stop-after", action="store_true", help="做完馬上釋放 GPU")
    r.add_argument("--max-minutes", type=float, default=120)
    reuse = r.add_mutually_exclusive_group()
    reuse.add_argument("--force", action="store_true", help="重新生成（預設行為，保留相容性）")
    reuse.add_argument("--skip-existing", action="store_true", help="明確採用 out/ 內已有成品，不重新生成")
    r.add_argument("--with-pictures", action="store_true", help="也預裝圖片工作流需要的模型")
    r.add_argument("--confirmed", action="store_true", help="使用者已經確認過規格（兩支以上影片必填）")
    rr = sub.add_parser("resume", help="只續監或下載已啟動的批次")
    rr.add_argument("record", help="run 印出的 .h3-batches/<批次 ID>.json")
    rr.add_argument("--max-minutes", type=float, default=120)
    sub.add_parser("sessions")
    k = sub.add_parser("keep")
    k.add_argument("--session", required=True)
    k.add_argument("--minutes", type=float, required=True)
    w = sub.add_parser("_watchdog")
    w.add_argument("--session", required=True)
    s = sub.add_parser("stop")
    s.add_argument("--session")
    s.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.cmd == "check":
        print("colab：", colab_exe())
        print("登入：", "✅ 已登入" if token_ok() else "❌ 還沒登入（python colab_login.py url）")
        if token_ok():
            units, raw = usage()
            print("運算單元：", units if units is not None else raw[-400:])
            alive, _ = sessions()
            print("開著的 runtime：", alive or "沒有")
    elif a.cmd == "run":
        run(a)
    elif a.cmd == "resume":
        resume(a)
    elif a.cmd == "sessions":
        print(sessions()[1])
        for f in sorted(KEEP.glob("*.json")) if KEEP.exists() else []:
            info = json.loads(f.read_text(encoding="utf-8"))
            left = "跑批次中" if info.get("expires") is None else f"{max(0, info['expires'] - time.time()) / 60:.0f} 分鐘後自動關閉"
            print(f"⏸️ {info['session']}：{left}")
    elif a.cmd == "keep":
        minutes = min(a.minutes, 240)  # at most 4 hours: a forgotten GPU still stops by itself
        with local_lock(keep_file(a.session).with_suffix(".lock")):
            endpoint, _ = identify_runtime(runtime_rows()[0], a.session)
            set_keep(a.session, time.time() + minutes * 60, endpoint=endpoint)
            start_watchdog(a.session)
        print(f"⏸️ {a.session} 再留 {minutes:g} 分鐘，之後沒有新工作就自動關閉")
    elif a.cmd == "_watchdog":
        watchdog(a.session)
    elif a.cmd == "stop":
        names = sessions()[0] if a.all else [a.session]
        if names == [None]:
            sys.exit("要指定 --session NAME 或 --all")
        errors = []
        for n in names:
            try:
                with local_lock(keep_file(n).with_suffix(".lock")):
                    release(n)
            except RuntimeError as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("；".join(errors))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError) as exc:
        sys.exit(f"❌ {exc}")
