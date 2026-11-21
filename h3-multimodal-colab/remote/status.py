# Concatenated with BATCH_ID and ROOT by h3run; executed on the Colab VM.
import json
from pathlib import Path


def status(root, batch_id):
    job = root / batch_id
    status_path = job / "status.json"
    if status_path.is_file():
        st = json.loads(status_path.read_text(encoding="utf-8"))
    elif not (job / "accepted.json").is_file():
        return dict(batch_id=batch_id, stage="not accepted", jobs={}, done=False)
    else:
        st = dict(batch_id=batch_id, stage="starting", jobs={}, done=False)
    # Do not replace the identity read from status.json with the requested ID.
    if not st.get("done") and (job / "accepted.json").is_file():
        pid = json.loads((job / "accepted.json").read_text())["pid"]
        try:
            args = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            args = []
        if str(job / "remote_job.py").encode() not in args:
            log = job / "remote_job.log"
            with log.open("rb") as f:
                f.seek(max(0, log.stat().st_size - 1500))
                tail = f.read().decode(errors="replace")
            # Worker may have committed its final status since the first read.
            if status_path.is_file():
                st = json.loads(status_path.read_text(encoding="utf-8"))
            if not st.get("done"):
                st.update(done=True, stage="crashed", error="the batch process exited\n" + tail)
    st["log"] = st.get("log", [])[-8:]
    return st


if "BATCH_ID" in globals():
    print("H3STATUS " + json.dumps(status(Path(ROOT), BATCH_ID), ensure_ascii=False))
