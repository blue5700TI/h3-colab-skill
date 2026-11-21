# Concatenated with BATCH_ID and ROOT by h3run; executed on the Colab VM.
import fcntl
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path


def launch(root, batch_id):
    job = root / batch_id
    bundle = root / f"{batch_id}.tar.gz"
    accepted = job / "accepted.json"
    if accepted.is_file():
        return json.loads(accepted.read_text())  # never start the same batch twice
    with (root / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            bundle.unlink(missing_ok=True)
            return dict(started=False, batch_id=batch_id, reason="another batch is already running")
        # Also guard against a worker from an older h3run before batch isolation existed.
        if subprocess.run(["pgrep", "-f", r"/content/h3job/remote_job\.py"], stdout=subprocess.DEVNULL).returncode == 0:
            bundle.unlink(missing_ok=True)
            return dict(started=False, batch_id=batch_id, reason="a legacy batch is already running")
        job.mkdir(exist_ok=False)
        with tarfile.open(bundle, "r:gz") as archive:
            archive.extractall(job, filter="data")
        bundle.unlink()
        # The worker inherits the lock so another launch cannot overwrite its ComfyUI state.
        env = dict(os.environ, H3JOB_DIR=str(job), H3JOB_BATCH_ID=batch_id)
        with (job / "remote_job.log").open("w") as log:
            worker = subprocess.Popen([sys.executable, str(job / "remote_job.py")], cwd=job, env=env,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                      pass_fds=(lock.fileno(),))
        record = dict(started=True, batch_id=batch_id, pid=worker.pid)
        tmp = accepted.with_suffix(".tmp")
        tmp.write_text(json.dumps(record))
        tmp.replace(accepted)
        return record


if "BATCH_ID" in globals():
    print("H3START " + json.dumps(launch(Path(ROOT), BATCH_ID)))
