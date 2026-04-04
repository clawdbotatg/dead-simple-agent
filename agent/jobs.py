"""
jobs.py - Generic LeftClaw job watcher and dispatcher.

Replaces per-agent watchJobs / doNextJob / keepRunning scripts with a single
configurable class.

Usage:
    from agent.jobs import JobWatcher

    def my_prompt(job, attempt, resume):
        return f"Do job #{job['id']}..."

    watcher = JobWatcher(
        worker_address="0x862b...",
        service_type_id=7,
        prompt_builder=my_prompt,
    )
    watcher.run()
"""

import os
import subprocess
import sys
import time
from datetime import datetime

from .leftclaw import _fetch_contract, get_active_jobs, get_ready_jobs, is_job_complete, worker_address as _derive_address


def _log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _load_env(script_dir):
    env_file = os.path.join(script_dir, ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


class JobWatcher:
    """Poll LeftClaw for jobs and dispatch the agent.

    Args:
        service_type_id: LeftClaw service type to watch (e.g. 7=research, 5=QA).
        prompt_builder: Callable(job_dict, attempt_int, resume_bool) -> str.
            Builds the prompt passed to run.py for each job dispatch.
        poll_interval: Seconds between polls when idle (default 60).
        agent_timeout: Max seconds for a single agent run (default 600).
        max_retries: Max attempts per job within a single cycle (default 2).
        max_resume_cycles: Max times to resume the same in-progress job before
            halting. Prevents infinite loops when a job can't complete (default 5).
        script_dir: Working directory where run.py lives (default CWD).
        logs_dir: Directory for log files (default <script_dir>/logs).
    """

    def __init__(
        self,
        service_type_id,
        prompt_builder,
        poll_interval=60,
        agent_timeout=600,
        max_retries=2,
        max_resume_cycles=3,
        script_dir=None,
        logs_dir=None,
    ):
        self.worker_address = None  # derived from ETH_PRIVATE_KEY at runtime
        self.service_type_id = service_type_id
        self.prompt_builder = prompt_builder
        self.poll_interval = poll_interval
        self.agent_timeout = agent_timeout
        self.max_retries = max_retries
        self.max_resume_cycles = max_resume_cycles
        self.script_dir = script_dir or os.getcwd()
        self.logs_dir = logs_dir or os.path.join(self.script_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        self._resume_counts = {}  # job_id -> number of resume cycles

    def _dispatch(self, job, attempt=1, resume=False):
        """Run the agent once for a job. Returns True if job completed on-chain."""
        job_id = job["id"]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        tag = "resume" if resume else f"attempt{attempt}"
        log_file = os.path.join(self.logs_dir, f"job-{job_id}-{tag}-{timestamp}.log")

        prompt = self.prompt_builder(job, attempt, resume)

        _log(f"Dispatching agent for Job #{job_id} ({tag}) -> {log_file}")

        with open(log_file, "w") as lf:
            lf.write(f"=== Job #{job_id} — {tag} — {timestamp} ===\n")
            lf.write(f"Client: {job.get('client', 'unknown')}\n---\n")
            lf.flush()

            try:
                proc = subprocess.run(
                    [sys.executable, "run.py", prompt],
                    cwd=self.script_dir,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    timeout=self.agent_timeout,
                )
                lf.write(f"\n=== Exit code: {proc.returncode} ===\n")
            except subprocess.TimeoutExpired:
                lf.write(f"\n=== TIMED OUT after {self.agent_timeout}s ===\n")
                _log(f"Job #{job_id} {tag} timed out")
                return False

        _log(f"Job #{job_id} {tag} agent exited. Checking on-chain...")
        time.sleep(5)

        if is_job_complete(job_id):
            _log(f"Job #{job_id} COMPLETED on-chain.")
            return True
        else:
            _log(f"Job #{job_id} NOT complete on-chain after {tag}.")
            return False

    def _try_job(self, job, resume=False):
        """Attempt a job up to max_retries times."""
        job_id = job["id"]
        for attempt in range(1, self.max_retries + 1):
            success = self._dispatch(job, attempt, resume=resume)
            if success:
                return True
            if attempt < self.max_retries:
                _log(f"Retrying Job #{job_id} in 10s...")
                time.sleep(10)
            else:
                _log(f"Job #{job_id} FAILED after {self.max_retries} attempts. Moving on.")
        return False

    def run(self):
        """Start the polling loop. Blocks forever (Ctrl-C to stop)."""
        _load_env(self.script_dir)
        _fetch_contract()
        self.worker_address = _derive_address()

        _log(f"Watching for jobs (Service Type {self.service_type_id})...")
        _log(f"Worker: {self.worker_address}")
        _log(f"Polling every {self.poll_interval}s | Max retries: {self.max_retries} | "
             f"Max resume cycles: {self.max_resume_cycles} | Timeout: {self.agent_timeout}s")
        _log(f"Logs: {self.logs_dir}/")

        try:
            self._poll_loop()
        except KeyboardInterrupt:
            _log("Stopped.")

    def _poll_loop(self):
        while True:
            # Priority 1: resume IN_PROGRESS jobs already assigned to us
            active = get_active_jobs(self.worker_address, self.service_type_id)
            if active:
                job = active[0]
                job_id = job["id"]

                cycle = self._resume_counts.get(job_id, 0) + 1
                if cycle > self.max_resume_cycles:
                    _log(f"⛔ Job #{job_id} hit max resume cycles ({self.max_resume_cycles}). "
                         f"HALTING — manual intervention required. "
                         f"The job is still in-progress on-chain.")
                    time.sleep(self.poll_interval * 10)
                    continue

                self._resume_counts[job_id] = cycle
                _log(f"Found IN-PROGRESS job assigned to us: #{job_id} — "
                     f"resume cycle {cycle}/{self.max_resume_cycles}")
                success = self._try_job(job, resume=True)
                if success:
                    self._resume_counts.pop(job_id, None)
                time.sleep(10)
                continue

            # Priority 2: pick up new open jobs
            ready = get_ready_jobs(self.service_type_id)
            if ready:
                job = ready[0]
                job_id = job.get("id")

                # Re-verify the job is still in /api/job/ready before accepting.
                # Jobs can be flagged by the sanitizer between poll cycles.
                time.sleep(2)
                still_ready = get_ready_jobs(self.service_type_id)
                ready_ids = {j.get("id") for j in still_ready}
                if job_id not in ready_ids:
                    _log(f"Job #{job_id} no longer in /api/job/ready — skipping (may have been flagged)")
                    time.sleep(10)
                    continue

                _log(f"Found new job: #{job_id}")
                self._try_job(job)
                time.sleep(10)
            else:
                _log("No jobs (active or open). Waiting...")
                time.sleep(self.poll_interval)
