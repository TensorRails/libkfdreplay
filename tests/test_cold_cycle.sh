#!/usr/bin/env bash
# GPU-free integration test of scripts/cold_cycle.sh under real DMTCP:
# single cycle, chain mode, launch-only + restart-only, negative flag, watchdog.
# Requires DMTCP (see scripts/install_dmtcp.sh) and python3.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/.." && pwd)
# shellcheck source=scripts/env.sh
. "$repo/scripts/env.sh"
kfdr_check_env

work=$(mktemp -d "${TMPDIR:-/tmp}/kfdr-cold-test.XXXXXX")
trap 'rm -rf "$work"' EXIT

cat >"$work/prog.py" <<'PY'
import ctypes, os, sys

ckdir = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "ok"
lib = ctypes.CDLL(None)
lib.kfdr_version.restype = ctypes.c_char_p
assert lib.kfdr_version() != b""
assert lib.kfdr_state() == 0

if mode == "hang_launch":
    import subprocess, time
    subprocess.Popen(["python3", "-c", "import time; time.sleep(300)"])
    time.sleep(300)

chain_file = os.path.join(ckdir, "chain_left")
gen = 0
while True:
    assert lib.kfdr_checkpoint(1) == 0
    assert lib.kfdr_state() == 2
    if lib.dmtcp_checkpoint() == 1:
        os._exit(0)
    assert lib.kfdr_restore() == 0
    assert lib.kfdr_state() == 3
    gen += 1
    if mode == "hang_restart":
        import time
        time.sleep(300)
    if os.environ.get("KFDR_NOCOPY") == "1":
        if mode == "abort":
            os.abort()
        print("RESULT FAIL", flush=True)
        sys.exit(1)
    left = int(open(chain_file).read()) if os.path.exists(chain_file) else 0
    if left <= 0:
        break
    open(chain_file, "w").write(str(left - 1))

print(f"gen={gen}", flush=True)
print("RESULT PASS", flush=True)
PY

# Outer wrapper whose argv contains both prog.py and the work directory (matches examples/*/run.sh).
cat >"$work/wrap.sh" <<SH
#!/usr/bin/env bash
set -euo pipefail
exec "$repo/scripts/cold_cycle.sh" -t "\${WD:-30}" -d "\$1" -c "\${CHAIN:-0}" \${EXTRA:-} -- python3 "$work/prog.py" "\$1" "\${2:-ok}"
SH
chmod +x "$work/wrap.sh"

out=$("$work/wrap.sh" "$work/single" 2>&1)
echo "$out" | grep -q "RESULT PASS"
if echo "$out" | grep -q "warning:"; then
    echo "test_cold_cycle: unexpected orphan warning: $out" >&2
    exit 1
fi

CHAIN=2 "$work/wrap.sh" "$work/chain" >/dev/null
grep -q "gen=3" "$work/chain/dst_2.log"

"$repo/scripts/cold_cycle.sh" -L -t 30 -d "$work/split" -- python3 "$work/prog.py" "$work/split" >/dev/null
"$repo/scripts/cold_cycle.sh" -R -t 30 -d "$work/split" >/dev/null

set +e
EXTRA="-E KFDR_NOCOPY=1" "$work/wrap.sh" "$work/neg-fail" ok >/dev/null 2>&1
rc_fail=$?
EXTRA="-E KFDR_NOCOPY=1" "$work/wrap.sh" "$work/neg-abort" abort >/dev/null 2>&1
rc_abort=$?
WD=2 "$work/wrap.sh" "$work/wd-launch" hang_launch >/dev/null 2>&1
rc_wd1=$?
WD=2 "$work/wrap.sh" "$work/wd-restart" hang_restart >/dev/null 2>&1
rc_wd2=$?
set -e

[ "$rc_fail" -eq 1 ] && [ "$rc_abort" -eq 1 ] && [ "$rc_wd1" -eq 3 ] && [ "$rc_wd2" -eq 1 ]
grep -q "WATCHDOG launch fired" "$work/wd-launch/watchdog.txt"
grep -q "WATCHDOG restart0 fired" "$work/wd-restart/watchdog.txt"

echo "test_cold_cycle: ok"
