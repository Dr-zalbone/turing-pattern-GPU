#!/usr/bin/env bash
# Self-test for the turing image. Run inside the container:
#   docker run ... <image> selftest       quiet: one line per section plus summary
#   docker run ... <image> selftest -v    also print every PASS and SKIP line
#
# Sections: toolchain, cli, gpu, then photo and video (only when a GPU is visible).
# Environment:
#   TURING_SELFTEST_VERBOSE=1  same as -v
#   TURING_REQUIRE_GPU=1       treat a missing GPU as a failure instead of a skip
# Exit status: 0 if no check failed, 1 otherwise.
set -uo pipefail

VERBOSE="${TURING_SELFTEST_VERBOSE:-0}"
for arg in "$@"; do
    case "$arg" in
        -v) VERBOSE=1 ;;
        *)  echo "usage: selftest.sh [-v]" >&2; exit 1 ;;
    esac
done

if [ -t 1 ]; then
    C_GREEN=$'\033[32m'; C_RED=$'\033[31m'; C_YELLOW=$'\033[33m'; C_RESET=$'\033[0m'
else
    C_GREEN=''; C_RED=''; C_YELLOW=''; C_RESET=''
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/turing-selftest.XXXXXX")" || { echo "selftest: mktemp failed" >&2; exit 1; }
trap 'rm -rf "$TMP_DIR"' EXIT

# ---- result bookkeeping -----------------------------------------------------
TOTAL_PASS=0; TOTAL_FAIL=0; TOTAL_SKIP=0
SEC=""; SEC_PASS=0; SEC_FAIL=0; SEC_SKIP=0

end_section() {
    [ -n "$SEC" ] || return 0
    printf '%-10s %d passed, %d failed, %d skipped\n' "$SEC" "$SEC_PASS" "$SEC_FAIL" "$SEC_SKIP"
    SEC=""
}

section() {  # section NAME: close the previous section and open NAME
    end_section
    SEC="$1"; SEC_PASS=0; SEC_FAIL=0; SEC_SKIP=0
    if [ "$VERBOSE" = 1 ]; then printf '\n== %s ==\n' "$SEC"; fi
}

pass() {
    TOTAL_PASS=$((TOTAL_PASS + 1)); SEC_PASS=$((SEC_PASS + 1))
    if [ "$VERBOSE" = 1 ]; then printf '  %s[PASS]%s %s\n' "$C_GREEN" "$C_RESET" "$*"; fi
}

fail() {  # failures are always printed in full
    TOTAL_FAIL=$((TOTAL_FAIL + 1)); SEC_FAIL=$((SEC_FAIL + 1))
    printf '  %s[FAIL]%s %s\n' "$C_RED" "$C_RESET" "$*"
}

skip() {
    TOTAL_SKIP=$((TOTAL_SKIP + 1)); SEC_SKIP=$((SEC_SKIP + 1))
    if [ "$VERBOSE" = 1 ]; then printf '  %s[SKIP]%s %s\n' "$C_YELLOW" "$C_RESET" "$*"; fi
}

# ---- helpers ----------------------------------------------------------------
# Names of the AMD GPUs rocminfo reports, comma-separated. gfx000 is the CPU
# agent and is ignored. Empty when no usable GPU is visible. ROCm uses /dev/kfd
# on Linux and /dev/dxg under WSL2; rocminfo decides whether either is usable.
gpu_names() {
    [ -e /dev/kfd ] || [ -e /dev/dxg ] || return 0
    local info
    info="$(rocminfo 2>/dev/null)"
    # Newer ROCm also lists ISA families like "gfx11-generic"; keep only real targets.
    grep -oE 'gfx[0-9a-z]+(-generic)?' <<<"$info" | grep -v -- '-generic$' | grep -vx 'gfx000' | sort -u | paste -sd, -
}
GPU_NAMES="$(gpu_names)"

probe_dims()   { ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$1"; }
probe_pix()    { ffprobe -v error -select_streams v:0 -show_entries stream=pix_fmt -of csv=p=0 "$1"; }
probe_rate()   { ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$1"; }
probe_frames() { ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$1"; }
has_stream()   { [ -n "$(ffprobe -v error -select_streams "$2" -show_entries stream=codec_type -of csv=p=0 "$1")" ]; }  # $2: a or v

check_dims() {  # check_dims FILE W H LABEL
    local dims
    if [ ! -s "$1" ]; then fail "$4: $1 missing or empty"; return 0; fi
    dims="$(probe_dims "$1")"
    if [ "$dims" = "${2}x${3}" ]; then pass "$4 is ${2}x${3}"
    else fail "$4: expected ${2}x${3}, got '$dims'"; fi
}

check_rate() {  # check_rate FILE EXPECTED
    local rate
    rate="$(probe_rate "$1")"
    if [ "$rate" = "$2" ]; then pass "frame rate is $2"
    else fail "frame rate: expected $2, got '$rate'"; fi
}

check_frames() {  # check_frames FILE MIN MAX
    local n
    n="$(probe_frames "$1")"; n="${n:-0}"
    if [ "$n" -ge "$2" ] && [ "$n" -le "$3" ]; then pass "frame count $n (expected $2-$3)"
    else fail "frame count: expected $2-$3, got $n"; fi
}

cli_expect() {  # cli_expect LABEL PATTERN [ARGS...]: must exit nonzero and print PATTERN
    local label="$1" pattern="$2" out rc
    shift 2
    out="$(turing "$@" 2>&1)"; rc=$?
    if [ "$rc" -ne 0 ] && grep -q "$pattern" <<<"$out"; then
        pass "$label: exit $rc, '$pattern' printed"
    else
        fail "$label: expected nonzero exit with '$pattern' (rc=$rc)"
    fi
}

# ---- sections ---------------------------------------------------------------
toolchain_tests() {
    section toolchain
    local bin ldd_out missing tool encoders
    bin="$(command -v turing 2>/dev/null)"
    if [ -n "$bin" ] && [ -x "$bin" ]; then
        pass "turing found at $bin"
        ldd_out="$(ldd "$bin" 2>&1)"
        if grep -q 'not found' <<<"$ldd_out"; then
            missing="$(grep 'not found' <<<"$ldd_out" | sed 's/^[[:space:]]*//; s/[[:space:]]*=>.*//' | paste -sd, -)"
            fail "unresolved shared libraries: $missing"
        else
            pass "all shared libraries resolve (ldd)"
        fi
    else
        fail "turing not found or not executable on PATH"
    fi

    for tool in ffmpeg ffprobe; do
        if command -v "$tool" >/dev/null 2>&1; then pass "$tool on PATH"
        else fail "$tool not on PATH"; fi
    done

    encoders="$(ffmpeg -hide_banner -encoders 2>/dev/null)"
    if grep -q 'libx264' <<<"$encoders"; then pass "ffmpeg has the libx264 encoder"
    else fail "ffmpeg has no libx264 encoder"; fi
}

cli_tests() {
    section cli
    cli_expect "no arguments" "Usage"
    cli_expect "video mode with 3 positionals" "Usage" in.mp4 out.mp4 320
    cli_expect "photo mode with out.txt" "extension" in.png out.txt
}

gpu_tests() {
    section gpu
    if [ -n "$GPU_NAMES" ]; then
        pass "AMD GPU visible: $GPU_NAMES"
    elif [ "${TURING_REQUIRE_GPU:-0}" = "1" ]; then
        fail "no AMD GPU visible (TURING_REQUIRE_GPU=1)"
    else
        skip "no AMD GPU visible"
        echo "selftest: warning: no AMD GPU visible, photo and video tests skipped." \
             "Run the container with: --device /dev/kfd --device /dev/dri --group-add video --group-add render" >&2
    fi
}

photo_tests() {
    section photo
    command -v turing >/dev/null 2>&1 || { fail "turing not on PATH"; return 0; }

    local in="$TMP_DIR/in.png" out="$TMP_DIR/out.png" out_jpg="$TMP_DIR/out.jpg"
    local log="$TMP_DIR/photo.log" stats ymin ymax pix
    if ! ffmpeg -v error -f lavfi -i testsrc=size=160x120 -frames:v 1 "$in"; then
        fail "could not generate photo test input"; return 0
    fi

    # 1) Auto-detected dimensions.
    if turing "$in" "$out" >"$log" 2>&1; then
        pass "turing in.png out.png exits 0"
        if grep -q 'Processed 160x120' "$log"; then pass "reports 'Processed 160x120'"
        else fail "no 'Processed 160x120' line in output: $(tail -n 1 "$log")"; fi
    else
        fail "turing in.png out.png failed: $(tail -n 1 "$log")"
    fi
    check_dims "$out" 160 120 "out.png"

    pix="$(probe_pix "$out")"
    case "$pix" in
        gray|gray8|ya8) pass "out.png pix_fmt is $pix" ;;
        *)              fail "out.png pix_fmt is '$pix', expected gray" ;;
    esac

    # Output must not be a single flat colour: YMAX has to exceed YMIN.
    stats="$(ffprobe -v error -f lavfi -i "movie=$out,signalstats" \
             -show_entries frame_tags=lavfi.signalstats.YMIN,lavfi.signalstats.YMAX \
             -of default=nw=1 2>/dev/null)"
    ymin="$(sed -n 's/^TAG:lavfi\.signalstats\.YMIN=//p' <<<"$stats" | head -n 1)"
    ymax="$(sed -n 's/^TAG:lavfi\.signalstats\.YMAX=//p' <<<"$stats" | head -n 1)"
    if [ -n "$ymin" ] && [ -n "$ymax" ] && [ "$ymax" -gt "$ymin" ]; then
        pass "out.png has contrast (YMIN=$ymin YMAX=$ymax)"
    else
        fail "out.png looks flat (YMIN='${ymin:-?}' YMAX='${ymax:-?}')"
    fi

    # 2) Explicit dimensions, JPEG output.
    if turing "$in" "$out_jpg" 160 120 >"$log" 2>&1 && [ -s "$out_jpg" ]; then
        pass "turing in.png out.jpg 160 120 writes a JPEG"
    else
        fail "turing in.png out.jpg 160 120 failed: $(tail -n 1 "$log")"
    fi
}

video_tests() {
    section video
    command -v turing >/dev/null 2>&1 || { fail "turing not on PATH"; return 0; }

    local in="$TMP_DIR/in.mp4" out="$TMP_DIR/out.mp4" out_na="$TMP_DIR/out_na.mp4"
    local ntsc_in="$TMP_DIR/ntsc.mp4" out_ntsc="$TMP_DIR/out_ntsc.mp4" log="$TMP_DIR/video.log"
    if ! ffmpeg -v error -f lavfi -i testsrc=size=160x120:rate=24 \
                -f lavfi -i sine=frequency=440:sample_rate=48000 \
                -t 1 -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "$in"; then
        fail "could not generate video test input"; return 0
    fi

    # 1) 24 fps source, 8-frame batches, audio copied.
    if turing "$in" "$out" 160 120 8 >"$log" 2>&1; then
        pass "turing in.mp4 out.mp4 160 120 8 exits 0"
        check_rate "$out" "24/1"
        check_frames "$out" 23 25
        if has_stream "$out" a; then pass "output has an audio stream"
        else fail "output has no audio stream"; fi
    else
        fail "turing in.mp4 out.mp4 160 120 8 failed: $(tail -n 1 "$log")"
    fi

    # 2) --no-audio (flag placed after the positionals, which is allowed).
    if turing "$in" "$out_na" 160 120 8 --no-audio >"$log" 2>&1 && [ -s "$out_na" ]; then
        pass "--no-audio run exits 0"
        if has_stream "$out_na" a; then fail "--no-audio output still has an audio stream"
        else pass "--no-audio output has no audio stream"; fi
    else
        fail "--no-audio run failed: $(tail -n 1 "$log")"
    fi

    # 3) NTSC rate (30000/1001) must survive exactly.
    if ! ffmpeg -v error -f lavfi -i testsrc=size=160x120:rate=30000/1001 -t 1 \
                -c:v libx264 -pix_fmt yuv420p "$ntsc_in"; then
        fail "could not generate NTSC test input"
    elif turing "$ntsc_in" "$out_ntsc" 160 120 4 >"$log" 2>&1; then
        check_rate "$out_ntsc" "30000/1001"
    else
        fail "NTSC run failed: $(tail -n 1 "$log")"
    fi
}

# ---- run --------------------------------------------------------------------
toolchain_tests
cli_tests
gpu_tests
if [ -n "$GPU_NAMES" ]; then
    photo_tests
    video_tests
else
    section photo; skip "photo mode: no AMD GPU"
    section video; skip "video mode: no AMD GPU"
fi
end_section

printf '\nturing self-test: %d passed, %d failed, %d skipped (gpu: %s)\n' \
    "$TOTAL_PASS" "$TOTAL_FAIL" "$TOTAL_SKIP" "${GPU_NAMES:-none}"

if [ "$TOTAL_FAIL" -eq 0 ]; then exit 0; fi
exit 1
