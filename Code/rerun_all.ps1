# =============================================================================
#  rerun_all.ps1  --  regenerate Phases 2 -> 7 in ONE Python environment.
#
#  WHY THIS EXISTS (audit vii, 2026-08-23)
#  ---------------------------------------
#  The committed artefacts were not all produced by the same interpreter. The
#  Phase-5 predictions were fitted under CPython 3.10 (2026-08-18) and the
#  Phase-7 combined model under 3.12 (2026-08-23), so Phase 7's "training
#  inherited unchanged" claim is false on the files on disk: rebuilding the
#  Phase-5 point forecast from Phase 7's dropout-off column misses it by up to
#  18% on 2020-03-03, against the ~1e-6 a same-environment refit reproduces to.
#  This script regenerates every downstream artefact with one interpreter --
#  the checked-in .venv (Python 3.12.1) -- in dependency order.
#
#  WHAT TO EXPECT
#  --------------
#  * Runtime is roughly 45-75 minutes on CPU, unattended. Phase 3's 18-config
#    hyperparameter sweep is the bulk of it; pass -NoSweep to skip it (the
#    sweep table and heatmap then keep their old provenance).
#  * THE PUBLISHED NUMBERS WILL MOVE SLIGHTLY. m03/m05/m06 become 3.12 fits, so
#    QLIKE figures can shift in the third decimal. Every milestone, table and
#    quoted number in ROADMAP.md needs re-checking afterwards.
#  * Phase 7 now hard-fails if its model is not the Phase-5 model. Running
#    run_combined on its own against today's m05 WILL raise, by design.
#
#  USAGE (from the Code\ directory, in PowerShell)
#  ----------------------------------------------
#      .\rerun_all.ps1                 # full run, with the Phase-3 sweep
#      .\rerun_all.ps1 -NoSweep        # skip the sweep
#      .\rerun_all.ps1 -SkipTests      # skip the pytest step at the end
#
#  Per-step logs land in results\logs\rerun-<timestamp>\.
# =============================================================================

[CmdletBinding()]
param(
    [switch]$NoSweep,
    [switch]$SkipTests,
    [switch]$KeepPycache
)

$ErrorActionPreference = 'Stop'
$global:LASTEXITCODE = 0

$Root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $Root

$Py = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Py)) {
    throw "Interpreter not found at $Py. Run this from the Code\ directory, with the project .venv in place."
}

$Stamp  = Get-Date -Format 'yyyyMMdd-HHmmss'
$LogDir = Join-Path $Root "results\logs\rerun-$Stamp"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# --------------------------------------------------------------------------- #
function Invoke-Step {
    param(
        [Parameter(Mandatory)][string]   $Name,
        [Parameter(Mandatory)][string[]] $Arguments
    )
    $logFile = Join-Path $LogDir ("{0}.log" -f $Name)
    Write-Host ''
    Write-Host ('=' * 78) -ForegroundColor DarkGray
    Write-Host ("  {0}" -f $Name) -ForegroundColor Cyan
    Write-Host ("  {0} {1}" -f (Split-Path -Leaf $Py), ($Arguments -join ' ')) -ForegroundColor DarkGray
    Write-Host ('=' * 78) -ForegroundColor DarkGray

    $t0   = Get-Date
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'      # python logs to stderr; do not treat that as a failure
    & $Py @Arguments 2>&1 | Tee-Object -FilePath $logFile
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev

    if ($code -ne 0) {
        Write-Host ("  FAILED (exit {0}) -- see {1}" -f $code, $logFile) -ForegroundColor Red
        throw "Step '$Name' failed. Nothing after it has run; fix and re-launch."
    }
    Write-Host ("  done in {0:mm\:ss}" -f ((Get-Date) - $t0)) -ForegroundColor Green
}
# --------------------------------------------------------------------------- #

# ---- 0. environment identity, printed before anything is written ------------
Write-Host ''
Write-Host 'Interpreter for every step below:' -ForegroundColor Yellow
& $Py -c "import sys, platform; print('  ', sys.executable); print('   Python', sys.version.split()[0], '|', platform.platform())"
& $Py -c "import torch, numpy, pandas, scipy, arch, hmmlearn, statsmodels, sklearn; print('   torch', torch.__version__, '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| scipy', scipy.__version__); print('   arch', arch.__version__, '| hmmlearn', hmmlearn.__version__, '| statsmodels', statsmodels.__version__, '| sklearn', sklearn.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'The venv is missing a required package. Install requirements-lock.txt first.' }

if (-not $KeepPycache) {
    Write-Host ''
    Write-Host 'Clearing __pycache__ (stale 3.10 bytecode is what made this defect hard to see)...' -ForegroundColor Yellow
    Get-ChildItem -Path $Root -Directory -Recurse -Filter '__pycache__' -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notlike '*\.venv\*' } |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

$Overall = Get-Date

# ---- Phase 2: econometric baselines (writes m02, both profiles) -------------
Invoke-Step '01-phase2-econometric' @('-m','src.experiments.run_econometric')

# ---- Phase 3: LSTM baselines (needs m02; writes m03) ------------------------
$lstmArgs = @('-m','src.experiments.run_lstm')
if ($NoSweep) { $lstmArgs += '--no-sweep' }
Invoke-Step '02-phase3-lstm' $lstmArgs

# ---- Phase 4: regimes (writes m04 -- must precede Phases 5-7) ---------------
Invoke-Step '03-phase4-regimes' @('-m','src.experiments.run_regimes')

# ---- Phase 5: regime-aware models, both profiles (needs m03 + m04) ----------
Invoke-Step '04-phase5-regime-lstm'     @('-m','src.experiments.run_regime_lstm')
Invoke-Step '05-phase5-regime-lstm-hmm' @('-m','src.experiments.run_regime_lstm','--config','regime_lstm_hmm')

# ---- Phase 5 reports: GW and the DM regularity diagnostics ------------------
# Parquet-only and fast, but they must be regenerated or they describe the old fits.
Invoke-Step '06-report-gw'     @('-m','src.experiments.report_gw')
Invoke-Step '07-report-gw-hmm' @('-m','src.experiments.report_gw','--predictions','results/predictions/m05_regime_dl_intraday_2019_2022_hmm.parquet')
Invoke-Step '08-report-dm-diagnostics' @('-m','src.experiments.report_dm_diagnostics')

# ---- Phase 6: uncertainty quantification, both profiles ---------------------
Invoke-Step '09-phase6-uq'     @('-m','src.experiments.run_uq')
Invoke-Step '10-phase6-uq-hmm' @('-m','src.experiments.run_uq','--config','uq_hmm')

# ---- Phase 7: combined model, both profiles (NEVER --fast) ------------------
Invoke-Step '11-phase7-combined'     @('-m','src.experiments.run_combined')
Invoke-Step '12-phase7-combined-hmm' @('-m','src.experiments.run_combined','--config','combined_hmm')

# ---- Test suite -------------------------------------------------------------
if (-not $SkipTests) {
    Invoke-Step '13-pytest' @('-m','pytest','-q')
}

# ---- Verification -----------------------------------------------------------
Write-Host ''
Write-Host ('=' * 78) -ForegroundColor DarkGray
Write-Host '  VERIFICATION' -ForegroundColor Cyan
Write-Host ('=' * 78) -ForegroundColor DarkGray

$verify = @'
import glob, os, sys
import numpy as np, pandas as pd

TOL = 1e-3
ok = True

print("\n1. Phase-5 inheritance -- does the combined model reproduce Regime-LSTM-B?")
pairs = [("headline",   "results/predictions/m07_combined_intraday_2019_2022.parquet",
                        "results/predictions/m05_regime_dl_intraday_2019_2022.parquet"),
         ("comparator", "results/predictions/m07_combined_intraday_2019_2022_hmm.parquet",
                        "results/predictions/m05_regime_dl_intraday_2019_2022_hmm.parquet")]
for label, pf, bf in pairs:
    if not (os.path.exists(pf) and os.path.exists(bf)):
        print(f"   {label:11s} SKIP (missing parquet)"); continue
    p, b = pd.read_parquet(pf), pd.read_parquet(bf)
    det = np.exp(p["mc_mu_log_det"].astype(float) + 0.5 * p["mc_sd_aleatoric"].astype(float) ** 2)
    j = pd.concat([det.rename("det"), b["Regime-LSTM-B"].astype(float).rename("ref")],
                  axis=1, join="inner").dropna()
    dev = np.abs(np.log(j["det"].to_numpy() / j["ref"].to_numpy()))
    good = dev.max() <= TOL
    ok &= good
    print(f"   {label:11s} max |log dev| = {dev.max():.3g}  over {len(j)} days   "
          + ("PASS" if good else "FAIL"))
print(f"   (before the re-run the headline profile read 0.183 -- the defect this fixes)")

print("\n2. One environment across every phase?")
seen = {}
for phase_dir in sorted(glob.glob("experiments/*/")):
    runs = sorted(glob.glob(os.path.join(phase_dir, "run_*", "config.yaml")))
    if not runs:
        continue
    latest = runs[-1]
    env = {}
    inblock = False
    for line in open(latest, encoding="utf-8"):
        if line.startswith("_environment:"):
            inblock = True; continue
        if inblock:
            if line[:1].strip():          # dedented -> block finished
                break
            if ":" in line:
                k, v = line.strip().split(":", 1)
                env[k] = v.strip()
    phase = os.path.basename(phase_dir.rstrip("/\\"))
    tag = "py{}/torch{}".format(env.get("python", "?"), env.get("torch", "?"))
    seen.setdefault(tag, []).append(phase)
    print("   {:18s} {}".format(phase, tag))
if not seen:
    print("   -> no run snapshots found under experiments/. Cannot verify. FAIL")
    ok = False
elif len(seen) == 1 and "?" not in next(iter(seen)):
    print("   -> single environment across all phases. PASS")
elif "?" in " ".join(seen):
    print("   -> some snapshots predate the environment stamp; re-run those phases. "
          "(Phases run before this change carry no _environment block.)")
    ok = False
else:
    print(f"   -> {len(seen)} DIFFERENT environments still present. FAIL")
    ok = False

print("\n3. Headline QLIKE, for comparison against the previous run")
m = "results/tables/m07_combined_intraday_2019_2022_master.csv"
if os.path.exists(m):
    t = pd.read_csv(m)
    print(t[["rank_qlike", "model", "qlike"]].to_string(index=False))
    print("   previously: LSTM-RVonly 0.259640 | MC-Dropout-Regime-LSTM-B 0.259854 | "
          "Regime-LSTM-B 0.262115 | HAR-RV 0.274490")

sys.exit(0 if ok else 1)
'@

$prev = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$verify | & $Py -
$verifyCode = $LASTEXITCODE
$ErrorActionPreference = $prev

Write-Host ''
Write-Host ("Total elapsed: {0:hh\:mm\:ss}" -f ((Get-Date) - $Overall))
Write-Host ("Logs: {0}" -f $LogDir)

if ($verifyCode -ne 0) {
    Write-Host ''
    Write-Host 'VERIFICATION DID NOT PASS -- read section 1 and 2 above before committing.' -ForegroundColor Red
    exit 1
}

Write-Host ''
Write-Host 'All phases regenerated in one environment and verified.' -ForegroundColor Green
Write-Host 'Next, by hand (git is yours -- this script never touches it):' -ForegroundColor Yellow
Write-Host '  1. Diff results\ and re-check every number quoted in ROADMAP.md and the milestones.'
Write-Host '  2. Commit, then re-tag m01..m07 onto the commits that actually contain each phase.'
