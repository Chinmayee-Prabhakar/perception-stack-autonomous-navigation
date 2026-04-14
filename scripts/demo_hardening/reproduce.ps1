param(
  [Parameter(Mandatory=$true)][string]$KittiImageArchive,
  [string]$OutputDirectory = 'outputs\demo_hardening_reproduced',
  [string]$PythonExecutable = ''
)

$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$outputRoot = [IO.Path]::GetFullPath((Join-Path $repo 'outputs'))
$archive = (Resolve-Path -LiteralPath $KittiImageArchive).Path
if ($PythonExecutable) {
  $python = (Resolve-Path -LiteralPath $PythonExecutable).Path
} else {
  $python = Join-Path $repo '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
  throw 'Provide an existing compatible Python with -PythonExecutable, or use the repository .venv.'
}
$target = if ([IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory } else { Join-Path $repo $OutputDirectory }
$target = [IO.Path]::GetFullPath($target)
if (-not $target.StartsWith($outputRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
  throw 'OutputDirectory must be a child of the ignored repository outputs directory.'
}
$relative = $target.Substring($outputRoot.Length + 1)
$first = ($relative -split '[\\/]')[0]
if ($first -in @('demo_validation', 'demo_hardening')) {
  throw 'Historical D0 and D1 output directories are read-only; choose a new output directory.'
}
if ((Test-Path -LiteralPath $target) -and (Get-ChildItem -LiteralPath $target -Force | Select-Object -First 1)) {
  throw 'OutputDirectory is not empty; choose a fresh isolated directory.'
}

$calibrationSource = Join-Path $repo 'datasets\kitti_object\training\calib\006781.txt'
$checkArgs = @()
if (Test-Path -LiteralPath $calibrationSource -PathType Leaf) {
  $checkArgs = @('--calibration-source', $calibrationSource)
}
& $python (Join-Path $PSScriptRoot 'check_assets.py') @checkArgs
if ($LASTEXITCODE -ne 0) { throw 'Pinned asset or calibration verification failed.' }

New-Item -ItemType Directory -Force -Path $target | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $target 'runtime') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $target 'runtime\matplotlib') | Out-Null
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:YOLO_CONFIG_DIR = Join-Path $target 'runtime'
$env:MPLCONFIGDIR = Join-Path $target 'runtime\matplotlib'
& $python (Join-Path $PSScriptRoot 'restore_inputs.py') --archive $archive --output (Join-Path $target 'inputs') 2>&1 |
  Tee-Object -FilePath (Join-Path $target 'restore_inputs.log')
if ($LASTEXITCODE -ne 0) { throw 'Selected input restoration failed.' }

$priorDirectory = (Get-Location).Path
try {
  Set-Location $target
  foreach ($fid in @('006263','000201','006781','003783','005481','000185','007200')) {
    $dir = Join-Path $target (Join-Path 'scenes' $fid)
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    & $python -m perception_stack.scene_cli `
      --input (Join-Path $target "inputs\$fid.png") `
      --config (Join-Path $repo 'configs\demo.json') `
      --detector-weights (Join-Path $repo 'models\yolo11n-seg.pt') `
      --depth-weights (Join-Path $repo 'models\depth_anything_v2_metric_vkitti_vits.pth') `
      --road-weights (Join-Path $repo 'models\twinlitenetplus_bdd100k_small.pth') `
      --json (Join-Path $dir 'scene.json') --overlay (Join-Path $dir 'overlay.png') `
      --association-json (Join-Path $dir 'road_association.json') `
      --drivable-mask (Join-Path $dir 'drivable_mask.png') `
      --lane-mask (Join-Path $dir 'lane_mask.png') --device cpu 2>&1 |
      Tee-Object -FilePath (Join-Path $dir 'run.log')
    if ($LASTEXITCODE -ne 0) { throw "Scene failed: $fid" }
  }
  $dir = Join-Path $target 'scenes\006781_calibrated'
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  & $python -m perception_stack.scene_cli `
    --input (Join-Path $target 'inputs\006781.png') `
    --config (Join-Path $repo 'configs\demo.json') `
    --detector-weights (Join-Path $repo 'models\yolo11n-seg.pt') `
    --depth-weights (Join-Path $repo 'models\depth_anything_v2_metric_vkitti_vits.pth') `
    --road-weights (Join-Path $repo 'models\twinlitenetplus_bdd100k_small.pth') `
    --intrinsics-json (Join-Path $repo 'configs\demo_hardening\006781_intrinsics.json') `
    --json (Join-Path $dir 'scene.json') --overlay (Join-Path $dir 'overlay.png') `
    --association-json (Join-Path $dir 'road_association.json') `
    --xz-plot (Join-Path $dir 'camera_xz.png') --device cpu 2>&1 |
    Tee-Object -FilePath (Join-Path $dir 'run.log')
  if ($LASTEXITCODE -ne 0) { throw 'Calibrated scene failed.' }
} finally {
  Set-Location $priorDirectory
}

& $python (Join-Path $PSScriptRoot 'verify_outputs.py') --output-dir $target --presentation-version r2 2>&1 |
  Tee-Object -FilePath (Join-Path $target 'verification.log')
if ($LASTEXITCODE -ne 0) { throw 'Reproduced output verification failed.' }
