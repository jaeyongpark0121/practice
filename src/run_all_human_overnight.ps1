# Run a batch script (src/run_all_human.sh by default) inside v2d-ubuntu while holding a Windows
# "system required" power request, so the laptop does not idle-sleep before the batch finishes.
# The request ends with this process and no power settings are changed. A closed lid can still
# put the laptop to sleep.
#
#     powershell -NoProfile -ExecutionPolicy Bypass -File src\run_all_human_overnight.ps1
#     powershell -NoProfile -ExecutionPolicy Bypass -File src\run_all_human_overnight.ps1 -Script src/run_all_object.sh -Log work/object/run_all.log

param(
    [string]$Script = "src/run_all_human.sh",
    [string]$Log = "work/human/run_all.log"
)

Add-Type -Namespace V2D -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
$ES_CONTINUOUS = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]1

[V2D.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
try {
    wsl.exe -d v2d-ubuntu --cd /mnt/c/Workspace/IIIXR/V2DChallenge --exec bash -lc "bash $Script >> $Log 2>&1"
} finally {
    [V2D.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
}
