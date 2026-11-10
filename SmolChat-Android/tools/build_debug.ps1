param(
    [string]$JavaHome = (Join-Path $env:USERPROFILE '.jdks\ms-17.0.16'),
    [int]$MaxWorkers = 4
)

$ErrorActionPreference = 'Stop'
$taskRepoRoot = Split-Path -Parent $PSScriptRoot
$taskJavaExe = Join-Path $JavaHome 'bin\java.exe'
if (-not (Test-Path -LiteralPath $taskJavaExe)) {
    throw 'JDK 17 was not found. Pass its installation directory using -JavaHome.'
}
if (-not (Test-Path -LiteralPath (Join-Path $taskRepoRoot 'local.properties'))) {
    throw 'Create local.properties with sdk.dir pointing to your Android SDK.'
}

$env:JAVA_HOME = $JavaHome
Push-Location $taskRepoRoot
try {
    # Native stderr may contain warnings; use Gradle's exit code to decide success.
    $ErrorActionPreference = 'Continue'
    & $taskJavaExe -Xmx64m -Xms64m -classpath gradle\wrapper\gradle-wrapper.jar `
        org.gradle.wrapper.GradleWrapperMain :app:assembleDebug :app:testDebugUnitTest `
        --console=plain "--max-workers=$MaxWorkers"
    $taskBuildExitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($taskBuildExitCode -ne 0) {
        throw "Android build failed with exit code $taskBuildExitCode."
    }
    Write-Output (Join-Path $taskRepoRoot 'app\build\outputs\apk\debug\app-debug.apk')
} finally {
    Pop-Location
}
