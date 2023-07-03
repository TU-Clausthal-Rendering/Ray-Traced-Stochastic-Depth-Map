@echo off

REM Set the folder paths
set "releaseFolder=../../build/windows-vs2022\bin\Release\data\NeuralNet"
set "debugFolder=../../build/windows-vs2022\bin\Debug\data\NeuralNet"

REM Check if the release folder exists
if not exist "%releaseFolder%" (
  echo Release folder does not exist.
) else (
  REM Delete the contents of the release folder
  del /q "%releaseFolder%\*.*" > nul 2>&1
)

REM Check if the debug folder exists
if not exist "%debugFolder%" (
  echo Debug folder does not exist.
) else (
  REM Delete the contents of the debug folder
  del /q "%debugFolder%\*.*" > nul 2>&1
)
