@echo off
echo ===================================================
echo     POST-TRAINING VALIDATION & BENCHMARK SUITE
echo ===================================================
echo.
echo Running Benchmarks (evaluate_baselines.py)...
python training/evaluate_baselines.py
if errorlevel 1 goto error

echo.
echo Running Reproducibility Verification...
python training/verify_reproducibility.py
if errorlevel 1 goto error

echo.
echo Running Visual/Behavioral Verification...
python training/verify_behavior.py
if errorlevel 1 goto error

echo.
echo Generating Plots...
python training/plot_metrics.py
if errorlevel 1 goto error

echo.
echo ===================================================
echo ALL TASKS COMPLETED SUCCESSFULLY.
echo Please review results in results/benchmarks/ and results/plots/
echo ===================================================
goto end

:error
echo.
echo [ERROR] A script failed to execute properly. Please check the output above.

:end
