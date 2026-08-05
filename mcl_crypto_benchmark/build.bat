@echo off
REM ============================================================
REM  MCL benchmark build script (Windows x64, MinGW-w64 g++)
REM  Requires g++ on PATH. See README.md section 2 for how to
REM  get a MinGW-w64 toolchain (conda m2w64 or MSYS2).
REM ============================================================
setlocal

where g++ >nul 2>nul
if errorlevel 1 (
    echo [ERROR] g++ not found on PATH.
    echo         See README.md section 2 to install a MinGW-w64 toolchain.
    exit /b 1
)

echo === building bench_apeg.exe ===
g++ -O3 -std=c++14 -DNDEBUG -DMCL_USE_XBYAK=1 -DMCL_MSM=0 ^
    -static -static-libgcc -static-libstdc++ ^
    -I include -I src ^
    bench_apeg.cpp src\fp.cpp src\asm\bint-x64-mingw.S ^
    -o bench_apeg.exe
if errorlevel 1 ( echo [ERROR] build failed & exit /b 1 )

echo === building curve_info.exe ===
g++ -O2 -std=c++14 -DNDEBUG -DMCL_USE_XBYAK=1 -DMCL_MSM=0 ^
    -static -static-libgcc -static-libstdc++ ^
    -I include -I src ^
    curve_info.cpp src\fp.cpp src\asm\bint-x64-mingw.S ^
    -o curve_info.exe
if errorlevel 1 ( echo [WARN] curve_info build failed ^(non-fatal^) )

echo.
echo === done. Run:  bench_apeg.exe bn254 ===
