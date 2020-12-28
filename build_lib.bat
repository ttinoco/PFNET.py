IF NOT EXIST "lib\pfnet" (
  cd lib
  7z x pfnet*.tar.gz
  move pfnet*.tar pfnet.tar
  7z x pfnet.tar
  del pfnet.tar
  for /d %%G in ("pfnet*") do move "%%~G" pfnet
  cd pfnet
  mkdir build
  cd build
  cmake -G"Visual Studio 16 2019" .. -A x64
  cmake --build . --config Release
  copy Release\*.lib ..\..\..\pfnet
  copy Release\*.dll ..\..\..\pfnet
  cd  ..\..\..
  python setup.py setopt --command build -o compiler -s msvc
)
