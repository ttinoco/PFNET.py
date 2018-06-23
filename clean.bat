echo "cleaning PFNET ..."
del /s /f %~dp0*.pyc
del /s /f %~dp0*.dll
del /s /f %~dp0*.a
del pfnet\cpfnet.c
rmdir /s /q %~dp0build
rmdir /s /q %~dp0dist
rmdir /s /q %~dp0PFNET.egg-info
rmdir /s /q %~dp0lib\pfnet
rmdir /s /q %~dp0lib\PFNET