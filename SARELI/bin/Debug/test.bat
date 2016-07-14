echo off
del "D:\Icarian\Dropbox\Mis Documentos\Mis Documentos\Doctorado\Tesis\SARELI\SARELI\SARELI\bin\Debug\kernel.ptx"
echo "Kernel.ptx eliminado"
copy "D:\Icarian\Dropbox\Mis Documentos\Mis Documentos\Doctorado\Tesis\Align4\Align4\x64\Debug\kernel.ptx" "D:\Icarian\Dropbox\Mis Documentos\Mis Documentos\Doctorado\Tesis\SARELI\SARELI\SARELI\bin\Debug"
echo "Kernel.ptx actualizado"
echo "Iniciando pruebas"
echo "------------------------"
echo "Cuda:"
sareli -in test.tfa -r1 .75 -r2 .375 -i 5 -s testa.txt -ptx kernel.ptx -r 3,4
echo "Serial:"
sareli -in test.tfa -r1 .75 -r2 .375 -i 5 -s testa.txt -r 3,4