
Get-ChildItem ".\" -Filter *.tfa | 
Foreach-Object {
    

    &"D:\Icarian\Dropbox\Mis Documentos\Mis Documentos\Doctorado\Tesis\SARELI\SARELI\SARELI\bin\Debug\sarely.exe" -in $_.FullName -in -r 3,10 -r1 .75 -r2 .375 -sp
    move *.fasta ..\out\

    
}


