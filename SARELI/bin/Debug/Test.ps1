
Get-ChildItem ".\" -Filter *.aln | 
Foreach-Object {
    

    &"D:\Icarian\Dropbox\Mis Documentos\Mis Documentos\Doctorado\Tesis\SARELI\SARELI\SARELI\bin\Debug\sarely.exe" -in $_.FullName -t -s t_Coffe.txt

    
}


