$missing = @("gene you want ex:ERR164550")
$missing | ForEach-Object {
    Start-Process "https://www.ebi.ac.uk/ena/browser/view/$_"
}
