$content = Get-Content docker-compose.yml
$newContent = @()
foreach ($line in $content) {
    $newContent += $line
    if ($line -eq "      - ollama-data:/root/.ollama") {
        $newContent += "      - ./nb/src/models:/models"
    }
}
$newContent | Set-Content docker-compose.yml
