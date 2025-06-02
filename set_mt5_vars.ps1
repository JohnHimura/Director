# Script para establecer variables de entorno permanentes para MT5

# Define los valores de tus credenciales y ruta de MT5
$mt5Login = "166542368"
$mt5Password = "StandarGem009*"
$mt5Server = "XMGlobal-MT5 2"
$mt5Path = "C:\Program Files\MetaTrader 5\terminal64.exe"

# Establece la variable de entorno MT5_LOGIN para el usuario actual
Write-Host "Estableciendo variable de entorno MT5_LOGIN..."
[System.Environment]::SetEnvironmentVariable("MT5_LOGIN", $mt5Login, "User")
Write-Host "MT5_LOGIN establecido."

# Establece la variable de entorno MT5_PASSWORD para el usuario actual
# Ten cuidado al manejar contraseñas. Considera métodos de almacenamiento más seguros para producción.
Write-Host "Estableciendo variable de entorno MT5_PASSWORD..."
[System.Environment]::SetEnvironmentVariable("MT5_PASSWORD", $mt5Password, "User")
Write-Host "MT5_PASSWORD establecido."

# Establece la variable de entorno MT5_SERVER para el usuario actual
Write-Host "Estableciendo variable de entorno MT5_SERVER..."
[System.Environment]::SetEnvironmentVariable("MT5_SERVER", $mt5Server, "User")
Write-Host "MT5_SERVER establecido."

# Establece la variable de entorno MT5_PATH para el usuario actual
Write-Host "Estableciendo variable de entorno MT5_PATH..."
[System.Environment]::SetEnvironmentVariable("MT5_PATH", $mt5Path, "User")
Write-Host "MT5_PATH establecido."

Write-Host "`nVariables de entorno establecidas con éxito para el usuario actual."
Write-Host "Puede que necesites reiniciar tu terminal o entorno de desarrollo para que los cambios surtan efecto."