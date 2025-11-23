#!/bin/bash
# Script de verificación para Railway

echo "🚀 Verificando configuración para Railway..."

# Verificar archivos requeridos
files=("Procfile" "railway.json" "requirements.txt" "main.py")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file existe"
    else
        echo "❌ $file NO encontrado"
        exit 1
    fi
done

# Verificar que Python está instalado
if command -v python3 &> /dev/null; then
    echo "✅ Python3 instalado: $(python3 --version)"
else
    echo "❌ Python3 no encontrado"
    exit 1
fi

echo ""
echo "✅ Todo listo para desplegar en Railway!"
echo ""
echo "Próximos pasos:"
echo "1. Asegúrate de que tu código esté en GitHub"
echo "2. Ve a railway.app y crea un nuevo proyecto"
echo "3. Conecta tu repositorio de GitHub"
echo "4. Railway detectará automáticamente la configuración"
echo "5. ¡Tu API estará en línea en minutos!"
