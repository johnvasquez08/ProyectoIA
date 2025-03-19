function updateCapture() {
    const captureImg = document.getElementById('capture');
    fetch('/capture')
        .then(response => {
            if (response.ok) {
                return response.blob();
            }
            throw new Error('No capture available');
        })
        .then(blob => {
            const url = URL.createObjectURL(blob);
            captureImg.src = url;
        })
        .catch(error => {
            console.error(error);
        });
}

function updateCentroid() {
    const centroidCoords = document.getElementById('centroid-coords');
    fetch('/centroid')
        .then(response => response.json())
        .then(data => {
            if (data.x !== null && data.y !== null) {
                centroidCoords.textContent = `Centroide: (${data.x}, ${data.y})`;
            } else {
                centroidCoords.textContent = 'Esperando detección...';
            }
        })
        .catch(error => {
            console.error(error);
        });
}

// Actualizar la captura y las coordenadas cada 2 segundos
setInterval(() => {
    updateCapture();
    updateCentroid();
}, 2000);