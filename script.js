// Глобальні змінні
let model = null;
let scalerParams = null;

// Завантаження параметрів нормалізації
async function loadScalerParams() {
    try {
        const response = await fetch('./scaler_params.json');
        scalerParams = await response.json();
        console.log('Параметри нормалізації завантажені');
    } catch (error) {
        console.error('Помилка завантаження параметрів нормалізації:', error);
    }
}

// Завантаження моделі
async function loadModel() {
    try {
        await loadScalerParams();
        model = await tf.loadLayersModel('./tfjs_model/model.json');
        document.getElementById('model-status').textContent = 'Модель завантажена!';
        document.getElementById('predict-btn').disabled = false;
        console.log('Модель успішно завантажена');
    } catch (error) {
        console.error('Помилка завантаження моделі:', error);
        document.getElementById('model-status').textContent = 'Помилка завантаження моделі';
    }
}

// Функція прогнозування
async function predict() {
    if (!model) {
        alert('Модель ще не завантажена!');
        return;
    }

    const inputData = [20, 5, 3, 2]; // температура, опади, вітер, сезон

    try {
        // Нормалізація вхідних даних
        const normalizedInput = inputData.map((val, i) =>
            (val - scalerParams.X_mean[i]) / scalerParams.X_scale[i]
        );

        const inputTensor = tf.tensor2d([normalizedInput]);
        const predictionTensor = model.predict(inputTensor);
        const predictionData = await predictionTensor.dataSync();

        // Денормалізація результатів
        const results = Array.from(predictionData).map((val, i) =>
            val * scalerParams.y_scale[i] + scalerParams.y_mean[i]
        );

        // Відображення результатів
        document.getElementById('results').innerHTML = `
            <h3>Результати прогнозу:</h3>
            <p>Кисень: ${results[0].toFixed(2)} мг/л</p>
            <p>pH: ${results[1].toFixed(2)}</p>
            <p>Каламутність: ${results[2].toFixed(2)} NTU</p>
            <p>Температура води: ${results[3].toFixed(2)} °C</p>
            <p>Забруднювачі: ${results[4].toFixed(2)} ppm</p>
        `;

        // Очищення пам'яті
        inputTensor.dispose();
        predictionTensor.dispose();
    } catch (error) {
        console.error('Помилка прогнозування:', error);
    }
}

// Ініціалізація при завантаженні сторінки
document.addEventListener('DOMContentLoaded', function () {
    loadModel();
    document.getElementById('predict-btn').addEventListener('click', predict);
});