// Глобальні змінні
let model = null;
let scalerParams = null;
let modelLoaded = false;

// Завантаження параметрів нормалізації
async function loadScalerParams() {
    try {
        console.log('Початок завантаження параметрів нормалізації...');
        const response = await fetch('./scaler_params.json');
        if (!response.ok) {
            throw new Error(`Помилка HTTP: ${response.status}`);
        }
        scalerParams = await response.json();
        console.log('Параметри нормалізації завантажені успішно');
        return true;
    } catch (error) {
        console.error('Помилка завантаження параметрів нормалізації:', error);
        document.getElementById('model-status').textContent = 'Помилка завантаження параметрів нормалізації';
        return false;
    }
}

// Завантаження моделі
async function loadModel() {
    try {
        console.log('Початок завантаження моделі...');

        // Завантажуємо параметри нормалізації
        const paramsLoaded = await loadScalerParams();
        if (!paramsLoaded) {
            throw new Error('Не вдалося завантажити параметри нормалізації');
        }

        // Завантажуємо модель TensorFlow.js
        // model = await tf.loadLayersModel('./tfjs_model/model.json');
        model = await tf.loadGraphModel('./tfjs_model/model.json');

        // Перевіряємо архітектуру моделі
        console.log('Архітектура моделі:');
        model.summary();

        // Оновлюємо статус
        document.getElementById('model-status').textContent = 'Модель завантажена успішно!';
        document.getElementById('predict-btn').disabled = false;
        modelLoaded = true;

        console.log('Модель завантажена успішно');

        // Додаємо тестовий прогноз для перевірки
        setTimeout(testPrediction, 1000);

    } catch (error) {
        console.error('Помилка завантаження моделі:', error);
        document.getElementById('model-status').textContent = 'Помилка завантаження моделі: ' + error.message;
    }
}

// Тестовий прогноз для перевірки роботи моделі
async function testPrediction() {
    if (!modelLoaded) return;

    try {
        console.log('Виконуємо тестовий прогноз...');
        const testInput = [20, 5, 3, 2]; // тестові дані

        // Нормалізація вхідних даних
        const normalizedInput = normalizeInput(testInput);
        const inputTensor = tf.tensor2d([normalizedInput]);

        // Прогнозування
        const predictionTensor = model.predict(inputTensor);
        const predictionData = await predictionTensor.dataSync();

        // Денормалізація результатів
        const results = denormalizeOutput(Array.from(predictionData));

        console.log('Тестовий прогноз успішний:', results);

        // Очищення пам'яті
        inputTensor.dispose();
        predictionTensor.dispose();

    } catch (error) {
        console.error('Помилка тестового прогнозу:', error);
    }
}

// Функція прогнозування
async function predict() {
    if (!modelLoaded) {
        alert('Модель ще не завантажена!');
        return;
    }

    // Отримання вхідних даних (приклад)
    const inputData = [20, 5, 3, 2]; // температура, опади, вітер, сезон

    try {
        // Нормалізація вхідних даних
        const normalizedInput = normalizeInput(inputData);
        const inputTensor = tf.tensor2d([normalizedInput]);

        // Прогнозування
        const predictionTensor = model.predict(inputTensor);
        const predictionData = await predictionTensor.dataSync();

        // Денормалізація результатів
        const results = denormalizeOutput(Array.from(predictionData));

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
        document.getElementById('results').innerHTML = `<p>Помилка прогнозування: ${error.message}</p>`;
    }
}

// Ініціалізація при завантаженні сторінки
document.addEventListener('DOMContentLoaded', function () {
    console.log('Сторінка завантажена, початок ініціалізації...');
    loadModel();
    document.getElementById('predict-btn').addEventListener('click', predict);
});