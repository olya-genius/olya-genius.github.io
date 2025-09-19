let model = null;
let scalerParams = null;

document.addEventListener('DOMContentLoaded', async function () {
    await loadScalerParams();
    await loadModel();
    setupEventListeners();
});

async function loadScalerParams() {
    try {
        const response = await fetch('scaler_params.json');
        scalerParams = await response.json();
        console.log('Параметри нормалізації завантажені');
    } catch (error) {
        console.error('Помилка завантаження параметрів нормалізації:', error);
    }
}

async function loadModel() {
    const statusElement = document.getElementById('model-status');
    try {
        statusElement.textContent = 'Завантаження моделі...';
        statusElement.className = 'model-status status-loading';

        model = await tf.loadLayersModel('tfjs_model/model.json');

        statusElement.textContent = 'Модель завантажена';
        statusElement.className = 'model-status status-ready';
        document.getElementById('predict-btn').disabled = false;

        console.log('Модель успішно завантажена');
        updateModelInfo();

    } catch (error) {
        console.error('Помилка завантаження моделі:', error);
        statusElement.textContent = 'Помилка завантаження моделі';
        statusElement.className = 'model-status status-error';
    }
}

function updateModelInfo() {
    const modelInfo = document.getElementById('model-info');
    if (model) {
        modelInfo.innerHTML = `
            <p><strong>Архітектура:</strong> нейронна мережа з 4 входами та 5 виходами</p>
            <p><strong>Вхідні параметри:</strong> температура повітря, опади, швидкість вітру, сезон</p>
            <p><strong>Вихідні параметри:</strong> кисень, pH, каламутність, температура води, забруднювачі</p>
        `;
    }
}

function setupEventListeners() {
    document.getElementById('predict-btn').addEventListener('click', runPrediction);
    document.getElementById('simulate-btn').addEventListener('click', simulateSensorUpdate);
}

function simulateSensorUpdate() {
    // Генерація випадкових значень для демонстрації
    const airTemp = (Math.random() * 30 + 5).toFixed(1);
    const precipitation = (Math.random() * 50).toFixed(1);
    const windSpeed = (Math.random() * 15 + 1).toFixed(1);
    const season = Math.floor(Math.random() * 4);
    const seasonNames = ['Зима', 'Весна', 'Літо', 'Осінь'];

    document.getElementById('current-air-temp').textContent = airTemp;
    document.getElementById('current-precipitation').textContent = precipitation;
    document.getElementById('current-wind-speed').textContent = windSpeed;
    document.getElementById('current-season').textContent = `${seasonNames[season]} (${season})`;
}

function normalizeInput(data) {
    if (!scalerParams) return data;
    return data.map((value, index) => {
        return (value - scalerParams.X_mean[index]) / scalerParams.X_scale[index];
    });
}

function denormalizeOutput(data) {
    if (!scalerParams) return data;
    return data.map((value, index) => {
        return value * scalerParams.y_scale[index] + scalerParams.y_mean[index];
    });
}

async function runPrediction() {
    if (!model || !scalerParams) {
        alert('Модель або параметри нормалізації не завантажені!');
        return;
    }

    try {
        // Отримання поточних значень
        const airTemp = parseFloat(document.getElementById('current-air-temp').textContent);
        const precipitation = parseFloat(document.getElementById('current-precipitation').textContent);
        const windSpeed = parseFloat(document.getElementById('current-wind-speed').textContent);
        const seasonText = document.getElementById('current-season').textContent;
        const season = parseInt(seasonText.match(/\((\d)\)/)[1]);

        const inputData = [airTemp, precipitation, windSpeed, season];
        const normalizedInput = normalizeInput(inputData);

        const inputTensor = tf.tensor2d([normalizedInput]);
        const predictionTensor = model.predict(inputTensor);
        const predictionData = await predictionTensor.dataSync();
        const denormalizedPrediction = denormalizeOutput(Array.from(predictionData));

        // Оновлення інтерфейсу
        updatePredictionDisplay(denormalizedPrediction);
        updatePredictionExplanation(inputData, denormalizedPrediction);

        inputTensor.dispose();
        predictionTensor.dispose();

    } catch (error) {
        console.error('Помилка прогнозування:', error);
        alert('Сталася помилка під час прогнозування');
    }
}

function updatePredictionDisplay(prediction) {
    const [oxygen, ph, turbidity, waterTemp, pollutants] = prediction;
    const currentValues = getCurrentValues();

    updateParameterElement('predict-oxygen', oxygen, currentValues.oxygen);
    updateParameterElement('predict-ph', ph, currentValues.ph);
    updateParameterElement('predict-turbidity', turbidity, currentValues.turbidity);
    updateParameterElement('predict-temperature', waterTemp, currentValues.waterTemp);
    updateParameterElement('predict-pollutants', pollutants, currentValues.pollutants);
}

function getCurrentValues() {
    // В реальному додатку ці значення повинні отримуватися з датчиків
    return {
        oxygen: 8.0,
        ph: 7.0,
        turbidity: 15.0,
        waterTemp: 20.0,
        pollutants: 5.0
    };
}

function updateParameterElement(elementId, value, currentValue) {
    const element = document.getElementById(elementId);
    const valueElement = element.querySelector('span:last-child');

    valueElement.textContent = value.toFixed(2);

    element.classList.remove('increase', 'decrease');
    if (value > currentValue * 1.05) {
        element.classList.add('increase');
    } else if (value < currentValue * 0.95) {
        element.classList.add('decrease');
    }
}

function updatePredictionExplanation(inputData, prediction) {
    const [airTemp, precipitation, windSpeed, season] = inputData;
    const [oxygen, ph, turbidity, waterTemp, pollutants] = prediction;

    let explanation = '<h4>Результати прогнозу:</h4>';

    explanation += `<p>На основі вхідних даних (температура: ${airTemp}°C, опади: ${precipitation}мм, вітер: ${windSpeed}м/с) модель прогнозує:</p>`;

    if (oxygen < 6.0) {
        explanation += `<p>⚠️ <strong>Низький рівень кисню:</strong> ${oxygen.toFixed(2)} мг/л. Можлива загроза для водних організмів.</p>`;
    } else {
        explanation += `<p>✅ <strong>Рівень кисню:</strong> ${oxygen.toFixed(2)} мг/л (в межах норми).</p>`;
    }

    if (ph < 6.5 || ph > 8.5) {
        explanation += `<p>⚠️ <strong>Кислотність поза межами норми:</strong> pH = ${ph.toFixed(2)}. Оптимальний діапазон: 6.5-8.5.</p>`;
    } else {
        explanation += `<p>✅ <strong>Кислотність:</strong> pH = ${ph.toFixed(2)} (в межах норми).</p>`;
    }

    explanation += `<p>📊 <strong>Інші параметри:</strong> Каламутність: ${turbidity.toFixed(2)} NTU, Температура води: ${waterTemp.toFixed(2)}°C, Забруднювачі: ${pollutants.toFixed(2)} ppm</p>`;

    document.getElementById('prediction-explanation').innerHTML = explanation;
}