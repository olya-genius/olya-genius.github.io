let model;
let scalerParams;

// Загрузка модели и параметров нормализации
async function loadModel() {
    try {
        // Загружаем модель
        model = await tf.loadLayersModel("tfjs_model/model.json");

        // Загружаем параметры нормализации
        const resp = await fetch("scaler_params.json");
        scalerParams = await resp.json();

        document.getElementById("model-status").innerText = "Модель завантажена!";
        document.getElementById("predict-btn").disabled = false;
    } catch (err) {
        console.error("Помилка завантаження:", err);
        document.getElementById("model-status").innerText = "Помилка завантаження моделі";
    }
}

// Нормализация входа
function normalizeInput(input) {
    return input.map((v, i) => (v - scalerParams.X_mean[i]) / scalerParams.X_scale[i]);
}

// Денормализация выхода
function denormalizeOutput(output) {
    return output.map((v, i) => v * scalerParams.y_scale[i] + scalerParams.y_mean[i]);
}

// Прогноз
async function predict() {
    const input = [20, 5, 3, 2]; // пример: t=20, осадки=5, вітер=3, сезон=2
    const normInput = normalizeInput(input);
    const tensor = tf.tensor2d([normInput]);

    const pred = model.predict(tensor);
    const data = Array.from(await pred.data());
    const result = denormalizeOutput(data);

    document.getElementById("results").innerHTML = `
        <p>Кисень: ${result[0].toFixed(2)} мг/л</p>
        <p>pH: ${result[1].toFixed(2)}</p>
        <p>Каламутність: ${result[2].toFixed(2)} NTU</p>
        <p>Температура води: ${result[3].toFixed(2)} °C</p>
        <p>Забруднювачі: ${result[4].toFixed(2)} ppm</p>
    `;

    tensor.dispose();
    pred.dispose();
}

// Инициализация
document.addEventListener("DOMContentLoaded", () => {
    loadModel();
    document.getElementById("predict-btn").addEventListener("click", predict);
});
