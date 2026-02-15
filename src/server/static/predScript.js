const modelSelectionForm = document.getElementById("modelForm");

// prevent submitting the 'default' option
if (modelSelectionForm) {
    modelSelectionForm.addEventListener("submit", (e) => {
        e.preventDefault();
    })
};


const predictButton = document.getElementById("predictButton");
if (predictButton) {
    predictButton.addEventListener("click", () => {
        const tickerInput = document.getElementById("stockTicker");
        const modelSelect = document.getElementById("model-select");

        if (!modelSelect.checkValidity()) {
            modelSelect.reportValidity();
            console.log("report validity");
            return;
        }

        const payload = {
            ticker: tickerInput.value,
            model: modelSelect.value
        };

        fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        })
        .then(response => response.json()) // if i wrap in braces, is that returning a function?
        .then(data => {
            console.log(data)
            document.getElementById("stockForecastResponse").innerText = data.prediction;
        })
        .catch(error => {
            console.log(error);
        })
    });
}