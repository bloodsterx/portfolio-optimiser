const predictButton = document.getElementById("predictButton");
if (predictButton) {
    predictButton.addEventListener("click", () => {
        const ticker = document.getElementById("stockTicker").value;

        fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ ticker: ticker })
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