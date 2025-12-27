// static/script.js

document.getElementById("magicButton").addEventListener("click", () => {
    fetch("/get_data")
    .then(response => response.json())
    .then(data => {
        document.getElementById("responseArea").innerText = data.message
    });
});