// static/script.js

document.getElementById("magicButton").addEventListener("click", () => {
    fetch("/get_data")
    .then(response => response.json())
    .then(data => {
        document.getElementById("responseArea").innerText = data.message
    });
});

document.getElementById("greetButton").addEventListener("click", () => {
    const name = document.getElementById("userName").value;

    fetch("/greet", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ name: name }) // send to the flask backend
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("greetingResponse").innerText = data.lebron;
        console.log("Received Username Data Successfully!");
    });
});