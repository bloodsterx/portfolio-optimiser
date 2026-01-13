const magicButton = document.getElementById("magicButton");
if (magicButton) {
    magicButton.addEventListener("click", () => {
        fetch("/get_data")
        .then(response => response.json())
        .then(data => {
            document.getElementById("responseArea").innerText = data.message
        });
    });
}

const greetButton = document.getElementById("greetButton");
if (greetButton) {
    greetButton.addEventListener("click", () => {
        const name = document.getElementById("userName").value;
        console.log("yippe1");

        fetch("/greet", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ name: name }) // send to the flask backend
        })
        .then(response => response.json())
        .then(data => {
            console.log(data)
            document.getElementById("greetingResponse").innerText = data.greeting;
            console.log("Received Username Data Successfully!");
        });
    });
}
const testButton = document.getElementById("testButton");
if (testButton) {
    testButton.addEventListener("click", () => {
        console.log("test!");
    });

}

const testButtonPredict = document.getElementById("testButtonPredict");
if (testButtonPredict) {
    testButtonPredict.addEventListener("click", () => {
        console.log("testPredict!");
    });
}

