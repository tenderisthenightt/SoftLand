// app.js
const canvas = document.getElementById("jsCanvas");
const ctx = canvas.getContext("2d");
const brush = document.getElementById("jsBrush");
const erase = document.getElementById("jsErase");
const submitButton = document.getElementById("jsSubmitButton");
const submitButton2 = document.getElementById("jsSubmitButton2");

// send image
// submitButton.addEventListener("click", () => {
//     const dataURI = canvas.toDataURL();
//     fetch('/image-similarity', {
//         method: 'POST',
//         body: JSON.stringify({ image: dataURI }),
//         headers: { 'Content-Type': 'application/json' },
//     })
//         .then(response => response.json())
//         .then(data => {
//             console.log('Success:', data);
//         })
//         .catch((error) => {
//             console.error('Error:', error);
//         });
// });

// submitButton.addEventListener("click", () => {
//     const dataURI = canvas.toDataURL();
//     // const imageBlob = dataURIToBlob(dataURI); // Convert the dataURI to a Blob object
//     const formData = new FormData();
//     formData.append('file', dataURLToFile(dataURI, 'image.png')); // Add the Blob object to the form data with a file name
//     const form = document.getElementById("file");
//     form.submit();
// });

submitButton.addEventListener("click", () => {
    const dataURI = canvas.toDataURL();
    const formData = new FormData();
    formData.append('file', dataURLToFile(dataURI, 'image.png'));
    formData.append('p_path', p_path);
    formData.append('sim', sim);

    fetch("/image-similarity", {
        method: "POST",
        body: formData
    })
    .then(response => response.text())
    .then(result => {
        document.getElementById("result").innerHTML = result;
    })
    .catch(error => {
        console.error("Error:", error);
    });
});
    ``

submitButton2.addEventListener("click", () => {
    const dataURI = canvas.toDataURL();
    const imageBlob = dataURIToBlob(dataURI); // Convert the dataURI to a Blob object
    const formData = new FormData();
    formData.append('image', imageBlob, 'image.png'); // Add the Blob object to the form data with a file name
    fetch('/predict', {
        method: 'POST', 
        body: formData, // Use the formData instead of JSON
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});

// Helper function to convert dataURI to Blob
function dataURIToBlob(dataURI) {
    const byteString = atob(dataURI.split(',')[1]);
    const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];
    const arrayBuffer = new ArrayBuffer(byteString.length);
    const int8Array = new Uint8Array(arrayBuffer);
    for (let i = 0; i < byteString.length; i += 1) {
        int8Array[i] = byteString.charCodeAt(i);
    }
    return new Blob([arrayBuffer], { type: mimeString });
}


// Brush option
const INITIAL_COLOR = "#2c2c2c";
const INITIAL_LINEWIDTH = 5.0;
const CANVAS_SIZE = 500;

// canvas option
ctx.strokeStyle = INITIAL_COLOR;
ctx.fillStyle = INITIAL_COLOR;
ctx.lineWidth = INITIAL_LINEWIDTH;
canvas.width = CANVAS_SIZE;
canvas.height = CANVAS_SIZE;

const MODE_BUTTON = [brush, erase];
let mode = brush;
let painting = false;

function startPainting() { painting = true; }
function stopPainting() { painting = false; }

// mousemove option
function onMouseMove(event) {
    // get the current size of the canvas
    let width = canvas.width;
    let height = canvas.height;

    // get the position of the mouse relative to the canvas
    let x = event.offsetX;
    let y = event.offsetY;

    // Scale the mouse coordinates based on the current canvas size
    x = x * width / canvas.offsetWidth;
    y = y * height / canvas.offsetHeight;

    ctx.lineWidth = 3.5;
    if (mode === brush) {
        if (!painting) {
            ctx.beginPath();
            ctx.moveTo(x, y);
        }
        else {
            ctx.lineTo(x, y);
            ctx.stroke();
        }
    }
    // else if(mode === erase){
    //     if(painting) {
    //         ctx.clearRect(x-ctx.lineWidth/2, y-ctx.lineWidth/2, ctx.lineWidth, ctx.lineWidth);
    //     }
    // }
}

function handleModeChange(event) {
    mode = event.target;
    // Button Highlight
    for (i = 0; i < MODE_BUTTON.length; i++) {
        var button = MODE_BUTTON[i];
        if (button === mode) {
            button.style.backgroundColor = "skyblue";
        }
        else {
            button.style.backgroundColor = "white";
        }
    }
}


// All Remove Bts

jsAllremove.addEventListener("click", () => ctx.clearRect(0, 0, canvas.width, canvas.height));

if (canvas) {
    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("mousedown", startPainting);
    canvas.addEventListener("mouseup", stopPainting);
    canvas.addEventListener("mouseleave", stopPainting);
}

MODE_BUTTON.forEach(mode => mode.addEventListener("click", handleModeChange)
);