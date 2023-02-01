// app.js
const canvas = document.getElementById("jsCanvas");
const ctx = canvas.getContext("2d");
const brush = document.getElementById("jsBrush");
const erase = document.getElementById("jsErase");
const submitButton = document.getElementById("jsSubmitButton");

// send image
submitButton.addEventListener("click", () => {
    const dataURI = canvas.toDataURL();
    fetch('/process-image', {
        method: 'POST',
        body: JSON.stringify({ image: dataURI }),
        headers: { 'Content-Type': 'application/json' },
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});

const INITIAL_COLOR = "#2c2c2c";
const INITIAL_LINEWIDTH = 5.0;
const CANVAS_SIZE = 500;

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
    if(mode === brush){
        if(!painting) {
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
    for(i = 0 ; i < MODE_BUTTON.length ; i++){
        var button = MODE_BUTTON[i];
        if(button === mode){
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