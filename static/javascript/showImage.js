const imageInput = document.querySelector("#imageInput");
const submitButton = document.querySelector("#submitButton");
var uploaded_image = "";

submitButton.disabled = true;

imageInput.addEventListener("change", function(){
    const reader = new FileReader();
    submitButton.disabled = false;
    reader.addEventListener("load", () => {
        uploaded_image = reader.result;
        document.querySelector("#displayImg").style.backgroundImage = `url(${uploaded_image})`;
        document.getElementById("revealText").style.display = "none";
    })
    reader.readAsDataURL(this.files[0]);
})