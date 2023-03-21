const imageInput = document.querySelector("#imageInput");
var uploaded_image = "";

imageInput.addEventListener("change", function(){
    const reader = new FileReader();
    reader.addEventListener("load", () => {
        uploaded_image = reader.result;
        document.querySelector("#displayImg").style.backgroundImage = `url(${uploaded_image})`;
        document.getElementById("revealText").style.display = "none";
    })
    reader.readAsDataURL(this.files[0]);
})