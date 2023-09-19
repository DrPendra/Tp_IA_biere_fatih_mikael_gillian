document.addEventListener("DOMContentLoaded", function() {
    document.getElementById('btn-predict').addEventListener('click', function (e) {
        e.preventDefault();
        // Récupérez les données du formulaire
        const formData = {}
        datas = document.getElementsByClassName('input-form')

        

        error = false
        for(i=0;i<datas.length;i++){
            if(!datas[i].value.match(/^[0-9]+\.?[0-9]*$/g)){
                datas[i].parentNode.style.border = "2px solid red"
                error = true
            }
            else datas[i].parentNode.style.border = "1px solid #ccc"

            formData[ datas[i].getAttribute("indication")] = datas[i].value
        }

        if(error){
            document.getElementById('error-form').style.display = "inline-block"
            document.getElementById('prediction').style.display = "none"
            return
        }
        else {
            document.getElementById('error-form').style.display = "none"
        }

        const xhr = new XMLHttpRequest();
    
        xhr.open('POST', 'query', true);
    
        xhr.onload = function () {
            if (xhr.status === 200) {
                jsonResp = JSON.parse(xhr.responseText)
                document.getElementById('prediction').style.display = "block"
                document.getElementById('predict-abv').children[1].textContent = jsonResp["abv"]
                document.getElementById('predict-ibu').children[1].textContent = jsonResp["ibu"]
                document.getElementById('error-server').style.display = "none"
            } else {
                document.getElementById('prediction').style.display = "none"
                document.getElementById('error-server').style.display = "inline-block"
                console.error('Erreur lors de la requête.');
            }
        };
    
        // Envoyez les données du formulaire
        xhr.setRequestHeader("Content-Type","application/json")
        xhr.send(JSON.stringify(formData));
    });


    
});