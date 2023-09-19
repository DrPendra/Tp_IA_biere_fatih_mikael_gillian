import re

def getHtmlTemplate():
    with open("App/templates/index.html") as f:
        return f.read()

def getHtmlFile():
    f = getHtmlTemplate()
    x = re.sub("({#[\w\ ]*#})","marcel", f)
    return x

file = getHtmlFile()
print(file)