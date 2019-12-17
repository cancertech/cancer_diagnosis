import os
import shutil
# %%
web_path = "/cse/web/research/cancertech/"
files = ["installation", "tutorial"]
# %%  Create README file for the Github Repository

# The readme file is a combination (concatenation) of all MD files in this folder
readme = open("../README.md", "w")
readme.write(open("intro.md", "r").read() + "n")

for f in files:
    readme.write(open(f + ".md", "r").read() + "n")

# %% Create HTML for each of the MD files (except intro.md)
header = open("header.html", "r").read()
footer = open("footer.html", "r").read()

tempfile = "tmp.md"
for f in files:
    content = open(f + ".md", "r").read()
    content = "\n\n".join([header % f.capitalize(), content, footer])
    open(tempfile, "w").write(content)
    os.system("pandoc %s -s -o %s.html" % (tempfile, f))
    

os.remove(tempfile)
# %% Move the HTML files into the website server
if os.path.exists(web_path):
    for f in files:
        html_name = f + ".html"
        shutil.move(html_name, os.path.join(web_path, html_name))
else:
    print("Unable to detect the website path")