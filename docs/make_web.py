import os
import shutil

# %%
web_path = "/cse/web/research/cancertech/"
files = ["installation", "tutorial"]
img_dirs = ["tutorial_img", "img_sedeen"]
# %%  Create README file for the Github Repository

# The readme file is a combination (concatenation) of all MD files in this folder
readme = open("intro.md", "r").read() + "\n"

for f in files:
    print("Reading:", f)
    readme += open(f + ".md", "r").read() + "\n"
    
for img_dir in img_dirs:
    readme = readme.replace(img_dir, "docs/%s" % img_dir)

open("../README.md", "w").write(readme)

# %% Create HTML for each of the MD files (except intro.md)
header = open("src/header.html", "r").read()
footer = open("src/footer.html", "r").read()

tempfile = "tmp.md"
for f in files:
    content = open(f + ".md", "r").read()
    content = "\n\n".join([header % f.capitalize(), content, footer])
    open(tempfile, "w").write(content)
    os.system('pandoc %s -s -o %s.html --metadata pagetitle="%s"' % (tempfile, f, f))

os.remove(tempfile)

# %% Move the HTML files into the website server
if os.path.exists(web_path):
    for f in files:
        html_name = f + ".html"
        shutil.move(html_name, os.path.join(web_path, html_name))
    os.system("chmod 775 %s/*.html" % web_path)
        
    for img_dir in img_dirs:
        for imgname in os.listdir(img_dir):
            shutil.copy2(os.path.join(img_dir, imgname), os.path.join(web_path, img_dir, imgname))
        os.system("chmod 775 %s/%s/*" % (web_path, img_dir))

else:
    print("Unable to detect the website path")
