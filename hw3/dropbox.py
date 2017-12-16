import urllib.request
print("Start Downloading model..")
    
try:
    #model.ckpt.data-00000-of-00001
    url = "https://www.dropbox.com/s/trdi2ylwl53jcd6/model.ckpt.data-00000-of-00001?dl=1"
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close() 
    with open("./model_best_dqn/model.ckpt.data-00000-of-00001", "wb") as f :
        f.write(data)
    print("End downloading.")
except:
    print("Download error.")