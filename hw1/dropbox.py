import urllib.request
print("Start Downloading model..")
    
try:
    # model24
    url = "https://www.dropbox.com/s/uks1rzq3tn3m7zf/model24_best.h5?dl=1"  # dl=1 is important
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close() 
    with open("./models/rnn_model24_d.h5", "wb") as f :
        f.write(data)

    #model27
    url = "https://www.dropbox.com/s/8zf6gyza8wduwnl/model27_best.h5?dl=1"
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close() 
    with open("./models/rnn_model27_d.h5", "wb") as f :
        f.write(data)
    print("End downloading.")
except:
    print("Download error.")