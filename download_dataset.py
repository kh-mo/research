import os
import requests

if __name__=="__main__":
    print("Beginning file download from \"http://mattmahoney.net/dc/text8.zip\"")
    url = 'http://mattmahoney.net/dc/text8.zip'
    resp = requests.get(url)
    if not os.path.exists(os.getcwd()+"/data"):
        print('Creating new directory', os.getcwd()+"/data")
        os.makedirs(os.getcwd()+"/data")
    zfile = open(os.getcwd() + "/data/text8.zip", 'wb')
    zfile.write(resp.content)
    zfile.close
    print("Finishing download")

