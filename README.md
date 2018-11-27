# research

### step 1
download text8.txt data
```
> python download_dataset.py

Beginning file download from "http://mattmahoney.net/dc/text8.zip"
Creating new directory "workspace + /data"
Finishing download
```

### step 2
preprocessing text8.txt data
```
> python preprocessing.py --window=3

Unzip text8.zip
Finish unzip text8.zip
Read text8
Finish making voca_count
Save tokenized_data.pickle
Save voca_count
```
