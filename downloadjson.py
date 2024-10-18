import urllib, json
import urllib.request

#change the url for each dataset, there are three links -> train, test and validation dateset
url = "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json"

response = urllib.request.urlopen(url)

data = json.loads(response.read())

#input file name of .json
with open("train.json", "w") as outfile:
    json.dump(data, outfile, indent=2)
outfile.close()