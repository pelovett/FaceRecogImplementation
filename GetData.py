import os
import urllib.request
import zipfile

def main():
  dl_link = "http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip"
  dir_path = os.path.dirname(os.path.realpath(__file__))
  getunzipped(dl_link, dir_path+"/data/")

def getunzipped(theurl, thedir):
  name = os.path.join(thedir, 'temp.zip')
  name, hdrs = urllib.request.urlretrieve(theurl, name)
  try:
    z = zipfile.ZipFile(name, 'r')
  except:
    print("Bad zipfile (from %r):" % (theurl))
    return
  z.extractall(thedir)
  z.close()
  os.unlink(name)

if __name__ == "__main__":
  main()
