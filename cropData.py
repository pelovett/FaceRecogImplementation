import sys, os


def main():
    indir = os.path.dirname(os.path.realpath(__file__))
    for root, dirs, filenames in os.walk(indir):
        cur_root = root
        for f in filenames:
            if f[-4:] == ".pgm":
                fix(f, cur_root)
    return


def fix(FILE_NAME, FILE_PATH):
    if not os.path.exists("ImageData"):
        os.makedirs("ImageData")

    #print(os.path.dirname(FILE_PATH))
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))

    myF = open(FILE_PATH+"/"+FILE_NAME, 'r+b')
    newF = open(CUR_DIR+"/ImageData/x"+FILE_NAME, 'w+b')

    image = myF.read()

    by = ["0x50", "0x35", "0x0A", "0x31", "0x36", "0x38", "0x20", "0x31", "0x36", "0x38", "0x0A", "0x32", "0x35", "0x35", "0x0A"]
    by = bytearray(int(x, 16) for x in by)
    newF.write(by)
    newF.write(image[(2*8):])
    myF.close()
    newF.close()
    return

if __name__ == "__main__":
    main()
