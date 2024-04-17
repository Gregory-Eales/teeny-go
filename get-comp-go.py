


# http://www.yss-aya.com/cgos/9x9/archives/9x9_2024_02.tar.bz2

# need to download all the tar files from the above link and unzip them
# from 2015-11 to 2024-02
# save into data/comp-go-sgf/

link = "http://www.yss-aya.com/cgos/9x9/archives/9x9_{}_{}.tar.bz2"

import requests
import os
import tqdm

for year in range(2015, 2024):
    
    print("pulling data for year:", year)
    for month in tqdm.tqdm(range(1, 13)):
        if year == 2015 and month < 11:
            continue
        if year == 2024 and month > 2:
            continue
        if year == 2024 and month == 2:
            break
        r = requests.get(link.format(year, str(month).zfill(2)))
        open('data/computer-go-sgf/9x9_{}_{}.tar.bz2'.format(year, str(month).zfill(2)), 'wb').write(r.content)
        os.system('tar -xvf data/computer-go-sgf/9x9_{}_{}.tar.bz2 -C data/computer-go-sgf/'.format(year, str(month).zfill(2)))
        #os.system('rm data/computer-go-sgf/9x9_{}_{}.tar.bz2'.format(year, str(month).zfill(2)))
        print("Completed", year, month)